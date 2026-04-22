#!/usr/bin/env python3
"""
raspycam.py - minimal digicam live preview for Raspberry Pi framebuffer TFTs.

Current feature set:
- Continuous live preview from Camera Module 3 (via picamera2)
- Display output to framebuffer LCD (/dev/fb*)
- Pixel format options for common SPI TFT drivers
"""

from __future__ import annotations

import argparse
from array import array
import fcntl
import json
import mmap
import os
import pathlib
import re
import select
import shutil
import struct
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw

FBIOGET_VSCREENINFO = 0x4600
FBIOPAN_DISPLAY = 0x4606
FBIO_WAITFORVSYNC = 0x4680
FB_VAR_SCREENINFO_SIZE = 160
FB_YOFFSET_OFFSET = 20
EV_KEY = 0x01
EV_ABS = 0x03
BTN_TOUCH = 0x014A
ABS_PRESSURE = 0x18
ABS_MT_POSITION_X = 0x35
ABS_MT_POSITION_Y = 0x36
ABS_MT_TRACKING_ID = 0x39
GPIO_BUTTON_PIN = 16
INPUT_EVENT_STRUCT = struct.Struct("llHHI")
WB_PROFILE_PATH = pathlib.Path(__file__).with_name("white_balance_gains.json")
PHOTO_DIR = pathlib.Path(__file__).resolve().parent / "DCIM"
PHOTO_PREFIX = "lmr_"
# Picamera2 "RGB888" arrays can still arrive in BGR byte order on some stacks.
WB_REFERENCE_FRAME_IS_BGR = True
# WB stabilizers to avoid extreme color casts from noisy reference frames.
WB_GAIN_BLEND = 0.55
WB_MIN_CHANNEL_MEAN = 32.0
WB_MIN_GAIN = 0.3
WB_MAX_GAIN = 3.0


def parse_virtual_size(fb_name: str) -> Optional[Tuple[int, int]]:
    size_file = f"/sys/class/graphics/{fb_name}/virtual_size"
    if not os.path.exists(size_file):
        return None
    try:
        with open(size_file, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        width_str, height_str = raw.split(",")
        return int(width_str), int(height_str)
    except Exception:
        return None


def read_int_sysfs(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None


def read_bits_per_pixel(fb_name: str) -> Optional[int]:
    return read_int_sysfs(f"/sys/class/graphics/{fb_name}/bits_per_pixel")


def read_channel_offset(fb_name: str, channel: str) -> Optional[int]:
    path = f"/sys/class/graphics/{fb_name}/{channel}"
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            # Format is usually "offset,length,msb_right".
            raw = f.read().strip()
        parts = raw.split(",")
        return int(parts[0])
    except Exception:
        return None


def list_framebuffers() -> List[str]:
    return sorted(str(p) for p in pathlib.Path("/dev").glob("fb*"))


def pick_framebuffer(preferred_fb: str) -> str:
    if preferred_fb and preferred_fb.lower() != "auto":
        return preferred_fb
    available = list_framebuffers()
    if "/dev/fb1" in available:
        return "/dev/fb1"
    if "/dev/fb0" in available:
        return "/dev/fb0"
    return "/dev/fb1"


def load_base_image(base_image_path: str, disp_w: int, disp_h: int) -> np.ndarray:
    try:
        img = Image.open(base_image_path).convert("RGB")
    except Exception as exc:
        raise ValueError(f"Failed to open base image '{base_image_path}': {exc}") from exc
    if img.size != (disp_w, disp_h):
        img = img.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def rgb888_to_rgb565_bytes(frame: np.ndarray, *, little_endian: bool, bgr: bool) -> bytes:
    rgb = frame
    if bgr:
        r = rgb[:, :, 2].astype(np.uint16)
        g = rgb[:, :, 1].astype(np.uint16)
        b = rgb[:, :, 0].astype(np.uint16)
    else:
        r = rgb[:, :, 0].astype(np.uint16)
        g = rgb[:, :, 1].astype(np.uint16)
        b = rgb[:, :, 2].astype(np.uint16)

    rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
    if little_endian:
        return rgb565.astype("<u2", copy=False).tobytes()
    return rgb565.astype(">u2", copy=False).tobytes()


def rgb888_to_xrgb8888_bytes(frame: np.ndarray) -> bytes:
    h, w = frame.shape[:2]
    out = np.empty((h, w, 4), dtype=np.uint8)
    out[:, :, 0] = frame[:, :, 2]  # B
    out[:, :, 1] = frame[:, :, 1]  # G
    out[:, :, 2] = frame[:, :, 0]  # R
    out[:, :, 3] = 0x00            # X
    return out.tobytes()


def encode_framebuffer_payload(frame: np.ndarray, pixel_format: str) -> bytes:
    if pixel_format == "rgb565le":
        return rgb888_to_rgb565_bytes(frame, little_endian=True, bgr=False)
    if pixel_format == "rgb565be":
        return rgb888_to_rgb565_bytes(frame, little_endian=False, bgr=False)
    if pixel_format == "bgr565le":
        return rgb888_to_rgb565_bytes(frame, little_endian=True, bgr=True)
    if pixel_format == "xrgb8888":
        return rgb888_to_xrgb8888_bytes(frame)
    raise ValueError(f"Unsupported pixel format: {pixel_format}")


def pixel_format_bytes_per_pixel(pixel_format: str) -> int:
    if pixel_format == "xrgb8888":
        return 4
    return 2


def infer_pixel_format(fb_name: str) -> str:
    bpp = read_bits_per_pixel(fb_name)
    if bpp == 32:
        return "xrgb8888"
    if bpp == 16:
        red_offset = read_channel_offset(fb_name, "red")
        blue_offset = read_channel_offset(fb_name, "blue")
        if red_offset is not None and blue_offset is not None:
            if red_offset < blue_offset:
                return "bgr565le"
            return "rgb565le"
    return "rgb565le"


def parse_size(size_text: str) -> Tuple[int, int]:
    try:
        w, h = [int(v) for v in size_text.lower().split("x", 1)]
    except Exception as exc:
        raise ValueError("Invalid size format. Use WIDTHxHEIGHT, e.g. 480x320") from exc
    if w <= 0 or h <= 0:
        raise ValueError("Size values must be positive")
    return w, h


def prepare_preview_frame(frame: np.ndarray, preview_w: int, preview_h: int, rotate: int) -> np.ndarray:
    img = frame
    if rotate:
        # np.rot90 rotates counter-clockwise.
        if rotate == 90:
            img = np.rot90(img, 1)
        elif rotate == 180:
            img = np.rot90(img, 2)
        elif rotate == 270:
            img = np.rot90(img, 3)
    if img.shape[1] != preview_w or img.shape[0] != preview_h:
        pil_img = Image.fromarray(img, mode="RGB")
        img = np.asarray(
            pil_img.resize((preview_w, preview_h), Image.Resampling.BILINEAR),
            dtype=np.uint8,
        )
    return img


def estimate_white_balance_stats(frame: np.ndarray) -> Tuple[float, float, float, float, float]:
    # Gray-world estimate: map red/blue averages toward green average.
    frame_f = frame.astype(np.float32)
    if frame_f.ndim != 3 or frame_f.shape[2] < 3:
        raise ValueError(f"Unexpected frame shape for WB estimation: {frame.shape}")
    h, w = frame_f.shape[:2]
    x0 = int(w * 0.2)
    x1 = int(w * 0.8)
    y0 = int(h * 0.2)
    y1 = int(h * 0.8)
    sample = frame_f[y0:y1, x0:x1, :3]
    luma = sample.mean(axis=2)
    valid_mask = (luma > 8.0) & (luma < 247.0)
    if int(valid_mask.sum()) < 1024:
        # Fall back to the crop when dynamic masking removes too many pixels.
        valid = sample.reshape(-1, 3)
    else:
        valid = sample[valid_mask]
    if WB_REFERENCE_FRAME_IS_BGR:
        r_values = valid[:, 2]
        g_values = valid[:, 1]
        b_values = valid[:, 0]
    else:
        r_values = valid[:, 0]
        g_values = valid[:, 1]
        b_values = valid[:, 2]

    def robust_mean(channel_values: np.ndarray) -> float:
        # Trim outliers so specular highlights/shadows do not dominate WB.
        lo = float(np.percentile(channel_values, 5.0))
        hi = float(np.percentile(channel_values, 95.0))
        trimmed = channel_values[(channel_values >= lo) & (channel_values <= hi)]
        if trimmed.size == 0:
            return float(channel_values.mean())
        return float(trimmed.mean())

    r_mean = robust_mean(r_values)
    g_mean = robust_mean(g_values)
    b_mean = robust_mean(b_values)
    eps = 1e-6
    r_gain_raw = g_mean / max(r_mean, WB_MIN_CHANNEL_MEAN, eps)
    b_gain_raw = g_mean / max(b_mean, WB_MIN_CHANNEL_MEAN, eps)
    r_gain = 1.0 + ((r_gain_raw - 1.0) * WB_GAIN_BLEND)
    b_gain = 1.0 + ((b_gain_raw - 1.0) * WB_GAIN_BLEND)
    # Picamera2 docs indicate ColourGains values in [0, 32].
    r_gain = min(max(r_gain, WB_MIN_GAIN), WB_MAX_GAIN)
    b_gain = min(max(b_gain, WB_MIN_GAIN), WB_MAX_GAIN)
    return r_gain, b_gain, r_mean, g_mean, b_mean


def estimate_white_balance_gains(frame: np.ndarray) -> Tuple[float, float]:
    r_gain, b_gain, _, _, _ = estimate_white_balance_stats(frame)
    return r_gain, b_gain


def run_still_capture_command(cmd: List[str]) -> None:
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    except subprocess.CalledProcessError as exc:
        output = (exc.stdout or "").strip()
        if output:
            raise RuntimeError(output.splitlines()[-1]) from exc
        raise RuntimeError(str(exc)) from exc


def draw_status_message(frame: np.ndarray, text: str) -> np.ndarray:
    image = Image.fromarray(frame, mode="RGB")
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad_x = 10
    pad_y = 6
    box_w = text_w + 2 * pad_x
    box_h = text_h + 2 * pad_y
    box_x = max((image.width - box_w) // 2, 0)
    box_y = max(image.height - box_h - 10, 0)
    draw.rectangle(
        (box_x, box_y, min(box_x + box_w, image.width), min(box_y + box_h, image.height)),
        fill=(0, 0, 0),
    )
    draw.text((box_x + pad_x, box_y + pad_y), text, fill=(255, 255, 255))
    return np.asarray(image, dtype=np.uint8)


def load_white_balance_profile(path: pathlib.Path) -> Optional[Tuple[float, float]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        r_gain = float(raw.get("r_gain"))
        b_gain = float(raw.get("b_gain"))
        # Keep within the practical range we apply elsewhere.
        r_gain = min(max(r_gain, 0.1), 8.0)
        b_gain = min(max(b_gain, 0.1), 8.0)
        return r_gain, b_gain
    except Exception:
        return None


def save_white_balance_profile(path: pathlib.Path, r_gain: float, b_gain: float) -> None:
    payload = {
        "r_gain": float(r_gain),
        "b_gain": float(b_gain),
        "saved_at_unix": time.time(),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def next_photo_path(photo_dir: pathlib.Path) -> pathlib.Path:
    photo_dir.mkdir(parents=True, exist_ok=True)
    max_index = 0
    pattern = re.compile(rf"^{re.escape(PHOTO_PREFIX)}(\d+)\.png$")
    for path in photo_dir.glob(f"{PHOTO_PREFIX}*.png"):
        match = pattern.match(path.name)
        if not match:
            continue
        max_index = max(max_index, int(match.group(1)))
    return photo_dir / f"{PHOTO_PREFIX}{max_index + 1:04d}.png"


def trigger_autofocus(picam) -> None:
    try:
        import libcamera  # type: ignore

        picam.set_controls(
            {
                "AfMode": libcamera.controls.AfModeEnum.Auto,
                "AfTrigger": libcamera.controls.AfTriggerEnum.Start,
            }
        )
        # Give AF a short window to settle before capture.
        time.sleep(0.6)
    except Exception:
        # Best effort only; not all modules/drivers expose AF controls.
        pass


def resize_frame_to_display(frame: np.ndarray, disp_w: int, disp_h: int) -> np.ndarray:
    if frame.shape[1] == disp_w and frame.shape[0] == disp_h:
        return frame
    pil_img = Image.fromarray(frame, mode="RGB")
    return np.asarray(pil_img.resize((disp_w, disp_h), Image.Resampling.BILINEAR), dtype=np.uint8)


def pick_max_still_size(picam, fallback_size: Tuple[int, int]) -> Tuple[int, int]:
    max_size = picam.camera_properties.get("PixelArraySize") if hasattr(picam, "camera_properties") else None
    if (
        isinstance(max_size, tuple)
        and len(max_size) == 2
        and int(max_size[0]) > 0
        and int(max_size[1]) > 0
    ):
        return int(max_size[0]), int(max_size[1])
    return fallback_size


def capture_still_photo_to_file(
    max_size: Tuple[int, int],
    current_wb_gains: Optional[Tuple[float, float]],
    output_path: pathlib.Path,
) -> None:
    still_cli = shutil.which("rpicam-still") or shutil.which("libcamera-still")
    if still_cli is None:
        raise RuntimeError("Neither rpicam-still nor libcamera-still is installed.")

    cmd = [
        still_cli,
        "-n",
        "--width",
        str(max_size[0]),
        "--height",
        str(max_size[1]),
        "--autofocus-mode",
        "auto",
        "--autofocus-on-capture",
        "--timeout",
        "1200",
        "-o",
        str(output_path),
    ]
    if current_wb_gains is not None:
        r_gain, b_gain = current_wb_gains
        cmd.extend(["--awbgains", f"{r_gain:.6f},{b_gain:.6f}"])
    run_still_capture_command(cmd)


def capture_wb_reference_frame(
    Picamera2,
    reference_size: Tuple[int, int],
    settle_frames: int = 12,
) -> np.ndarray:
    wb_cam = Picamera2()
    try:
        config = wb_cam.create_video_configuration(
            main={"size": reference_size, "format": "RGB888"},
            buffer_count=4,
            queue=True,
        )
        wb_cam.configure(config)
        wb_cam.start()
        wb_cam.set_controls({"AwbEnable": False, "ColourGains": (1.0, 1.0)})
        time.sleep(0.10)
        frame = wb_cam.capture_array("main")
        for _ in range(max(settle_frames, 0)):
            frame = wb_cam.capture_array("main")
        return frame
    finally:
        try:
            wb_cam.stop()
        except Exception:
            pass
        try:
            wb_cam.close()
        except Exception:
            pass


def create_preview_camera(Picamera2, cam_w: int, cam_h: int, frame_duration_us: int):
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (cam_w, cam_h), "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        buffer_count=4,
        queue=True,
    )
    picam.configure(config)
    picam.start()
    return picam


def load_photo_preview(photo_path: pathlib.Path, disp_w: int, disp_h: int) -> np.ndarray:
    with Image.open(photo_path) as captured:
        rgb = captured.convert("RGB")
        resized = rgb.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def read_u32(buf: bytearray, offset: int) -> int:
    return struct.unpack_from("=I", buf, offset)[0]


def write_u32(buf: bytearray, offset: int, value: int) -> None:
    struct.pack_into("=I", buf, offset, value)


def read_var_screeninfo(fd: int) -> Optional[bytearray]:
    info = bytearray(FB_VAR_SCREENINFO_SIZE)
    try:
        fcntl.ioctl(fd, FBIOGET_VSCREENINFO, info, True)
        return info
    except OSError:
        return None


def wait_for_vsync(fd: int) -> bool:
    token = array("I", [0])
    try:
        fcntl.ioctl(fd, FBIO_WAITFORVSYNC, token, True)
        return True
    except OSError:
        return False


def pan_display(fd: int, yoffset: int) -> bool:
    var = read_var_screeninfo(fd)
    if var is None:
        return False
    write_u32(var, FB_YOFFSET_OFFSET, yoffset)
    try:
        fcntl.ioctl(fd, FBIOPAN_DISPLAY, var, True)
        return True
    except OSError:
        return False


class FramebufferPresenter:
    def __init__(
        self,
        fb_file,
        disp_w: int,
        disp_h: int,
        pixel_format: str,
        sync_mode: str,
    ) -> None:
        self.fb_file = fb_file
        self.fd = fb_file.fileno()
        self.disp_w = disp_w
        self.disp_h = disp_h
        self.pixel_bytes = pixel_format_bytes_per_pixel(pixel_format)
        self.frame_bytes = disp_w * disp_h * self.pixel_bytes
        self.fb_size = os.fstat(self.fd).st_size
        self.mm: Optional[mmap.mmap] = None
        if self.fb_size >= self.frame_bytes and self.fb_size > 0:
            try:
                self.mm = mmap.mmap(self.fd, self.fb_size, access=mmap.ACCESS_WRITE)
            except OSError:
                self.mm = None

        self.pageflip_enabled = False
        self.vsync_enabled = False
        self.front_idx = 0
        self.back_idx = 0

        if sync_mode != "none":
            var = read_var_screeninfo(self.fd)
            if var is not None:
                xres_virtual = read_u32(var, 8)
                yres_virtual = read_u32(var, 12)
                supports_pageflip = (
                    xres_virtual >= disp_w
                    and yres_virtual >= disp_h * 2
                    and self.fb_size >= self.frame_bytes * 2
                )
                if sync_mode in ("auto", "pageflip") and supports_pageflip:
                    self.pageflip_enabled = True
                    self.front_idx = 0
                    self.back_idx = 1
                elif sync_mode == "pageflip":
                    print("Requested pageflip sync, but framebuffer does not expose two pages.")

        if sync_mode in ("auto", "vsync", "pageflip"):
            if wait_for_vsync(self.fd):
                self.vsync_enabled = True
            elif sync_mode == "vsync":
                print("Requested vsync sync, but FBIO_WAITFORVSYNC is not supported.")

    def present(self, payload: bytes) -> None:
        if len(payload) != self.frame_bytes:
            raise ValueError(
                f"Payload size mismatch: got {len(payload)}, expected {self.frame_bytes}"
            )

        if self.pageflip_enabled:
            write_offset = self.back_idx * self.frame_bytes
            if self.mm is not None:
                self.mm[write_offset : write_offset + self.frame_bytes] = payload
            else:
                os.pwrite(self.fd, payload, write_offset)
            if self.vsync_enabled:
                wait_for_vsync(self.fd)
            if pan_display(self.fd, self.back_idx * self.disp_h):
                self.front_idx = self.back_idx
                self.back_idx = 1 - self.front_idx
            else:
                # Fallback to a normal write if panning unexpectedly fails.
                self.pageflip_enabled = False
                os.pwrite(self.fd, payload, 0)
            return

        if self.vsync_enabled:
            wait_for_vsync(self.fd)
        # For single-buffer streaming, prefer one contiguous write syscall.
        os.pwrite(self.fd, payload, 0)

    def present_region(self, payload: bytes, x: int, y: int, w: int, h: int) -> None:
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("Invalid region bounds")
        if x + w > self.disp_w or y + h > self.disp_h:
            raise ValueError("Region exceeds display bounds")

        expected_len = w * h * self.pixel_bytes
        if len(payload) != expected_len:
            raise ValueError(
                f"Region payload size mismatch: got {len(payload)}, expected {expected_len}"
            )

        if self.vsync_enabled:
            wait_for_vsync(self.fd)

        row_bytes = w * self.pixel_bytes
        row_stride = self.disp_w * self.pixel_bytes
        page_base = self.front_idx * self.frame_bytes if self.pageflip_enabled else 0

        if x == 0 and w == self.disp_w:
            offset = page_base + y * row_stride
            os.pwrite(self.fd, payload, offset)
            return

        if self.mm is not None:
            for row in range(h):
                src_off = row * row_bytes
                dst_off = page_base + ((y + row) * row_stride) + (x * self.pixel_bytes)
                self.mm[dst_off : dst_off + row_bytes] = payload[src_off : src_off + row_bytes]
            return

        for row in range(h):
            src_off = row * row_bytes
            dst_off = page_base + ((y + row) * row_stride) + (x * self.pixel_bytes)
            os.pwrite(self.fd, payload[src_off : src_off + row_bytes], dst_off)

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()

    @property
    def has_sync(self) -> bool:
        return self.pageflip_enabled or self.vsync_enabled


def discover_touch_devices() -> List[str]:
    devices: List[str] = []
    input_class = pathlib.Path("/sys/class/input")
    if not input_class.exists():
        return devices

    keywords = ("touch", "xpt2046", "ads7846", "goodix", "ft5", "stmpe")
    for event_node in sorted(input_class.glob("event*")):
        dev_name_path = event_node / "device" / "name"
        dev_name = ""
        try:
            dev_name = dev_name_path.read_text(encoding="utf-8").strip().lower()
        except Exception:
            continue
        if any(k in dev_name for k in keywords):
            dev_path = f"/dev/input/{event_node.name}"
            if os.path.exists(dev_path):
                devices.append(dev_path)
    return devices


@dataclass
class TouchInputMonitor:
    fds: List[int]
    paths: List[str]
    last_touch_ts: float = 0.0
    debounce_s: float = 0.2
    last_touch_x: Optional[int] = None
    last_touch_y: Optional[int] = None
    touch_active: bool = False

    @classmethod
    def create(cls) -> "TouchInputMonitor":
        paths = discover_touch_devices()
        fds: List[int] = []
        for path in paths:
            try:
                fds.append(os.open(path, os.O_RDONLY | os.O_NONBLOCK))
            except OSError:
                continue
        return cls(fds=fds, paths=paths)

    def poll_touched(self) -> Optional[Tuple[Optional[int], Optional[int]]]:
        if not self.fds:
            return None
        touched = False
        now = time.monotonic()
        try:
            ready, _, _ = select.select(self.fds, [], [], 0.0)
        except Exception:
            return None
        for fd in ready:
            while True:
                try:
                    chunk = os.read(fd, INPUT_EVENT_STRUCT.size)
                except BlockingIOError:
                    break
                except OSError:
                    break
                if len(chunk) < INPUT_EVENT_STRUCT.size:
                    break
                _, _, event_type, event_code, event_value = INPUT_EVENT_STRUCT.unpack(chunk)
                if event_type == EV_KEY and event_code == BTN_TOUCH:
                    is_down = event_value != 0
                    if is_down and not self.touch_active:
                        touched = True
                    self.touch_active = is_down
                elif event_type == EV_ABS and event_code == ABS_MT_POSITION_X:
                    self.last_touch_x = int(event_value)
                elif event_type == EV_ABS and event_code == ABS_MT_POSITION_Y:
                    self.last_touch_y = int(event_value)
                elif event_type == EV_ABS and event_code == ABS_MT_TRACKING_ID:
                    is_down = event_value != 0xFFFFFFFF
                    if is_down and not self.touch_active:
                        touched = True
                    self.touch_active = is_down
                elif (
                    event_type == EV_ABS
                    and event_code == ABS_PRESSURE
                ):
                    is_down = event_value > 0
                    if is_down and not self.touch_active:
                        touched = True
                    self.touch_active = is_down
        if touched and (now - self.last_touch_ts) >= self.debounce_s:
            self.last_touch_ts = now
            return self.last_touch_x, self.last_touch_y
        return None

    def close(self) -> None:
        for fd in self.fds:
            try:
                os.close(fd)
            except OSError:
                pass
        self.fds.clear()


@dataclass
class GpioButtonMonitor:
    pin: int
    backend: str
    value_fd: Optional[int]
    previous_value: Optional[int]
    active_value: Optional[int]
    exported_here: bool
    button: Optional[object] = None
    pending_presses: int = 0
    last_press_ts: float = 0.0
    debounce_s: float = 0.10

    @classmethod
    def create(cls, pin: int) -> "GpioButtonMonitor":
        # First try gpiozero (modern/reliable on Raspberry Pi OS).
        try:
            from gpiozero import Button  # type: ignore

            monitor = cls(
                pin=pin,
                backend="gpiozero",
                value_fd=None,
                previous_value=None,
                active_value=None,
                exported_here=False,
            )
            button = Button(pin, pull_up=True, bounce_time=0.05)
            monitor.button = button

            def _on_pressed() -> None:
                monitor.pending_presses += 1

            button.when_pressed = _on_pressed
            return monitor
        except Exception:
            pass

        # Fallback: legacy sysfs GPIO.
        gpio_path = pathlib.Path(f"/sys/class/gpio/gpio{pin}")
        exported_here = False
        if not gpio_path.exists():
            try:
                pathlib.Path("/sys/class/gpio/export").write_text(f"{pin}", encoding="utf-8")
                exported_here = True
            except Exception:
                return cls(
                    pin=pin,
                    backend="none",
                    value_fd=None,
                    previous_value=None,
                    active_value=None,
                    exported_here=False,
                )

        try:
            (gpio_path / "direction").write_text("in", encoding="utf-8")
        except Exception:
            pass
        try:
            # Track both edges and infer active level from idle state.
            (gpio_path / "edge").write_text("both", encoding="utf-8")
        except Exception:
            pass

        value_fd: Optional[int] = None
        previous_value: Optional[int] = None
        active_value: Optional[int] = None
        try:
            value_fd = os.open(str(gpio_path / "value"), os.O_RDONLY | os.O_NONBLOCK)
            os.lseek(value_fd, 0, os.SEEK_SET)
            initial = os.read(value_fd, 1)
            if initial:
                previous_value = int(initial)
                active_value = 0 if previous_value == 1 else 1
        except Exception:
            if value_fd is not None:
                try:
                    os.close(value_fd)
                except OSError:
                    pass
            value_fd = None
        backend = "sysfs" if value_fd is not None else "none"
        return cls(
            pin=pin,
            backend=backend,
            value_fd=value_fd,
            previous_value=previous_value,
            active_value=active_value,
            exported_here=exported_here,
        )

    def poll_pressed(self) -> bool:
        if self.backend == "gpiozero":
            if self.pending_presses > 0:
                self.pending_presses -= 1
                return True
            return False

        if self.value_fd is None:
            return False
        try:
            os.lseek(self.value_fd, 0, os.SEEK_SET)
            raw = os.read(self.value_fd, 1)
            if not raw:
                return False
            value = int(raw)
        except Exception:
            return False

        now = time.monotonic()
        previous = self.previous_value
        if self.active_value is None and previous is not None:
            self.active_value = 0 if previous == 1 else 1
        pressed = (
            previous is not None
            and self.active_value is not None
            and previous != value
            and value == self.active_value
            and (now - self.last_press_ts) >= self.debounce_s
        )
        self.previous_value = value
        if pressed:
            self.last_press_ts = now
        return pressed

    def close(self) -> None:
        if self.button is not None:
            try:
                self.button.close()
            except Exception:
                pass
            self.button = None
        if self.value_fd is not None:
            try:
                os.close(self.value_fd)
            except OSError:
                pass
            self.value_fd = None
        if self.exported_here:
            try:
                pathlib.Path("/sys/class/gpio/unexport").write_text(f"{self.pin}", encoding="utf-8")
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Raspberry Pi digicam live preview")
    parser.add_argument(
        "--fb",
        default="auto",
        help="Framebuffer device path (default: auto; prefers /dev/fb1 then /dev/fb0)",
    )
    parser.add_argument(
        "--size",
        default="480x320",
        help="Display size fallback if framebuffer sysfs is unavailable (default: 480x320)",
    )
    parser.add_argument(
        "--camera-size",
        default="420x280",
        help="Camera capture resolution for preview (default: 420x280)",
    )
    parser.add_argument(
        "--base-image",
        default="baseimage.png",
        help="Background image drawn once before preview overlay (default: baseimage.png)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=120.0,
        help="Target preview framerate (default: 120)",
    )
    parser.add_argument(
        "--pixel-format",
        choices=["auto", "rgb565le", "rgb565be", "bgr565le", "xrgb8888"],
        default="bgr565le",
        help="Framebuffer pixel format (default: auto)",
    )
    parser.add_argument(
        "--rotate",
        type=int,
        choices=[0, 90, 180, 270],
        default=0,
        help="Rotate preview output (default: 0)",
    )
    parser.add_argument(
        "--sync-mode",
        choices=["auto", "none", "vsync", "pageflip"],
        default="auto",
        help=(
            "Framebuffer sync strategy: auto prefers pageflip+vsync, "
            "then vsync, then unsynced writes (default: auto)"
        ),
    )
    parser.add_argument(
        "--unsynced-fps",
        type=float,
        default=90.0,
        help=(
            "When no framebuffer sync is available, cap preview updates to this FPS "
            "to reduce visible tearing. Set <=0 to disable (default: 90)"
        ),
    )
    args = parser.parse_args()

    try:
        fallback_w, fallback_h = parse_size(args.size)
        cam_w, cam_h = parse_size(args.camera_size)
    except ValueError as e:
        print(str(e))
        return 2

    fb_path = pick_framebuffer(args.fb)
    if not os.path.exists(fb_path):
        print(f"Framebuffer device not found: {fb_path}")
        available = list_framebuffers()
        if available:
            print(f"Available framebuffer devices: {', '.join(available)}")
        else:
            print("No framebuffer devices were found under /dev/fb*.")
        return 1

    fb_name = os.path.basename(fb_path)
    detected_size = parse_virtual_size(fb_name)
    if detected_size:
        disp_w, disp_h = detected_size
    else:
        disp_w, disp_h = fallback_w, fallback_h

    pixel_format = args.pixel_format
    if pixel_format == "auto":
        pixel_format = infer_pixel_format(fb_name)

    try:
        base_image = load_base_image(args.base_image, disp_w, disp_h)
    except ValueError as e:
        print(str(e))
        return 1

    preview_w = min(cam_w, disp_w)
    preview_h = min(cam_h, disp_h)

    print(f"Framebuffer: {fb_path}")
    print(f"Display size: {disp_w}x{disp_h}")
    print(f"Preview capture size: {cam_w}x{cam_h}")
    print(f"Preview overlay area: {preview_w}x{preview_h} at (0,0)")
    print(f"Base image: {args.base_image}")
    print(f"Pixel format: {pixel_format}")
    print(f"Sync mode: {args.sync_mode}")
    print(f"Unsynced FPS cap: {args.unsynced_fps}")
    print("Starting live preview. Press Ctrl+C to stop.")

    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception as e:
        print(f"picamera2 import failed: {e}")
        print("Install with: sudo apt install -y python3-picamera2")
        return 1

    target_fps = max(args.fps, 1.0)
    frame_duration_us = int(1_000_000 / target_fps)
    picam = Picamera2()
    config = picam.create_video_configuration(
        main={"size": (cam_w, cam_h), "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        buffer_count=4,
        queue=True,
    )
    picam.configure(config)
    max_still_size = pick_max_still_size(picam, fallback_size=(cam_w, cam_h))
    wb_reference_size = (min(640, max_still_size[0]), min(480, max_still_size[1]))

    try:
        picam.start()
        time.sleep(0.5)
    except Exception as e:
        print(f"Failed to start camera preview stream: {e}")
        picam.close()
        return 1

    interval = 1.0 / max(args.fps, 0.1)
    frames = 0
    started = time.monotonic()
    last_stats = started
    presenter: Optional[FramebufferPresenter] = None
    touch_monitor = TouchInputMonitor.create()
    button_monitor = GpioButtonMonitor.create(GPIO_BUTTON_PIN)
    wb_message_until = 0.0
    startup_wb_gains = load_white_balance_profile(WB_PROFILE_PATH)
    current_wb_gains: Optional[Tuple[float, float]] = None

    try:
        with open(fb_path, "r+b", buffering=0) as fb:
            presenter = FramebufferPresenter(
                fb_file=fb,
                disp_w=disp_w,
                disp_h=disp_h,
                pixel_format=pixel_format,
                sync_mode=args.sync_mode,
            )
            if presenter.pageflip_enabled:
                print("Framebuffer sync: pageflip enabled")
            elif presenter.vsync_enabled:
                print("Framebuffer sync: vsync enabled")
            else:
                print("Framebuffer sync: unsynced writes")
                if args.unsynced_fps > 0:
                    effective_fps = min(args.fps, args.unsynced_fps)
                    interval = 1.0 / max(effective_fps, 0.1)
                    print(f"Unsynced fallback: capping update rate to {effective_fps:.1f} fps")
                else:
                    print("Unsynced fallback: fps cap disabled")
            if touch_monitor.fds:
                print(f"Touch input debug enabled on: {', '.join(touch_monitor.paths)}")
            else:
                print("Touch input debug not enabled (no touchscreen event device found or no permission).")
            if button_monitor.backend != "none":
                print(
                    f"GPIO button debug enabled on GPIO{GPIO_BUTTON_PIN} "
                    f"(backend: {button_monitor.backend})."
                )
            else:
                print(
                    f"GPIO button debug not enabled on GPIO{GPIO_BUTTON_PIN} "
                    "(check gpiozero/sysfs permissions and wiring)."
                )
            if startup_wb_gains is not None:
                r_gain, b_gain = startup_wb_gains
                try:
                    picam.set_controls({"AwbEnable": False, "ColourGains": (r_gain, b_gain)})
                    current_wb_gains = (r_gain, b_gain)
                    print(
                        "Loaded white balance profile from disk: "
                        f"ColourGains=({r_gain:.3f}, {b_gain:.3f})"
                    )
                except Exception as e:
                    print(f"Failed to apply saved white balance profile: {e}")
            else:
                print("No saved white balance profile found; using current camera AWB behavior.")
            base_payload = encode_framebuffer_payload(base_image, pixel_format)
            black_payload = encode_framebuffer_payload(
                np.zeros((disp_h, disp_w, 3), dtype=np.uint8),
                pixel_format,
            )
            presenter.present(base_payload)
            while True:
                frame_begin = time.monotonic()
                if button_monitor.poll_pressed():
                    print(f"Button press on GPIO{GPIO_BUTTON_PIN}: starting autofocus...")
                    presenter.present(black_payload)
                    print("Capturing full-resolution photo...")
                    try:
                        photo_path = next_photo_path(PHOTO_DIR)
                        # Release camera fully so CLI still capture can acquire it.
                        picam.stop()
                        picam.close()
                        time.sleep(0.15)
                        capture_still_photo_to_file(
                            max_size=max_still_size,
                            current_wb_gains=current_wb_gains,
                            output_path=photo_path,
                        )
                        print(f"Saved photo: {photo_path}")
                        full_preview = load_photo_preview(photo_path, disp_w=disp_w, disp_h=disp_h)
                        presenter.present(encode_framebuffer_payload(full_preview, pixel_format))
                        time.sleep(2.0)
                    except Exception as capture_err:
                        print(f"Photo capture failed: {capture_err}")
                    finally:
                        # Recreate preview camera after external capture command.
                        picam = create_preview_camera(Picamera2, cam_w, cam_h, frame_duration_us)
                        if current_wb_gains is not None:
                            picam.set_controls({"AwbEnable": False, "ColourGains": current_wb_gains})
                    continue
                frame = picam.capture_array("main")
                touch_pos = touch_monitor.poll_touched()
                if touch_pos is not None:
                    try:
                        touch_x, touch_y = touch_pos
                        presenter.present(black_payload)
                        # Fully release preview stream/device before WB reference still capture.
                        picam.stop()
                        picam.close()
                        time.sleep(0.15)
                        wb_reference_frame = capture_wb_reference_frame(
                            Picamera2=Picamera2,
                            reference_size=wb_reference_size,
                        )
                        r_gain, b_gain, r_mean, g_mean, b_mean = estimate_white_balance_stats(
                            wb_reference_frame
                        )
                        if (r_gain <= 0.11 and b_gain <= 0.11) or (r_gain >= 7.9 and b_gain >= 7.9):
                            raise ValueError(
                                "WB estimate saturated at clamp limits; ignoring unstable reference frame."
                            )
                        current_wb_gains = (r_gain, b_gain)
                        try:
                            save_white_balance_profile(WB_PROFILE_PATH, r_gain, b_gain)
                        except Exception as save_err:
                            print(f"Failed to save white balance profile: {save_err}")
                        wb_message_until = time.monotonic() + 1.0
                        touch_coords = (
                            f"touch=({touch_x}, {touch_y})"
                            if touch_x is not None and touch_y is not None
                            else "touch=(unknown)"
                        )
                        print(
                            "White balance reset from touch: "
                            f"{touch_coords} "
                            f"RGB means=({r_mean:.2f}, {g_mean:.2f}, {b_mean:.2f}) "
                            f"ColourGains=({r_gain:.3f}, {b_gain:.3f}) "
                            f"(saved to {WB_PROFILE_PATH.name})"
                        )
                    except Exception as e:
                        print(f"Failed to reset white balance from touch: {e}")
                    finally:
                        picam = create_preview_camera(Picamera2, cam_w, cam_h, frame_duration_us)
                        if current_wb_gains is not None:
                            picam.set_controls({"AwbEnable": False, "ColourGains": current_wb_gains})
                    continue
                preview = prepare_preview_frame(frame, preview_w, preview_h, args.rotate)
                if time.monotonic() < wb_message_until:
                    preview = draw_status_message(preview, "White Balance Reset")
                payload = encode_framebuffer_payload(preview, pixel_format)
                presenter.present_region(payload, x=0, y=0, w=preview_w, h=preview_h)

                frames += 1
                now = time.monotonic()
                if now - last_stats >= 2.0:
                    elapsed = now - started
                    avg_fps = frames / elapsed if elapsed > 0 else 0.0
                    print(f"Frames: {frames} | avg fps: {avg_fps:.1f}")
                    last_stats = now

                frame_elapsed = time.monotonic() - frame_begin
                sleep_time = interval - frame_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\nStopped by user.")
        return 0
    except PermissionError:
        print(f"Permission denied writing {fb_path}. Try running with sudo.")
        return 1
    except Exception as e:
        print(f"Preview loop failed: {e}")
        return 1
    finally:
        try:
            presenter.close()
        except Exception:
            pass
        touch_monitor.close()
        button_monitor.close()
        try:
            picam.stop()
        except Exception:
            pass
        try:
            picam.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
