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
import mmap
import os
import pathlib
import select
import struct
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

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
GPIO_BUTTON_PIN = 16
INPUT_EVENT_STRUCT = struct.Struct("llHHI")


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

    def poll_touched(self) -> bool:
        if not self.fds:
            return False
        touched = False
        now = time.monotonic()
        try:
            ready, _, _ = select.select(self.fds, [], [], 0.0)
        except Exception:
            return False
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
                if event_type == EV_KEY and event_code == BTN_TOUCH and event_value == 1:
                    touched = True
                elif (
                    event_type == EV_ABS
                    and event_code in (ABS_PRESSURE, ABS_MT_POSITION_X, ABS_MT_POSITION_Y)
                    and event_value > 0
                ):
                    touched = True
        if touched and (now - self.last_touch_ts) >= self.debounce_s:
            self.last_touch_ts = now
            return True
        return False

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

    picam = Picamera2()
    target_fps = max(args.fps, 1.0)
    frame_duration_us = int(1_000_000 / target_fps)
    config = picam.create_video_configuration(
        main={"size": (cam_w, cam_h), "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_us, frame_duration_us)},
        buffer_count=4,
        queue=True,
    )
    picam.configure(config)

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
            base_payload = encode_framebuffer_payload(base_image, pixel_format)
            presenter.present(base_payload)
            while True:
                frame_begin = time.monotonic()
                if touch_monitor.poll_touched():
                    print("DEBUG input: LCD touch detected")
                if button_monitor.poll_pressed():
                    print(f"DEBUG input: GPIO{GPIO_BUTTON_PIN} button press detected")
                frame = picam.capture_array("main")
                preview = prepare_preview_frame(frame, preview_w, preview_h, args.rotate)
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
        picam.stop()
        picam.close()


if __name__ == "__main__":
    sys.exit(main())
