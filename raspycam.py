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
import struct
import sys
import time
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

FBIOGET_VSCREENINFO = 0x4600
FBIOPAN_DISPLAY = 0x4606
FBIO_WAITFORVSYNC = 0x4680
FB_VAR_SCREENINFO_SIZE = 160
FB_YOFFSET_OFFSET = 20


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


def fit_cover_numpy(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    src_h, src_w = frame.shape[:2]
    src_ratio = src_w / src_h
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        crop_h = src_h
        crop_w = int(crop_h * dst_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / dst_ratio)

    left = max(0, (src_w - crop_w) // 2)
    top = max(0, (src_h - crop_h) // 2)
    cropped = frame[top : top + crop_h, left : left + crop_w]

    if cropped.shape[1] == width and cropped.shape[0] == height:
        return cropped

    pil_img = Image.fromarray(cropped, mode="RGB")
    resized = pil_img.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


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
    return "rgb565le"


def parse_size(size_text: str) -> Tuple[int, int]:
    try:
        w, h = [int(v) for v in size_text.lower().split("x", 1)]
    except Exception as exc:
        raise ValueError("Invalid size format. Use WIDTHxHEIGHT, e.g. 480x320") from exc
    if w <= 0 or h <= 0:
        raise ValueError("Size values must be positive")
    return w, h


def prepare_preview_frame(frame: np.ndarray, disp_w: int, disp_h: int, rotate: int) -> np.ndarray:
    img = fit_cover_numpy(frame, disp_w, disp_h)
    if rotate:
        # np.rot90 rotates counter-clockwise.
        if rotate == 90:
            img = np.rot90(img, 1)
        elif rotate == 180:
            img = np.rot90(img, 2)
        elif rotate == 270:
            img = np.rot90(img, 3)
        if img.shape[1] != disp_w or img.shape[0] != disp_h:
            pil_img = Image.fromarray(img, mode="RGB")
            img = np.asarray(
                pil_img.resize((disp_w, disp_h), Image.Resampling.BILINEAR),
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
        self.frame_bytes = disp_w * disp_h * pixel_format_bytes_per_pixel(pixel_format)
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

    def close(self) -> None:
        if self.mm is not None:
            self.mm.close()

    @property
    def has_sync(self) -> bool:
        return self.pageflip_enabled or self.vsync_enabled


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
        default="480x320",
        help="Camera capture resolution for preview (default: 480x320)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Target preview framerate (default: 30)",
    )
    parser.add_argument(
        "--pixel-format",
        choices=["auto", "rgb565le", "rgb565be", "bgr565le", "xrgb8888"],
        default="auto",
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
        default=12.0,
        help=(
            "When no framebuffer sync is available, cap preview updates to this FPS "
            "to reduce visible tearing. Set <=0 to disable (default: 12)"
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

    print(f"Framebuffer: {fb_path}")
    print(f"Display size: {disp_w}x{disp_h}")
    print(f"Preview capture size: {cam_w}x{cam_h}")
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
            while True:
                frame_begin = time.monotonic()
                frame = picam.capture_array("main")
                preview = prepare_preview_frame(frame, disp_w, disp_h, args.rotate)
                payload = encode_framebuffer_payload(preview, pixel_format)
                presenter.present(payload)

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
        picam.stop()
        picam.close()


if __name__ == "__main__":
    sys.exit(main())
