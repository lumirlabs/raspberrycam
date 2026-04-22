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
import os
import pathlib
import sys
import time
from typing import List, Optional, Tuple

from PIL import Image


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


def fit_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        new_h = src_h
        new_w = int(new_h * dst_ratio)
    else:
        new_w = src_w
        new_h = int(new_w / dst_ratio)
    left = (src_w - new_w) // 2
    top = (src_h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((width, height), Image.Resampling.BILINEAR)


def rgb888_to_rgb565_bytes(image: Image.Image, *, little_endian: bool, bgr: bool) -> bytes:
    image = image.convert("RGB")
    out = bytearray(image.width * image.height * 2)
    i = 0
    for r, g, b in image.getdata():
        if bgr:
            r, b = b, r
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        if little_endian:
            out[i] = rgb565 & 0xFF
            out[i + 1] = (rgb565 >> 8) & 0xFF
        else:
            out[i] = (rgb565 >> 8) & 0xFF
            out[i + 1] = rgb565 & 0xFF
        i += 2
    return bytes(out)


def rgb888_to_xrgb8888_bytes(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    out = bytearray(image.width * image.height * 4)
    i = 0
    for r, g, b in image.getdata():
        # Little-endian XRGB8888 byte order in memory: B, G, R, X
        out[i] = b
        out[i + 1] = g
        out[i + 2] = r
        out[i + 3] = 0x00
        i += 4
    return bytes(out)


def encode_framebuffer_payload(image: Image.Image, pixel_format: str) -> bytes:
    if pixel_format == "rgb565le":
        return rgb888_to_rgb565_bytes(image, little_endian=True, bgr=False)
    if pixel_format == "rgb565be":
        return rgb888_to_rgb565_bytes(image, little_endian=False, bgr=False)
    if pixel_format == "bgr565le":
        return rgb888_to_rgb565_bytes(image, little_endian=True, bgr=True)
    if pixel_format == "xrgb8888":
        return rgb888_to_xrgb8888_bytes(image)
    raise ValueError(f"Unsupported pixel format: {pixel_format}")


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


def prepare_preview_frame(
    frame_image: Image.Image, disp_w: int, disp_h: int, rotate: int
) -> Image.Image:
    img = fit_cover(frame_image, disp_w, disp_h)
    if rotate:
        rotated = img.rotate(rotate, expand=True)
        img = rotated.resize((disp_w, disp_h), Image.Resampling.BILINEAR)
    return img


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
        default="960x540",
        help="Camera capture resolution for preview (default: 960x540)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Target preview framerate (default: 10)",
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
    print("Starting live preview. Press Ctrl+C to stop.")

    try:
        from picamera2 import Picamera2  # type: ignore
    except Exception as e:
        print(f"picamera2 import failed: {e}")
        print("Install with: sudo apt install -y python3-picamera2")
        return 1

    picam = Picamera2()
    config = picam.create_preview_configuration(main={"size": (cam_w, cam_h), "format": "RGB888"})
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

    try:
        with open(fb_path, "wb", buffering=0) as fb:
            while True:
                frame_begin = time.monotonic()
                frame = picam.capture_array("main")
                image = Image.fromarray(frame).convert("RGB")
                preview = prepare_preview_frame(image, disp_w, disp_h, args.rotate)
                payload = encode_framebuffer_payload(preview, pixel_format)
                fb.seek(0)
                fb.write(payload)

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
        picam.stop()
        picam.close()


if __name__ == "__main__":
    sys.exit(main())
