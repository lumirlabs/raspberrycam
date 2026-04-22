#!/usr/bin/env python3
"""
Raspberry Pi camera + framebuffer LCD smoke test.

What this script does:
1) Captures one frame from the Raspberry Pi camera.
2) Saves the frame as a JPEG (default: ./capture_test.jpg).
3) Renders a simple status screen to the LCD framebuffer (default: /dev/fb1).

Designed for a 480x320 GPIO TFT display and a Raspberry Pi Camera Module 3.
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import subprocess
import sys
import tempfile
import time
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw


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


def capture_with_picamera2(width: int, height: int) -> Image.Image:
    from picamera2 import Picamera2  # type: ignore

    picam = Picamera2()
    config = picam.create_still_configuration(main={"size": (width, height)})
    picam.configure(config)
    picam.start()
    try:
        # Give auto-exposure/auto-white-balance a moment to settle.
        picam.set_controls({"AeEnable": True, "AwbEnable": True})
        time.sleep(0.8)
        frame = picam.capture_array("main")
    finally:
        picam.stop()
        picam.close()
    return Image.fromarray(frame).convert("RGB")


def capture_with_libcamera_still(width: int, height: int) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        cmd = [
            "libcamera-still",
            "-n",
            "-t",
            "800",
            "--width",
            str(width),
            "--height",
            str(height),
            "-o",
            tmp_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return Image.open(tmp_path).convert("RGB")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def capture_camera_image(width: int, height: int) -> Tuple[Optional[Image.Image], str]:
    try:
        return capture_with_picamera2(width, height), "picamera2"
    except Exception as e_picam:
        try:
            return capture_with_libcamera_still(width, height), "libcamera-still"
        except Exception as e_libcam:
            reason = f"picamera2 failed: {e_picam}; libcamera-still failed: {e_libcam}"
            return None, reason


def fit_cover(image: Image.Image, width: int, height: int) -> Image.Image:
    src_w, src_h = image.size
    src_ratio = src_w / src_h
    dst_ratio = width / height
    if src_ratio > dst_ratio:
        # Source is wider; crop width.
        new_h = src_h
        new_w = int(new_h * dst_ratio)
    else:
        # Source is taller; crop height.
        new_w = src_w
        new_h = int(new_w / dst_ratio)
    left = (src_w - new_w) // 2
    top = (src_h - new_h) // 2
    cropped = image.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((width, height), Image.Resampling.LANCZOS)


def build_status_image(
    camera_image: Optional[Image.Image], width: int, height: int, camera_status: str
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    if camera_image is not None:
        canvas = fit_cover(camera_image, width, height)
        draw = ImageDraw.Draw(canvas)
    else:
        draw.rectangle((0, 0, width, height), fill=(20, 20, 20))
        draw.rectangle((20, 20, width - 20, height - 20), outline=(255, 80, 80), width=3)
        draw.text((32, 36), "CAMERA CAPTURE FAILED", fill=(255, 100, 100))

    overlay_h = 64
    draw.rectangle((0, height - overlay_h, width, height), fill=(0, 0, 0, 180))
    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    draw.text((12, height - 54), "RaspberryCam smoke test", fill=(255, 255, 255))
    draw.text((12, height - 34), f"time: {now}", fill=(200, 200, 200))
    draw.text((12, height - 16), f"camera: {camera_status}", fill=(160, 255, 160))
    return canvas


def rgb888_to_rgb565_bytes(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    out = bytearray(image.width * image.height * 2)
    i = 0
    for r, g, b in image.getdata():
        rgb565 = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3)
        out[i] = (rgb565 >> 8) & 0xFF
        out[i + 1] = rgb565 & 0xFF
        i += 2
    return bytes(out)


def write_framebuffer(image: Image.Image, fb_path: str, width: int, height: int) -> None:
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    payload = rgb888_to_rgb565_bytes(image)
    with open(fb_path, "wb") as fb:
        fb.write(payload)


def main() -> int:
    parser = argparse.ArgumentParser(description="Camera + LCD framebuffer smoke test")
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
        "--output",
        default="capture_test.jpg",
        help="Output file for captured camera frame (default: capture_test.jpg)",
    )
    parser.add_argument(
        "--skip-display",
        action="store_true",
        help="Capture and save image, but do not write to framebuffer",
    )
    args = parser.parse_args()

    try:
        fallback_w, fallback_h = [int(v) for v in args.size.lower().split("x", 1)]
    except Exception:
        print("Invalid --size format. Use WIDTHxHEIGHT, for example 480x320.")
        return 2

    fb_path = pick_framebuffer(args.fb)
    fb_name = os.path.basename(fb_path)
    detected_size = parse_virtual_size(fb_name)
    if detected_size:
        disp_w, disp_h = detected_size
    else:
        disp_w, disp_h = fallback_w, fallback_h

    print(f"Display target: {disp_w}x{disp_h}")
    print(f"Framebuffer selected: {fb_path}")

    camera_img, camera_status = capture_camera_image(1536, 864)
    if camera_img is not None:
        camera_img.save(args.output, format="JPEG", quality=92)
        print(f"Camera capture saved: {args.output}")
    else:
        print("Camera capture failed.")
        print(camera_status)

    status_img = build_status_image(camera_img, disp_w, disp_h, camera_status)
    status_path = "lcd_test_render.jpg"
    status_img.save(status_path, format="JPEG", quality=90)
    print(f"LCD render preview saved: {status_path}")

    if args.skip_display:
        print("Skipping framebuffer write (--skip-display).")
        return 0

    if not os.path.exists(fb_path):
        print(f"Framebuffer device not found: {fb_path}")
        available = list_framebuffers()
        if available:
            print(f"Available framebuffer devices: {', '.join(available)}")
            print("Try setting one explicitly with --fb /dev/fb0 or --fb /dev/fb1.")
        else:
            print("No framebuffer devices were found under /dev/fb*.")
            print("Hint: verify LCD overlay/driver in /boot/firmware/config.txt")
        return 1

    try:
        write_framebuffer(status_img, fb_path, disp_w, disp_h)
        print(f"Wrote status image to framebuffer: {fb_path}")
        print("If the display remains blank, try running as root (sudo).")
        return 0
    except PermissionError:
        print(f"Permission denied writing {fb_path}. Try: sudo python3 scripts/hw_smoketest.py")
        return 1
    except Exception as e:
        print(f"Failed to write framebuffer: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
