# raspberrycam

Simple Raspberry Pi digicam project (Pi Zero 2 W + camera + GPIO LCD).

## First hardware smoke test

Use this script to test camera capture and LCD output in one go:

`scripts/hw_smoketest.py`

### 1) Install dependencies on the Pi

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-pil libcamera-apps
```

### 2) Run the test

```bash
python3 scripts/hw_smoketest.py
```

If framebuffer permissions block the LCD write:

```bash
sudo python3 scripts/hw_smoketest.py
```

### 3) Useful options

```bash
# only test camera capture
python3 scripts/hw_smoketest.py --skip-display

# explicit framebuffer and display size
python3 scripts/hw_smoketest.py --fb /dev/fb1 --size 480x320

# if colors look wrong on LCD, try explicit pixel formats
python3 scripts/hw_smoketest.py --pixel-format rgb565le
python3 scripts/hw_smoketest.py --pixel-format bgr565le
python3 scripts/hw_smoketest.py --pixel-format rgb565be
```

Default framebuffer selection is `auto` (tries `/dev/fb1`, then `/dev/fb0`).

## LCD HAT driver setup (Joy-IT 3.5")

Based on the vendor manual (`lcd_manual.pdf`), this panel is installed using the
`LCD-show` driver package (not by manually guessing overlays).

```bash
sudo rm -rf LCD-show
git clone https://github.com/goodtft/LCD-show.git
chmod -R 755 LCD-show
cd LCD-show
sudo ./LCD35-show
```

The installer reboots the Pi.

After reboot:

```bash
ls -l /dev/fb*
python3 ~/raspberrycam/scripts/hw_smoketest.py
```

Optional rotation (from vendor manual):

```bash
cd LCD-show
sudo ./rotate.sh 90
```

## Live preview app

Run the first digicam preview prototype:

```bash
python3 raspycam.py
```

Useful options:

```bash
# choose specific framebuffer
python3 raspycam.py --fb /dev/fb0

# adjust speed/resolution on Pi Zero 2 W
python3 raspycam.py --camera-size 640x480 --fps 8
python3 raspycam.py --camera-size 480x320 --fps 15

# if orientation is wrong
python3 raspycam.py --rotate 90

# if colors are wrong
python3 raspycam.py --pixel-format rgb565le

# reduce tearing/scanlines (auto uses vsync/pageflip when supported)
python3 raspycam.py --sync-mode auto
python3 raspycam.py --sync-mode vsync
python3 raspycam.py --sync-mode pageflip
```
