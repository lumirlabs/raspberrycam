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
```