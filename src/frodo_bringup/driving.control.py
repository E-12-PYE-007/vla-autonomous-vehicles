"""
Driving Control API Client
===========================
Programmatic control for the device at http://192.168.11.1
Based on the driving.html / driving.js web interface.

API Endpoints discovered:
  POST /api/v1/control  - Joystick movement (x, y from -1.0 to 1.0)
  POST /api/v1/lamp     - Lamp on/off (state: 1 or 0)
  POST /api/v1/autotest - Auto-test mode on/off (autotest: 1 or 0)
  POST /api/v1/fan      - Fan on/off (fanSwitch: 1 or 0)

Usage:
  from driving_control import DrivingController
  bot = DrivingController()
  bot.forward(0.5)    # drive forward at half speed
  bot.lamp_on()
"""

import requests
import time

BASE_URL = "http://192.168.11.1"


class DrivingController:
    def __init__(self, base_url=BASE_URL, timeout=5):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Low-level API calls ──────────────────────────────────────────

    def send_control(self, x: float, y: float):
        """
        Send a joystick control command.
        x: -1.0 (left) to 1.0 (right)
        y: -1.0 (up/forward) to 1.0 (down/backward)
        Note: In the web UI the y-axis is inverted (negative = forward).
        """
        x = max(-1.0, min(1.0, round(x, 2)))
        y = max(-1.0, min(1.0, round(y, 2)))
        return self._post("/api/v1/control", {"x": x, "y": y})

    def set_lamp(self, on: bool):
        """Turn the lamp on (True) or off (False)."""
        return self._post("/api/v1/lamp", {"state": 1 if on else 0})

    def set_autotest(self, on: bool):
        """Enable (True) or disable (False) auto-test mode."""
        return self._post("/api/v1/autotest", {"autotest": 1 if on else 0})

    def set_fan(self, on: bool):
        """Turn the fan on (True) or off (False)."""
        return self._post("/api/v1/fan", {"fanSwitch": 1 if on else 0})

    # ── Convenience helpers ──────────────────────────────────────────

    def stop(self):
        """Send a zero-movement command (stop)."""
        return self.send_control(0, 0)

    def forward(self, speed=1.0):
        """Drive forward. speed: 0.0 to 1.0"""
        return self.send_control(0, -abs(speed))

    def backward(self, speed=1.0):
        """Drive backward. speed: 0.0 to 1.0"""
        return self.send_control(0, abs(speed))

    def turn_left(self, speed=1.0):
        """Turn left. speed: 0.0 to 1.0"""
        return self.send_control(-abs(speed), 0)

    def turn_right(self, speed=1.0):
        """Turn right. speed: 0.0 to 1.0"""
        return self.send_control(abs(speed), 0)

    def forward_left(self, speed=1.0):
        return self.send_control(-abs(speed), -abs(speed))

    def forward_right(self, speed=1.0):
        return self.send_control(abs(speed), -abs(speed))

    def lamp_on(self):
        return self.set_lamp(True)

    def lamp_off(self):
        return self.set_lamp(False)

    def fan_on(self):
        return self.set_fan(True)

    def fan_off(self):
        return self.set_fan(False)

    # ── Sustained movement (mirrors the 100ms interval from the JS) ──

    def drive(self, x: float, y: float, duration: float, interval: float = 0.1):
        """
        Send repeated control commands for `duration` seconds,
        every `interval` seconds (default 100ms, same as the web UI).
        Sends a stop command when done.
        """
        end_time = time.time() + duration
        try:
            while time.time() < end_time:
                self.send_control(x, y)
                time.sleep(interval)
        finally:
            self.stop()

    def drive_forward(self, duration: float, speed: float = 1.0):
        """Drive forward for `duration` seconds."""
        self.drive(0, -abs(speed), duration)

    def drive_backward(self, duration: float, speed: float = 1.0):
        """Drive backward for `duration` seconds."""
        self.drive(0, abs(speed), duration)

    # ── Internal ─────────────────────────────────────────────────────

    def _post(self, path, payload):
        url = self.base_url + path
        try:
            resp = self.session.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            print(f"[ERROR] {url}: {e}")
            raise


# ── Quick demo ───────────────────────────────────────────────────────

if __name__ == "__main__":
    bot = DrivingController()

    print("Lamp ON")
    bot.lamp_on()

    print("Driving forward for 2 seconds at half speed...")
    bot.drive_forward(duration=2, speed=0.5)

    print("Turning right for 1 second...")
    bot.drive(x=0.8, y=-0.3, duration=1)

    print("Stopping & lamp OFF")
    bot.stop()
    bot.lamp_off()

    print("Done!")
