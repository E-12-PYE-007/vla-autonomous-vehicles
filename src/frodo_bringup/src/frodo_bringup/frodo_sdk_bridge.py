#!/usr/bin/env python3

"""
Frodo SDK Bridge Node

This ROS2 node connects ROS stack to the FrodoBot SDK.

It does three main things:

1. Subscribes to /cmd_vel
   - Receives velocity commands from your controller / VLA stack.
   - Sends those commands to the Frodo SDK using POST /control.

2. Polls /v2/screenshot
   - Gets front and rear camera images from the SDK.
   - Converts them into ROS Image messages.

3. Polls /data
   - Gets basic telemetry from the SDK.
   - Publishes battery, GPS, IMU, speed, orientation, and signal level.

"""

import base64
import requests
import numpy as np
import cv2

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, Imu, NavSatFix, BatteryState
from std_msgs.msg import Float32, Int32
from cv_bridge import CvBridge


class FrodoSDKBridge(Node):
    """
    ROS2 bridge between the Frodo SDK HTTP API and ROS topics.
    """

    def __init__(self):
        super().__init__("frodo_sdk_bridge")

        # -------------------------------------------------
        # Parameters
        # -------------------------------------------------
        # Local URL where the Frodo SDK server is running.
        self.declare_parameter("base_url", "http://localhost:8000")

        # How often to resend the latest /cmd_vel command to the SDK.
        self.declare_parameter("control_rate_hz", 10.0)

        # How often to poll telemetry from /data.
        self.declare_parameter("telemetry_rate_hz", 5.0)

        # How often to poll camera frames from /v2/screenshot.
        self.declare_parameter("camera_rate_hz", 2.0)

        # Approximate max robot speed.
        # Frodo SDK expects linear command in [-1, 1].
        # ROS /cmd_vel uses m/s.
        # 4 km/h ≈ 1.111 m/s.
        self.declare_parameter("max_linear_mps", 1.111)

        # Scaling factor for angular velocity.
        # ROS uses rad/s, SDK expects normalized [-1, 1].
        self.declare_parameter("max_angular_cmd", 1.0)

        # Lamp state sent with every command.
        # 0 = off, 1 = on.
        self.declare_parameter("lamp", 0)

        # Read parameter values.
        self.base_url = self.get_parameter("base_url").value
        self.max_linear_mps = float(self.get_parameter("max_linear_mps").value)
        self.max_angular_cmd = float(self.get_parameter("max_angular_cmd").value)
        self.lamp = int(self.get_parameter("lamp").value)

        # Converts OpenCV images to ROS Image messages.
        self.bridge = CvBridge()

        # Latest velocity command received from /cmd_vel.
        # These values are resent periodically by send_latest_command().
        self.latest_linear = 0.0
        self.latest_angular = 0.0

        # -------------------------------------------------
        # Subscribers
        # -------------------------------------------------
        # Receive velocity commands from controller / edge adapter / teleop.
        self.cmd_sub = self.create_subscription(
            Twist,
            "/cmd_vel",
            self.cmd_vel_callback,
            10,
        )

        # -------------------------------------------------
        # Publishers
        # -------------------------------------------------
        # Camera image publishers.
        self.front_image_pub = self.create_publisher(
            Image,
            "/frodo/front/image_raw",
            10,
        )

        self.rear_image_pub = self.create_publisher(
            Image,
            "/frodo/rear/image_raw",
            10,
        )

        # Telemetry publishers.
        self.gps_pub = self.create_publisher(
            NavSatFix,
            "/frodo/gps/fix",
            10,
        )

        self.imu_pub = self.create_publisher(
            Imu,
            "/frodo/imu",
            10,
        )

        self.battery_pub = self.create_publisher(
            BatteryState,
            "/frodo/battery",
            10,
        )

        self.orientation_pub = self.create_publisher(
            Float32,
            "/frodo/orientation_deg",
            10,
        )

        self.speed_pub = self.create_publisher(
            Float32,
            "/frodo/speed",
            10,
        )

        self.signal_pub = self.create_publisher(
            Int32,
            "/frodo/signal_level",
            10,
        )

        # -------------------------------------------------
        # Timers
        # -------------------------------------------------
        # Convert frequencies into timer periods.
        control_period = 1.0 / float(self.get_parameter("control_rate_hz").value)
        telemetry_period = 1.0 / float(self.get_parameter("telemetry_rate_hz").value)
        camera_period = 1.0 / float(self.get_parameter("camera_rate_hz").value)

        # Periodically send the latest command to the SDK.
        self.control_timer = self.create_timer(
            control_period,
            self.send_latest_command,
        )

        # Periodically read robot telemetry.
        self.telemetry_timer = self.create_timer(
            telemetry_period,
            self.poll_telemetry,
        )

        # Periodically read front/rear camera frames.
        self.camera_timer = self.create_timer(
            camera_period,
            self.poll_cameras,
        )

        self.get_logger().info(f"Frodo SDK bridge started using {self.base_url}")

    def cmd_vel_callback(self, msg: Twist):
        """
        Store the latest /cmd_vel command.

        This callback does not directly send the command to the robot.
        Instead, send_latest_command() repeatedly sends the newest command
        at control_rate_hz.
        """
        self.latest_linear = msg.linear.x
        self.latest_angular = msg.angular.z

    def clamp(self, value, low, high):
        """
        Clamp value to a selected range.
        Used to keep SDK commands inside [-1, 1].
        """
        return max(low, min(high, value))

    def send_latest_command(self):
        """
        Convert ROS /cmd_vel into Frodo SDK command format and send it.

        ROS:
            linear.x  = metres/second
            angular.z = radians/second

        SDK:
            linear  = normalized value in [-1, 1]
            angular = normalized value in [-1, 1]
        """

        # Scale linear velocity from m/s into SDK range [-1, 1].
        linear_cmd = self.clamp(
            self.latest_linear / self.max_linear_mps,
            -1.0,
            1.0,
        )

        # Scale angular velocity into SDK range [-1, 1].
        angular_cmd = self.clamp(
            self.latest_angular / self.max_angular_cmd,
            -1.0,
            1.0,
        )

        # Payload format expected by POST /control.
        payload = {
            "command": {
                "linear": linear_cmd,
                "angular": angular_cmd,
                "lamp": self.lamp,
            }
        }

        # Send command to SDK.
        try:
            requests.post(
                f"{self.base_url}/control",
                json=payload,
                timeout=0.2,
            )
        except requests.RequestException as e:
            self.get_logger().warn(f"Failed to send control command: {e}")

    def poll_telemetry(self):
        """
        Poll telemetry from the SDK /data endpoint.

        Publishes:
            /frodo/gps/fix
            /frodo/battery
            /frodo/orientation_deg
            /frodo/speed
            /frodo/signal_level
            /frodo/imu
        """

        # Request telemetry JSON from SDK.
        try:
            response = requests.get(f"{self.base_url}/data", timeout=0.5)
            data = response.json()
        except requests.RequestException as e:
            self.get_logger().warn(f"Failed to get telemetry: {e}")
            return
        except ValueError:
            self.get_logger().warn("Telemetry response was not valid JSON")
            return

        now = self.get_clock().now().to_msg()

        # -----------------------------
        # GPS
        # -----------------------------
        if "latitude" in data and "longitude" in data:
            gps = NavSatFix()
            gps.header.stamp = now
            gps.header.frame_id = "gps_link"
            gps.latitude = float(data.get("latitude", 0.0))
            gps.longitude = float(data.get("longitude", 0.0))
            gps.altitude = 0.0
            self.gps_pub.publish(gps)

        # -----------------------------
        # Battery
        # -----------------------------
        battery = BatteryState()
        battery.header.stamp = now

        # BatteryState.percentage expects [0.0, 1.0].
        # SDK battery is assumed to be [0, 100].
        battery.percentage = float(data.get("battery", 0.0)) / 100.0
        self.battery_pub.publish(battery)

        # -----------------------------
        # Orientation
        # -----------------------------
        orientation = Float32()
        orientation.data = float(data.get("orientation", 0.0))
        self.orientation_pub.publish(orientation)

        # -----------------------------
        # Speed
        # -----------------------------
        speed = Float32()
        speed.data = float(data.get("speed", 0.0))
        self.speed_pub.publish(speed)

        # -----------------------------
        # Signal level
        # -----------------------------
        signal = Int32()
        signal.data = int(data.get("signal_level", data.get("signal", 0)))
        self.signal_pub.publish(signal)

        # IMU is handled separately because the SDK gives arrays of samples.
        self.publish_latest_imu(data, now)

    def publish_latest_imu(self, data, stamp):
        """
        Publish the latest accelerometer and gyroscope readings as sensor_msgs/Imu.

        Expected SDK format:
            accels = [[ax, ay, az, timestamp], ...]
            gyros  = [[gx, gy, gz, timestamp], ...]

        This function uses only the newest sample from each list.
        """

        accels = data.get("accels", [])
        gyros = data.get("gyros", [])

        # If no IMU samples exist, do not publish.
        if not accels and not gyros:
            return

        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "imu_link"

        # Latest accelerometer sample.
        if accels:
            ax, ay, az, *_ = accels[-1]
            imu.linear_acceleration.x = float(ax)
            imu.linear_acceleration.y = float(ay)
            imu.linear_acceleration.z = float(az)

        # Latest gyroscope sample.
        if gyros:
            gx, gy, gz, *_ = gyros[-1]
            imu.angular_velocity.x = float(gx)
            imu.angular_velocity.y = float(gy)
            imu.angular_velocity.z = float(gz)

        self.imu_pub.publish(imu)

    def poll_cameras(self):
        """
        Poll front and rear camera images from the SDK.

        The SDK /v2/screenshot endpoint returns:
            front_frame: base64 encoded image
            rear_frame:  base64 encoded image
            timestamp:   SDK timestamp
        """

        try:
            response = requests.get(f"{self.base_url}/v2/screenshot", timeout=1.0)
            data = response.json()
        except requests.RequestException as e:
            self.get_logger().warn(f"Failed to get camera frames: {e}")
            return
        except ValueError:
            self.get_logger().warn("Camera response was not valid JSON")
            return

        timestamp = data.get("timestamp", None)

        # Publish front camera.
        self.publish_frame(
            data.get("front_frame"),
            self.front_image_pub,
            "front_camera",
            timestamp,
        )

        # Publish rear camera.
        self.publish_frame(
            data.get("rear_frame"),
            self.rear_image_pub,
            "rear_camera",
            timestamp,
        )

    def publish_frame(self, base64_image, publisher, frame_id, timestamp=None):
        """
        Decode one SDK base64 image and publish it as a ROS Image.

        Args:
            base64_image:
                Base64 encoded image string from the SDK.

            publisher:
                ROS Image publisher to publish on.

            frame_id:
                Frame name for the image header.

            timestamp:
                SDK timestamp. Currently unused; ROS receive time is used instead.
        """

        # Skip if SDK did not return an image.
        if not base64_image:
            return

        try:
            # Convert base64 string into raw compressed image bytes.
            image_bytes = base64.b64decode(base64_image)

            # Convert bytes into NumPy array.
            image_array = np.frombuffer(image_bytes, dtype=np.uint8)

            # Decode JPEG/PNG/WebP into OpenCV BGR image.
            cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            if cv_image is None:
                self.get_logger().warn(f"Could not decode image for {frame_id}")
                return

            # Convert OpenCV image into ROS Image message.
            msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = frame_id

            publisher.publish(msg)

        except Exception as e:
            self.get_logger().warn(f"Failed to publish {frame_id} image: {e}")

    def destroy_node(self):
        """
        Shutdown hook.

        Sends a final stop command so the robot does not keep moving after
        the ROS node exits.
        """

        self.get_logger().info("Stopping FrodoBot before shutdown...")

        try:
            requests.post(
                f"{self.base_url}/control",
                json={
                    "command": {
                        "linear": 0.0,
                        "angular": 0.0,
                        "lamp": self.lamp,
                    }
                },
                timeout=0.2,
            )
        except requests.RequestException:
            pass

        super().destroy_node()


def main(args=None):
    """
    Main ROS2 entry point.
    """
    rclpy.init(args=args)

    node = FrodoSDKBridge()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()