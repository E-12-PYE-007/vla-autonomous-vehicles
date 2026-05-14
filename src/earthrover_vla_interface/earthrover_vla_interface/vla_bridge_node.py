import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import TwistStamped

import cv2
from cv_bridge import CvBridge

import zenoh
from zenoh import Encoding

import struct
import json

import numpy as np

# --- Testing only ---
from ament_index_python.packages import get_package_share_directory
from pathlib import Path
# --------------------


class VLABridgeNode(Node):
    def __init__(self, z_session):
        super().__init__('vla_bridge_node')

        self.get_logger().debug("VLA Bridge Node initialised.")

        self.bridge = CvBridge()
        
        self.z_session = z_session

        # --- Testing only ----
        share_dir = Path(get_package_share_directory("earthrover_vla_interface"))
        self.test_past_path = str(share_dir / "past.png")
        self.test_curr_path = str(share_dir / "cur.png")
        self.test_instruction = "move toward blue trash bin"

        self.test_done = False
        self.test_tries = 0
        self.test_max_tries = 500  # e.g. 30 seconds at 1Hz
        # self.test_timer = self.create_timer(1.0, self.publish_test_retry)
        # self.test_timer2 = self.create_timer(1.0, self.publish_test_retry2)
        # ---------------------

        self.img_publisher = self.z_session.declare_publisher(
            'camera/img_compressed',
        )

        self.img_curr_publisher = self.z_session.declare_publisher(
            'camera/img_compressed_curr',
        )

        self.inst_publisher = self.z_session.declare_publisher(
            'robot/instruction',
        )

        self.cmd_publisher = self.create_publisher(
            TwistStamped,
            '/cmd_vel',
            10
        )

        # TODO: Subscribe to hidden action tokens, publish to action head

        self.vla_cmd_subscriber = self.z_session.declare_subscriber(
            "vla/cmd_vel",
            self.vla_cmd_callback
        )

        self.sim_img_subscriber = self.create_subscription(
            Image,
            '/cam',
            self.sim_img_callback,
            10
        )

        self.robot_img_subscriber = self.create_subscription(
            Image,
            '/robot/cam',
            self.robot_img_callback,
            10
        )

        self.curr_img = None
        self.prev_img = None
    
    # --- Testing only ---
    @staticmethod
    def encode_jpg_bytes(self, path):
        img = cv2.imread(path)
        if img is None:
            self.get_logger().error(f"Failed to read image: {path}")
            return None
        ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            self.get_logger().error(f"Failed to encode image: {path}")
            return None
        return enc.tobytes()

    def publish_test_retry(self):
        if self.test_done:
            self.get_logger().warn("test_done is True, skipping!") 
            return

        if self.test_tries >= self.test_max_tries:
            self.get_logger().warn("Test publish timed out; stopping retries.")
            self.test_done = True
            self.test_timer.cancel()
            return

        past = self.encode_jpg_bytes(self.test_past_path)
        if past is None:
            return
        curr = self.encode_jpg_bytes(self.test_curr_path)
        if curr is None:
            return

        payload = {
            "past_img": past.decode("latin-1"),
            "curr_img": curr.decode("latin-1"),
        }
        self.img_publisher.put(json.dumps(payload).encode("utf-8"))
        self.inst_publisher.put(self.test_instruction.encode("utf-8"))

        self.test_tries += 1
        self.get_logger().info(f"Published test pair+instruction (try {self.test_tries})")

    def publish_test_retry2(self):
        if self.test_done:
            self.get_logger().warn("test_done is True, skipping!") 
            return
        
        if self.test_tries >= self.test_max_tries:
            self.get_logger().warn("Test publish timed out; stopping retries.")
            self.test_done = True
            self.test_timer2.cancel()
            return

        curr = self.encode_jpg_bytes(self.test_curr_path)
        if curr is None:
            return

        payload = {
            "curr_img": curr.decode("latin-1"),
        }
        self.img_curr_publisher.put(json.dumps(payload).encode("utf-8"))

        self.test_tries += 1
        self.get_logger().info(f"Published test pair+instruction (try {self.test_tries})")

    # -------------------

    def vla_cmd_callback(self, msg):
        time_cmd_sent, lin_x, ang_z = struct.unpack('ddd', msg.payload.to_bytes())

        time_cmd = rclpy.time.Time(seconds=time_cmd_sent)

        # Hardware: local NTP time
        # Sim: reads Gazebo time
        # TODO: Confirm latency

        cmd_msg = TwistStamped()
        cmd_msg.header.stamp = time_cmd.to_msg()
        cmd_msg.header.frame_id = "body_link"
        cmd_msg.twist.linear.x = lin_x
        cmd_msg.twist.angular.z = ang_z

        self.cmd_publisher.publish(cmd_msg)

        # Test
        # if not self.test_done:
        #     self.test_done = True
        #     self.test_timer.cancel()
        #     self.get_logger().info("Received vla/cmd_vel; stopping test retries.")
        # ---
        self.get_logger().info(f"Cmd sent from VLA. lin_x: {lin_x:.2f} | ang_z: {ang_z:.3f}")

    @staticmethod
    def process_img(img):
        height, width = img.shape[:2]
        min_dim = min(height, width)
        start_x = (width - min_dim) // 2
        start_y = (height - min_dim) // 2
        square_img = img[
            start_y : start_y + min_dim, start_x : start_x + min_dim
        ]

        return cv2.resize(square_img, (224, 224), interpolation=cv2.INTER_LINEAR)

    def sim_img_callback(self, msg):
        # /cam publishes images at 10Hz
        # every 0.1s, we keep updating current image and previous image to be fed into vla and action head
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        img_resized = self.process_img(img)

        # TODO: Include times - taking into account clock synchronisation
        
        # TODO: Play around with the quality and compression method
        success, encoded_img = cv2.imencode('.jpg', img_resized, [cv2.IMWRITE_JPEG_QUALITY, 80])

        img_bytes = encoded_img.tobytes()

        if self.prev_img is None:
            self.prev_img = img_bytes
            return

        self.curr_img = img_bytes

        if success:
            payload = {
                "past_img" : self.prev_img.decode("latin-1"),
                "curr_img" : self.curr_img.decode("latin-1"),
            }
            json_string = json.dumps(payload)
            self.img_publisher.put(json_string.encode("utf-8"))

        self.prev_img = self.curr_img

    def robot_img_callback(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # TODO: Must resize image so it is roughly square, test the camera to see what resize would be good
        # img_resized = cv2.resize(img, (224, 224))
        
        # TODO: Play around with the quality and compression method
        success, encoded_img = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if success:
            img_bytes = encoded_img.tobytes()

            self.img_publisher.put(img_bytes)



def main(args=None):
    rclpy.init(args=args)

    z_conf = zenoh.Config()
    z_conf.insert_json5("mode", '"client"')
    z_conf.insert_json5("connect/endpoints", '["tcp/127.0.0.1:7447"]')
    z_conf.insert_json5("scouting/multicast/enabled", "false")
    with zenoh.open(z_conf) as z_session:
        vla_bridge_node = VLABridgeNode(z_session)
        rclpy.spin(vla_bridge_node)

        vla_bridge_node.destroy_node()
        rclpy.shutdown()
    

if __name__ == '__main__':
    main()