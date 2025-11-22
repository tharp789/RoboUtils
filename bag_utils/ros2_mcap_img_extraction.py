import os
import cv2
import numpy as np
from cv_bridge import CvBridge
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory as Ros2DecoderFactory
from rosidl_runtime_py.utilities import get_message
import rclpy
from rclpy.serialization import deserialize_message
from rclpy.node import Node
from sensor_msgs.msg import Image
import rosbag2_py


class McapImageExtractor(Node):
    def __init__(self, bag_folder, rgb_topic_name, depth_topic_name, 
                 depth_camera_info_topic_name, rgb_camera_info_topic_name):
        super().__init__('mcap_image_extractor')
        self.bag_folder = bag_folder
        self.rgb_topic_name = rgb_topic_name
        self.depth_topic_name = depth_topic_name
        self.depth_camera_info_topic_name = depth_camera_info_topic_name
        self.rgb_camera_info_topic_name = rgb_camera_info_topic_name
        self.rgb_camera_info = None
        self.depth_camera_info = None
        self.bridge = CvBridge()
        self.image_count = 0
        self.camera_info_processed = {}

        if not os.path.isdir(self.bag_folder):
            self.bag_files = [self.bag_folder]
            self.bag_folder = os.path.dirname(self.bag_folder)
        else:
            self.bag_files = [os.path.join(self.bag_folder, f) for f in os.listdir(self.bag_folder) if f.endswith('.mcap')]

        self.rgb_output_folder = os.path.join(self.bag_folder, "rgb_images")
        self.depth_output_folder = os.path.join(self.bag_folder, "depth_images")
        self.depth_viz_output_folder = os.path.join(self.bag_folder, "depth_viz_images")

        # Ensure output folder exists
        if not os.path.exists(self.rgb_output_folder):
            os.makedirs(self.rgb_output_folder, exist_ok=True)
        if not os.path.exists(self.depth_output_folder):
            os.makedirs(self.depth_output_folder, exist_ok=True)
        if not os.path.exists(self.depth_viz_output_folder):
            os.makedirs(self.depth_viz_output_folder, exist_ok=True)

    def extract_images(self):
        for filename in self.bag_files:
                with open(filename, 'rb') as f:
                    reader = make_reader(f, decoder_factories=[Ros2DecoderFactory()])
                    for msg_view in reader.iter_decoded_messages():
                        if msg_view.channel.topic == self.rgb_topic_name:
                            msg_type_str = msg_view.schema.name
                            msg_type = get_message(msg_type_str)
                            timestamp = msg_view.message.log_time
                            self.process_rgb_image(msg_view.message.data, timestamp, msg_type)

                        elif msg_view.channel.topic == self.depth_topic_name:
                            msg_type_str = msg_view.schema.name
                            msg_type = get_message(msg_type_str)
                            timestamp = msg_view.message.log_time
                            self.process_depth_image(msg_view.message.data, timestamp, msg_type)

                        elif msg_view.channel.topic in [self.depth_camera_info_topic_name, self.rgb_camera_info_topic_name]:
                            msg_type_str = msg_view.schema.name
                            msg_type = get_message(msg_type_str)
                            timestamp = msg_view.message.log_time
                            self.process_camera_info(
                                msg_view.message.data,
                                msg_type,
                                msg_view.channel.topic  # pass topic string
                            )
 
                self.get_logger().info(f"Saved {self.image_count} images to {self.rgb_output_folder} and {self.depth_output_folder}")

    def extract_images_alternate(self):
        for filename in self.bag_files:
            with open(filename, "rb") as f:
                reader = make_reader(f)   # IMPORTANT: no Ros2DecoderFactory()

                for msg_view in reader.iter_messages():   # raw messages, no UTF-8 decoding
                    try:
                        topic = msg_view[1].topic
                        raw_data = msg_view[2].data
                        timestamp = msg_view[2].log_time
                        msg_type_str = msg_view[0].name
                        msg_type = get_message(msg_type_str)

                        if topic == self.rgb_topic_name:
                            self.process_rgb_image(raw_data, timestamp, msg_type)

                        elif topic == self.depth_topic_name:
                            self.process_depth_image(raw_data, timestamp, msg_type)

                        elif topic in [
                            self.depth_camera_info_topic_name,
                            self.rgb_camera_info_topic_name,
                        ]:
                            self.process_camera_info(
                                raw_data,
                                msg_type,
                                topic
                            )
                    except Exception as e:
                        self.get_logger().error(f"Failed to process message: {e}")

            self.get_logger().info(
                f"Saved {self.image_count} images to {self.rgb_output_folder} "
                f"and {self.depth_output_folder}"
            )

    def process_rgb_image(self, raw_data, timestamp, msg_type):
        try:
            msg = deserialize_message(raw_data, msg_type)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            timestamp_ns = str(timestamp)
            filename = os.path.join(self.rgb_output_folder, f"{timestamp_ns}.png")
            cv2.imwrite(filename, cv_image)
            self.image_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def process_depth_image(self, raw_data, timestamp, msg_type):
        try:
            msg = deserialize_message(raw_data, msg_type)
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            depth_viz = self.create_depth_viz(depth)
            timestamp_ns = str(timestamp)
            viz_filename = os.path.join(self.depth_viz_output_folder, f"{timestamp_ns}_viz.png")
            depth_filename = os.path.join(self.depth_output_folder, f"{timestamp_ns}.npy")
            np.save(depth_filename, depth)
            cv2.imwrite(viz_filename, depth_viz)
            self.image_count += 1
        except Exception as e:
            self.get_logger().error(f"Failed to process depth image: {e}")

    def process_camera_info(self, raw_data, msg_type, topic):
        try:
            if self.camera_info_processed.get(topic):
                return  # Already processed

            msg = deserialize_message(raw_data, msg_type)
            camera_intrinsics = np.array(msg.k).reshape(3, 3)

            # Save intrinsics to file, e.g., rgb_camera_intrinsics.npy or depth_camera_intrinsics.npy
            suffix = "depth" if "depth" in topic else "rgb"
            intrinsics_filename = os.path.join(self.bag_folder, f"{suffix}_camera_intrinsics.npy")
            np.save(intrinsics_filename, camera_intrinsics)

            self.camera_info_processed[topic] = True
            self.get_logger().info(f"{suffix.capitalize()} Camera intrinsics: {camera_intrinsics}")

        except Exception as e:
            self.get_logger().error(f"Failed to process camera info: {e}")


    def create_depth_viz(self, depth):
        min_depth = 0.5
        max_depth = 20.0
        depth[np.isnan(depth)] = min_depth
        depth[np.isinf(depth)] = min_depth
        depth[np.isneginf(depth)] = min_depth

        depth = (depth - min_depth) / (max_depth - min_depth)
        depth = (depth * 255).astype(np.uint8)
        depth = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
        return depth


def main():
    rclpy.init()
    bag_folder = "/media/tyler/hummingbird/tracking/vtol1_tracking/tracking_filtered.mcap"
    # rgb_topic_name = '/wire_cam/zed_node/left/image_rect_color'
    rgb_topic_name = '/pincer/tracking_2d_viz'
    depth_topic_name = '/wire_cam/zed_node/depth/depth_registered'
    depth_camera_info_topic_name = '/wire_cam/zed_node/depth/camera_info'
    rgb_camera_info_topic_name = '/wire_cam/zed_node/left/camera_info'

    try:
        extractor = McapImageExtractor(bag_folder, rgb_topic_name, depth_topic_name, 
                                       depth_camera_info_topic_name, rgb_camera_info_topic_name)
        # extractor.extract_images()
        extractor.extract_images_alternate()
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
