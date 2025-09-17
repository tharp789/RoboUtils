import os
import numpy as np
import rosbag2_py
from rosidl_runtime_py.utilities import get_message
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message

class McapPoseExtractor(Node):
    def __init__(self, bag_path, pose_topic_name, csv_output_path):
        super().__init__('mcap_pose_extractor')
        self.bag_path = bag_path
        self.pose_topic_name = pose_topic_name
        self.csv_output_path = csv_output_path

        csv_output_dir = os.path.dirname(csv_output_path)
        if not os.path.exists(csv_output_dir):
            os.makedirs(csv_output_dir)

        # create internal csv file
        self.csv_file = open(self.csv_output_path, 'w')
        self.csv_file.write("timestamp,x,y,z,qx,qy,qz,qw\n")
        self.get_logger().info(f"Initialized McapPoseExtractor with bag: {bag_path}, topic: {pose_topic_name}, output: {csv_output_path}")

    def typename(self, topic_name):
        for topic_type in self.topic_types:
            if topic_type.name == topic_name:
                return topic_type.type
        raise ValueError(f"topic {topic_name} not in bag")

    def extract_poses(self):
        self.get_logger().info(f"Extracting poses from topic: {self.pose_topic_name}")
        reader = rosbag2_py.SequentialReader()
        reader.open(
            rosbag2_py.StorageOptions(uri=self.bag_path, storage_id="mcap"),
            rosbag2_py.ConverterOptions(
                input_serialization_format="cdr", output_serialization_format="cdr"
            ),
        )
        self.topic_types = reader.get_all_topics_and_types()

        while reader.has_next():
            topic, data, timestamp = reader.read_next()
            if topic == self.pose_topic_name:
                msg_type = get_message(self.typename(topic))
                msg = deserialize_message(data, msg_type)
                pose = msg.pose
                self.csv_file.write(
                    f"{timestamp},{pose.position.x},{pose.position.y},{pose.position.z},"
                    f"{pose.orientation.x},{pose.orientation.y},{pose.orientation.z},{pose.orientation.w}\n"
                )

        del reader
        self.csv_file.close()
        self.get_logger().info(f"Saved pose data to {self.csv_output_path}")


def main():
    rclpy.init()
    bag_path = "/home/tyler/Documents/MSR/field_tests/250828_afca/split_data/wire_data_20250828_092730/wire_data_20250828_092730_0.mcap"
    pose_topic_name = "/mavros/local_position/pose"
    csv_output_path = "/home/tyler/Documents/MSR/field_tests/250828_afca/split_data/wire_data_20250828_092730/pose_data.csv"
    extractor = McapPoseExtractor(bag_path, pose_topic_name, csv_output_path)
    extractor.extract_poses()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
