from time import time

import numpy as np
import pyrealsense2 as rs
from PIL import Image

import rospy


def main():
    rospy.init_node("collect_hand_imgs")

    # left
    serial_numbers = [
        '128422270394',  # left
        '126122270471',  # right
    ]
    tool_idx = 1

    config = rs.config()
    config.enable_device(serial_numbers[tool_idx])

    pipe = rs.pipeline()
    pipe.start(config)

    for idx in range(10):
        rospy.sleep(2)

        frames = pipe.wait_for_frames()

        align = rs.align(rs.stream.color)
        aligned_frames = align.process(frames)
        rgb_frame = aligned_frames.first(rs.stream.color)
        rgb = np.asanyarray(rgb_frame.get_data())
        now = int(time())

        filename = f"imgs/hand_{now}.png"
        print(filename)

        Image.fromarray(rgb).save(filename)


if __name__ == '__main__':
    main()
