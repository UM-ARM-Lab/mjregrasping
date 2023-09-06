import pickle

import cv2
import numpy as np
import datetime
from pathlib import Path
from typing import List

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper

import zivid
from tf.transformations import quaternion_from_matrix


def assisted_capture(camera: zivid.Camera) -> zivid.Frame:
    suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=800),
        ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
    )
    settings = zivid.capture_assistant.suggest_settings(camera, suggest_settings_parameters)
    return camera.capture(settings)


def main():
    rospy.init_node('zivid_calib')

    tfw = TF2Wrapper()

    app = zivid.Application()

    print("Connecting to camera")
    camera = app.connect_camera()

    current_pose_id = 0
    hand_eye_input = []
    board_in_cam_list = []
    robot_to_hand_transforms = []

    n_poses = 10
    while current_pose_id < n_poses:
        frame = assisted_capture(camera)

        pc = frame.point_cloud()

        try:
            transform = tfw.get_transform('hdt_michigan_root', 'drive7')
            transform[:3, 3] *= 1000
            robot_to_hand_transforms.append(transform)
            robot_pose = zivid.calibration.Pose(transform)

            print("Detecting checkerboard in point cloud")
            detection_result = zivid.calibration.detect_feature_points(pc)

            if detection_result.valid():
                board_in_cam = detection_result.pose().to_matrix()
                board_in_cam_list.append(board_in_cam)
                print("Calibration board detected")
                hand_eye_input.append(zivid.calibration.HandEyeInput(robot_pose, detection_result))
                current_pose_id += 1

                print(f"{current_pose_id}/{n_poses} poses collected, press enter to continue or ctrl+c to exit")
                input("Move robot to next pose and press enter to continue")

            else:
                print("Failed!!!")
                input("Move robot to a better pose and press enter to continue")
        except ValueError as ex:
            print(ex)

    raw_inputs = {
        'board_in_cam_list':        board_in_cam_list,
        'robot_to_hand_transforms': robot_to_hand_transforms
    }
    with open("latest_raw_inputs.pkl", "wb") as f:
        pickle.dump(raw_inputs, f)

    calibration_result = zivid.calibration.calibrate_eye_to_hand(hand_eye_input)

    if calibration_result.valid():
        print("Hand-Eye calibration OK")
        print(f"Result:\n{calibration_result}")
    else:
        print("Hand-Eye calibration FAILED")
        return

    print(calibration_result)
    transform = calibration_result.transform()
    print_result(transform)


def print_result(prev_result: zivid.Matrix4x4):
    prev_result = np.array(prev_result)
    q_wxyz = quaternion_from_matrix(prev_result)
    pos = prev_result[:3, 3] / 1000
    print("previous result:")
    print(
        f"rosrun tf2_ros static_transform_publisher {pos[0]} {pos[1]} {pos[2]} {q_wxyz[0]} {q_wxyz[1]} {q_wxyz[2]} {q_wxyz[3]} world zivid_optical_frame")


if __name__ == "__main__":
    main()
