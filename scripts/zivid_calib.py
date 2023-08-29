import cv2
import numpy as np
import datetime
from pathlib import Path
from typing import List

import rospy
from arc_utilities.tf2wrapper import TF2Wrapper

import zivid
from mjregrasping.save_load_matrix import assert_affine_matrix_and_save
from tf.transformations import quaternion_from_matrix


def enter_robot_pose(tfw, index: int) -> zivid.calibration.Pose:
    """Robot pose user input.

    iospy.init_node('zivid_calib')

    Args:
        index: Robot pose ID

    Returns:
        robot_pose: Robot pose

    """

    transform = tfw.get_transform('hdt_michigan_root', 'drive7')
    robot_pose = zivid.calibration.Pose(transform)
    print(f"The following pose was entered:\n{robot_pose}")
    return robot_pose


def perform_calibration(hand_eye_input: List[zivid.calibration.HandEyeInput]) -> zivid.calibration.HandEyeOutput:
    """Hand-Eye calibration type user input.

    Args:
        hand_eye_input: Hand-Eye calibration input

    Returns:
        hand_eye_output: Hand-Eye calibration result

    """
    print("Performing eye-to-hand calibration")
    hand_eye_output = zivid.calibration.calibrate_eye_to_hand(hand_eye_input)
    return hand_eye_output


def assisted_capture(camera: zivid.Camera) -> zivid.Frame:
    """Acquire frame with capture assistant.

    Args:
        camera: Zivid camera

    Returns:
        frame: Zivid frame

    """
    suggest_settings_parameters = zivid.capture_assistant.SuggestSettingsParameters(
        max_capture_time=datetime.timedelta(milliseconds=800),
        ambient_light_frequency=zivid.capture_assistant.SuggestSettingsParameters.AmbientLightFrequency.none,
    )
    settings = zivid.capture_assistant.suggest_settings(camera, suggest_settings_parameters)
    return camera.capture(settings)


def main():
    try:
        prev_result = zivid.Matrix4x4()
        prev_result.load(file_path="transform.yaml")
        print_result(prev_result)
    except:
        print("Failed to load previous result")

    rospy.init_node('zivid_calib')

    tfw = TF2Wrapper()

    app = zivid.Application()

    print("Connecting to camera")
    camera = app.connect_camera()

    current_pose_id = 0
    hand_eye_input = []

    while current_pose_id < 10:
        frame = assisted_capture(camera)

        pc = frame.point_cloud()

        rgba = pc.copy_data('rgba')
        rgb = rgba[:, :, :3]

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (0, 0), fx=0.5, fy=0.5)
        cv2.imshow('rgb', bgr)
        cv2.waitKey(1)

        try:
            robot_pose = enter_robot_pose(tfw, current_pose_id)

            print("Detecting checkerboard in point cloud")
            detection_result = zivid.calibration.detect_feature_points(pc)

            if detection_result.valid():
                detection_result.pose().to_matrix()
                print("Calibration board detected")
                hand_eye_input.append(zivid.calibration.HandEyeInput(robot_pose, detection_result))
                current_pose_id += 1
            else:
                print("Failed to detect calibration board, ensure that the entire board is in the view of the camera")
        except ValueError as ex:
            print(ex)

        rospy.sleep(5)

    calibration_result = perform_calibration(hand_eye_input)
    transform = calibration_result.transform()
    print_result(transform)
    transform_file_path = Path(Path(__file__).parent / "transform.yaml")
    assert_affine_matrix_and_save(transform, transform_file_path)

    if calibration_result.valid():
        print("Hand-Eye calibration OK")
        print(f"Result:\n{calibration_result}")
    else:
        print("Hand-Eye calibration FAILED")


def print_result(prev_result: zivid.Matrix4x4):
    prev_result = np.array(prev_result)
    q_wxyz = quaternion_from_matrix(prev_result)
    pos = prev_result[:3, 3] / 1000
    print("previous result:")
    print(
        f"rosrun tf2_ros static_transform_publisher {pos[0]} {pos[1]} {pos[2]} {q_wxyz[0]} {q_wxyz[1]} {q_wxyz[2]} {q_wxyz[3]} world zivid_optical_frame")


if __name__ == "__main__":
    main()
