#!/usr/bin/env python

from pathlib import Path
import zivid

import numpy as np
import rospkg
import torch
from cdcpd_torch.core.deformable_object_configuration import RopeConfiguration
from cdcpd_torch.core.tracking_map import TrackingMap
from cdcpd_torch.core.visibility_prior import visibility_prior
from cdcpd_torch.data_utils.types.grippers import GrippersInfo, GripperInfoSingle
from cdcpd_torch.modules.cdcpd_module_arguments import CDCPDModuleArguments
from cdcpd_torch.modules.cdcpd_network import CDCPDModule
from cdcpd_torch.modules.cdcpd_parameters import CDCPDParamValues
from cdcpd_torch.modules.post_processing.configuration import PostProcConfig, PostProcModuleChoice
from zivid.experimental import calibration

import message_filters
import ros_numpy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from zivid_camera.srv import LoadSettingsFromFile, Capture


def cdcpd_helper(cdcpd_module: CDCPDModule, tracking_map: TrackingMap, depth: np.ndarray, mask: np.ndarray,
                 intrinsic: np.ndarray, grasped_points: GrippersInfo, segmented_pointcloud: np.ndarray, kvis=100.0):
    """

    Args:
        cdcpd_module:
        tracking_map:
        depth:
        mask:
        intrinsic:
        grasped_points:
        segmented_pointcloud: [3, N]
        kvis:

    Returns:

    """
    verts = tracking_map.form_vertices_cloud()
    verts_np = verts.detach().numpy()
    Y_emit_prior = visibility_prior(verts_np, depth, mask, intrinsic[:3, :3], kvis)
    Y_emit_prior = torch.from_numpy(Y_emit_prior.reshape(-1, 1)).to(torch.double)

    model_input = CDCPDModuleArguments(tracking_map, segmented_pointcloud, gripper_info=grasped_points)
    model_input.set_Y_emit_prior(Y_emit_prior)

    cdcpd_module_arguments = cdcpd_module(model_input)

    post_processing_out = cdcpd_module_arguments.get_Y_opt()
    tracking_map.update_def_obj_vertices(post_processing_out)

    verts = tracking_map.form_vertices_cloud()
    cur_state_estimate = verts.detach().numpy().T

    return cur_state_estimate


def estimate_to_msg(current_estimate):
    current_estimate_msg = MarkerArray()
    for i in range(len(current_estimate)):
        marker = Marker()
        marker.header.frame_id = "world"
        marker.id = i
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = current_estimate[i][0]
        marker.pose.position.y = current_estimate[i][1]
        marker.pose.position.z = current_estimate[i][2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        current_estimate_msg.markers.append(marker)
    marker = Marker()
    marker.type = Marker.LINE_STRIP
    marker.action = Marker.ADD
    marker.header.frame_id = "world"
    marker.id = len(current_estimate)
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    for i in range(len(current_estimate)):
        marker.points.append(current_estimate[i])
    current_estimate_msg.markers.append(marker)
    return current_estimate_msg


class CDCPDTorchNode:
    """
    triggers the Capture service as fast as possible, and every time the data comes back (via topics),
    it runs the CDCPD algorithm and publishes the result.
    """

    def __init__(self):
        # CDCPD
        self.cdcpd_pred_pub = rospy.Publisher('cdcpd_pred', MarkerArray, queue_size=10)

        device = 'cpu'
        rope_start_pos = torch.tensor([-0.5, 0.0, 1.0]).to(device)
        rope_end_pos = torch.tensor([0.5, 0.0, 1.0]).to(device)

        NUM_TRACKED_POINTS = 25
        MAX_ROPE_LENGTH = .768

        USE_DEBUG_MODE = False

        ALPHA = 0.5
        BETA = 1.0
        LAMBDA = 1.0
        ZETA = 50.0
        OBSTACLE_COST_WEIGHT = 0.02
        FIXED_POINTS_WEIGHT = 100.0
        OBJECTIVE_VALUE_THRESHOLD = 1.0
        MIN_DISTANCE_THRESHOLD = 0.04

        def_obj_config = RopeConfiguration(NUM_TRACKED_POINTS, MAX_ROPE_LENGTH, rope_start_pos, rope_end_pos)
        def_obj_config.initialize_tracking()

        self.previous_estimate = def_obj_config.initial_.vertices_.detach().numpy()
        self.cdcpd_pred_pub.publish(estimate_to_msg(self.previous_estimate))

        self.tracking_map = TrackingMap()
        self.tracking_map.add_def_obj_configuration(def_obj_config)

        postproc_config = PostProcConfig(module_choice=PostProcModuleChoice.PRE_COMPILED)

        param_vals = CDCPDParamValues()
        param_vals.alpha_.val = ALPHA
        param_vals.beta_.val = BETA
        param_vals.lambda_.val = LAMBDA
        param_vals.zeta_.val = ZETA
        param_vals.obstacle_cost_weight_.val = OBSTACLE_COST_WEIGHT
        param_vals.obstacle_constraint_min_dist_threshold.val = MIN_DISTANCE_THRESHOLD

        self.grasped_points = GrippersInfo()
        for idx, point in [(0, rope_start_pos), (24, rope_end_pos)]:
            self.grasped_points.append(GripperInfoSingle(fixed_pt_pred=point, grasped_vertex_idx=idx))

        print("Compiling CDCPD module...")
        cdcpd_module = CDCPDModule(deformable_object_config_init=def_obj_config, param_vals=param_vals,
                                   postprocessing_option=postproc_config, debug=USE_DEBUG_MODE, device=device)
        self.cdcpd_module = cdcpd_module.eval()
        print("done.")

        self.intrinsics = None

    def run(self, camera):
        camera_matrix = calibration.intrinsics(camera).camera_matrix
        self.intrinsics = np.array([
            [camera_matrix.fx, 0, camera_matrix.cx],
            [0, camera_matrix.fy, camera_matrix.cy],
            [0, 0, 1]
        ])
        settings = zivid.Settings.load('camera_settings.yml')

        from time import perf_counter
        last_t = perf_counter()
        while True:
            with camera.capture(settings) as frame:
                point_cloud = frame.point_cloud()
                xyz = point_cloud.copy_data("xyz")
                xyz_flat = xyz.reshape(-1, 3)
                xyz_flat = xyz_flat[~np.isnan(xyz_flat).any(axis=1)]
                rgba = point_cloud.copy_data("rgba")
                rgb = rgba[:, :, :3]
            now = perf_counter()
            dt = now - last_t
            print(f'dt: {dt:.4f}')
            last_t = now

    def on_camera_info(self, camera_info_msg: CameraInfo):
        self.intrinsic = np.array(camera_info_msg.K).reshape(3, 3)

    def on_data(self, rgb_msg: Image, depth_msg: Image):
        if self.intrinsic is None:
            return

        rospy.loginfo("data received")
        depth_np = ros_numpy.numpify(depth_msg)
        rgb_np = ros_numpy.numpify(rgb_msg)

        rgb_mask = (1 <= rgb_np[:, :, 0]) & (rgb_np[:, :, 0] <= 250)

        # apply the mask to the depth image
        us, vs = np.where(rgb_mask)
        depths = depth_np[us, vs]
        # convert to XYZ point cloud using the camera intrinsics
        us = us.astype(np.float64)
        vs = vs.astype(np.float64)
        xs = (us - self.intrinsic[0, 2]) * depths / self.intrinsic[0, 0]
        ys = (vs - self.intrinsic[1, 2]) * depths / self.intrinsic[1, 1]
        zs = depths
        segmented_pointcloud = np.stack([xs, ys, zs], axis=1)
        # remove NaNs
        segmented_pointcloud = segmented_pointcloud[~np.isnan(segmented_pointcloud).any(axis=1)]
        segmented_pointcloud = torch.from_numpy(segmented_pointcloud).T

        current_estimate = cdcpd_helper(self.cdcpd_module, self.tracking_map, depth_np, rgb_mask, self.intrinsic,
                                        self.grasped_points,
                                        segmented_pointcloud)
        current_estimate_msg = estimate_to_msg(current_estimate)
        self.cdcpd_pred_pub.publish(current_estimate_msg)


def main():
    rospy.init_node("cdcpd_torch_node")
    n = CDCPDTorchNode()
    with zivid.Application() as app:
        with app.connect_camera() as camera:
            n.run(camera)


if __name__ == "__main__":
    main()
