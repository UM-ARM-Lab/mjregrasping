#!/usr/bin/env python
import cv2

import numpy as np
import torch
import zivid
from arm_segmentation.predictor import Predictor
from cdcpd_torch.core.deformable_object_configuration import RopeConfiguration
from cdcpd_torch.core.tracking_map import TrackingMap
from cdcpd_torch.data_utils.types.grippers import GrippersInfo, GripperInfoSingle
from cdcpd_torch.modules.cdcpd_module_arguments import CDCPDModuleArguments
from cdcpd_torch.modules.cdcpd_network import CDCPDModule
from cdcpd_torch.modules.cdcpd_parameters import CDCPDParamValues
from cdcpd_torch.modules.post_processing.configuration import PostProcConfig, PostProcModuleChoice
from zivid.experimental import calibration

import ros_numpy
import rospy
from geometry_msgs.msg import Point
from mjregrasping.rviz import pc_np_to_pc_msg
from sensor_msgs.msg import Image, PointCloud2
from visualization_msgs.msg import MarkerArray, Marker

CAMERA_FRAME = 'zivid_optical_frame'
MASK_THRESHOLD = 0.3


def pairwise_squared_distances(a: torch.tensor, b: torch.tensor):
    a_s = np.sum(np.square(a), axis=-1, keepdims=True)  # [b, ..., n, 1]
    b_s = np.sum(np.square(b), axis=-1, keepdims=True)  # [b, ..., m, 1]
    dist = a_s - 2 * a @ np.moveaxis(b, -1, -2) + np.moveaxis(b_s, -1, -2)  # [b, ..., n, m]
    return dist


def cdcpd_helper(cdcpd_module: CDCPDModule, tracking_map: TrackingMap, depth: np.ndarray, mask: np.ndarray,
                 intrinsic: np.ndarray, grasped_points: GrippersInfo, seg_pc: torch.tensor, kvis=100.0):
    """

    Args:
        cdcpd_module:
        tracking_map:
        depth:
        mask:
        intrinsic:
        grasped_points:
        seg_pc: [3, N]
        kvis:

    Returns:

    """
    # verts = tracking_map.form_vertices_cloud()
    # verts_np = verts.detach().numpy()
    # Y_emit_prior = visibility_prior(verts_np, depth, mask, intrinsic[:3, :3], kvis)
    # Y_emit_prior = torch.from_numpy(Y_emit_prior.reshape(-1, 1)).to(torch.double)
    # assert torch.all(~torch.isnan(Y_emit_prior))

    # model_input = CDCPDModuleArguments(tracking_map, seg_pc, gripper_info=grasped_points)
    # model_input.set_Y_emit_prior(Y_emit_prior)
    model_input = CDCPDModuleArguments(tracking_map, seg_pc)

    cdcpd_module_arguments = cdcpd_module(model_input)

    post_processing_out = cdcpd_module_arguments.get_Y_opt()
    tracking_map.update_def_obj_vertices(post_processing_out)

    verts = tracking_map.form_vertices_cloud()
    cur_state_estimate = verts.detach().numpy().T

    return cur_state_estimate


def estimate_to_msg(current_estimate):
    current_estimate_msg = MarkerArray()
    for i, p in enumerate(current_estimate):
        marker = Marker()
        marker.header.frame_id = CAMERA_FRAME
        marker.id = i
        marker.ns = f'point_{i}'
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = p[0]
        marker.pose.position.y = p[1]
        marker.pose.position.z = p[2]
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
    marker.header.frame_id = CAMERA_FRAME
    marker.id = len(current_estimate)
    marker.scale.x = 0.01
    marker.scale.y = 0.01
    marker.scale.z = 0.01
    marker.color.a = 1.0
    marker.color.r = 0.0
    marker.color.g = 1.0
    marker.color.b = 0.0
    for i, p in enumerate(current_estimate):
        marker.points.append(Point(*p))
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
        self.pc_pub = rospy.Publisher('pc', PointCloud2, queue_size=10)
        self.segmented_pc_pub = rospy.Publisher('segmented_pc', PointCloud2, queue_size=10)
        self.rgb_pub = rospy.Publisher('rgb', Image, queue_size=10)
        self.depth_pub = rospy.Publisher('depth', Image, queue_size=10)
        self.mask_pub = rospy.Publisher('mask', Image, queue_size=10)

        device = 'cpu'
        rope_start_pos = torch.tensor([0.25, -0.5, 1.2]).to(device)
        rope_end_pos = torch.tensor([0.12, 0.5, 1.2]).to(device)

        # Segmentation
        self.seg_pred = Predictor()

        NUM_TRACKED_POINTS = 15
        MAX_ROPE_LENGTH = 1.24
        USE_DEBUG_MODE = False
        ALPHA = 0.5
        BETA = 1.0
        LAMBDA = 1.0
        ZETA = 50.0
        W = 0.1
        OBSTACLE_COST_WEIGHT = 0.02
        FIXED_POINTS_WEIGHT = 100.0
        OBJECTIVE_VALUE_THRESHOLD = 1.0
        MIN_DISTANCE_THRESHOLD = 0.04

        def_obj_config = RopeConfiguration(NUM_TRACKED_POINTS, MAX_ROPE_LENGTH, rope_start_pos, rope_end_pos)
        def_obj_config.initialize_tracking()

        self.previous_estimate = def_obj_config.initial_.vertices_.detach().numpy().T
        self.cdcpd_pred_pub.publish(estimate_to_msg(self.previous_estimate))

        self.tracking_map = TrackingMap()
        self.tracking_map.add_def_obj_configuration(def_obj_config)

        postproc_config = PostProcConfig(module_choice=PostProcModuleChoice.PRE_COMPILED)

        param_vals = CDCPDParamValues()
        param_vals.alpha_.val = ALPHA
        param_vals.beta_.val = BETA
        param_vals.lambda_.val = LAMBDA
        param_vals.zeta_.val = ZETA
        param_vals.w_.val = W
        param_vals.obstacle_cost_weight_.val = OBSTACLE_COST_WEIGHT
        param_vals.obstacle_constraint_min_dist_threshold.val = MIN_DISTANCE_THRESHOLD

        self.grasped_points = GrippersInfo()
        self.grasped_points.append(GripperInfoSingle(fixed_pt_pred=torch.tensor([0.25, -0.5, 1.2]), grasped_vertex_idx=0))
        self.grasped_points.append(GripperInfoSingle(fixed_pt_pred=torch.tensor([-0.25, 0.0, 1.0]), grasped_vertex_idx=14))
        # for idx, point in [(0, rope_start_pos), (14, rope_end_pos)]:

        print("Compiling CDCPD module...")
        cdcpd_module = CDCPDModule(deformable_object_config_init=def_obj_config, param_vals=param_vals,
                                   postprocessing_option=postproc_config, debug=USE_DEBUG_MODE, device=device)
        self.cdcpd_module = cdcpd_module.eval()
        print("done.")

        self.intrinsics = None

    def run(self, camera):
        filter_radius = 1.00

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
                xyz_mm = point_cloud.copy_data("xyz")
                rgba = point_cloud.copy_data("rgba")

            now = perf_counter()
            dt = now - last_t
            print(f'dt: {dt:.4f}, {filter_radius:=.3f}')
            last_t = now

            xyz = xyz_mm / 1000.0
            rgb = rgba[:, :, :3]
            depth = xyz[:, :, 2]

            self.viz_pc(depth, rgb, xyz)

            # run segmentation
            predictions = self.seg_pred.predict(rgb)
            mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
            for p in predictions:
                if p['class'] == 'red_cable':
                    binary_mask = (p['mask'] > MASK_THRESHOLD).astype(np.uint8)
                    mask = cv2.bitwise_or(mask, binary_mask)
            mask = mask.astype(bool)

            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            hsv_mask = (hsv[:, :, 0]) <= 15 | (hsv[:, :, 0] >= 220)
            mask = mask & hsv_mask

            seg_rgb = rgb.copy()
            seg_rgb[~mask] = 0
            mask_msg = ros_numpy.msgify(Image, seg_rgb, encoding='rgb8')
            self.mask_pub.publish(mask_msg)

            seg_xyz = xyz.copy()
            seg_xyz[~mask] = np.nan
            seg_pc = np.concatenate([seg_xyz, seg_rgb], axis=-1)
            seg_pc_flat = seg_pc.reshape(-1, 6).T
            seg_pc_flat = seg_pc_flat[:, ~np.isnan(seg_pc_flat).any(axis=0)]

            if seg_pc_flat.shape[1] < 100:
                print("Too few points in the segmented point cloud!")
                continue
            else:
                filter_radius = max(0.15, filter_radius * 0.9)

            seg_pc_xyz = seg_pc_flat[:3, :]
            # remove all points that are far from the current estimate
            dists = np.sqrt(pairwise_squared_distances(seg_pc_xyz.T, self.previous_estimate))
            min_dists = dists.min(axis=1)
            near_idxs = np.where(min_dists < filter_radius)[0]
            seg_pc_near = seg_pc_flat[:, near_idxs]

            self.segmented_pc_pub.publish(pc_np_to_pc_msg(seg_pc_near, names='x,y,z,r,g,b', frame_id=CAMERA_FRAME))

            # run cdcpd
            seg_pc_cdcpd = torch.from_numpy(seg_pc_flat[:3, :]).double()  # CDCPD only wants XYZ, and as a torch tensor
            current_estimate = cdcpd_helper(self.cdcpd_module, self.tracking_map, depth, mask, self.intrinsics,
                                            self.grasped_points, seg_pc_cdcpd)
            current_estimate_msg = estimate_to_msg(current_estimate)
            self.cdcpd_pred_pub.publish(current_estimate_msg)

            self.previous_estimate = current_estimate

    def viz_pc(self, depth, rgb, xyz):
        xyz_flat = xyz.reshape(-1, 3)
        is_valid = ~np.isnan(xyz_flat).any(axis=1)
        valid_idxs = np.where(is_valid)[0]
        xyz_flat_filtered = xyz_flat[valid_idxs]  # remove NaNs
        rgb_flat = rgb.reshape(-1, 3)
        rgb_flat = rgb_flat[valid_idxs]  # remove NaNs
        # publish inputs
        self.rgb_pub.publish(ros_numpy.msgify(Image, rgb, encoding='rgb8'))
        self.depth_pub.publish(ros_numpy.msgify(Image, depth, encoding='32FC1'))
        # create record array with x, y, and z fields
        pc = np.concatenate([xyz_flat_filtered, rgb_flat], axis=1).T
        pc_msg = pc_np_to_pc_msg(pc, names='x,y,z,r,g,b', frame_id=CAMERA_FRAME)  # FIXME: use camera frame
        self.pc_pub.publish(pc_msg)


def main():
    rospy.init_node("cdcpd_torch_node")
    n = CDCPDTorchNode()
    with zivid.Application() as app:
        with app.connect_camera() as camera:
            n.run(camera)


if __name__ == "__main__":
    main()
