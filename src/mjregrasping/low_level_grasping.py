from time import perf_counter, time
from typing import Optional

import cv2
import numpy as np
import open3d as o3d
import pymanopt
import pyrealsense2 as rs
import rerun as rr
import torch
from arm_segmentation.predictor import Predictor
from pymanopt.manifolds import SpecialOrthogonalGroup

import ros_numpy
import rospy
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.jacobian_ctrl import get_jacobian_ctrl
from mjregrasping.my_transforms import mj_transform_points, xyzw_quat_from_matrix
from mjregrasping.params import hp
from mjregrasping.physics import Physics, get_q
from mjregrasping.real_val import RealValCommander
from mjregrasping.rerun_visualizer import log_frame
from mjregrasping.rollout import control_step
from mjregrasping.rviz import plot_points_rviz
from mjregrasping.val_dup import val_dedup
from mjregrasping.viz import Viz
from ros_numpy.point_cloud2 import merge_rgb_fields
from sensor_msgs.msg import PointCloud2, Image


class GraspFailed(Exception):
    pass


def run_grasp_controller(val_cmd: RealValCommander, phy: Physics, tool_idx: int, viz: Viz, finger_q_closed: float,
                         finger_q_open: float, grasp_pos_inset: float):
    if val_cmd is None:
        print(f"Cannot run grasp controller without a RealValCommander! skipping")
        return True

    rgb_pub = rospy.Publisher("grasp_rgb", Image, queue_size=10)
    mask_pub = rospy.Publisher("grasp_mask", Image, queue_size=10)
    pc_pub = rospy.Publisher("grasp_pc", PointCloud2, queue_size=10)
    tfw = TF2Wrapper()

    tool_site_name = phy.o.rd.tool_sites[tool_idx]
    mj_tool_site_frame = 'mj_' + tool_site_name
    tool_site = phy.d.site(tool_site_name)
    joint_indices_for_tool = np.array([
        [2, 3, 4, 5, 6, 7, 8, 9],
        [11, 12, 13, 14, 15, 16, 17, 18],
    ])
    act_indices_for_tool = np.array([
        [2, 3, 4, 5, 6, 7, 8, 9],
        [10, 11, 12, 13, 14, 15, 16, 17],
    ])
    joint_indices = joint_indices_for_tool[tool_idx]
    act_indices = act_indices_for_tool[tool_idx]
    FAR_THRESHOLD = 0.3

    serial_numbers = [
        '128422270394',  # left
        '126122270471',  # right
    ]
    config = rs.config()
    config.enable_device(serial_numbers[tool_idx])
    pipe = rs.pipeline()
    pipe.start(config)

    v_scale_bfield = 0.06
    v_scale_direct = 0.3
    w_scale = 0.06
    jnt_lim_avoidance = 0.04
    max_v_norm = 0.03
    gripper_kp = 1.25

    gripper_q_indices = [phy.m.actuator(a).trnid[0] for a in phy.o.rd.gripper_actuator_names]
    tool_frame_name = phy.o.rd.tool_sites[tool_idx]

    camera_name = phy.o.rd.camera_names[tool_idx]
    camera_site_name = f'{camera_name}_cam'
    camera_frame = camera_site_name
    log_frame('grasping', scale=0.1)

    predictor = Predictor("/home/peter/Documents/arm_segmentation/model.pth")
    radius = 0.02

    last_t = perf_counter()
    success = False
    prev_grasp_pos = None
    prev_grasp_mat = None
    for idx in range(125):
        t = perf_counter()
        dt = t - last_t
        print(f"{dt=:.3f}")

        dcam_site = phy.d.site(camera_site_name)

        tool2world_mat = tool_site.xmat.reshape(3, 3)
        tool_site_pos = tool_site.xpos

        gripper_q = phy.d.qpos[gripper_q_indices[tool_idx]]

        rope_points_in_cam = read_and_segment(FAR_THRESHOLD, pipe, predictor, camera_frame, pc_pub, rgb_pub, mask_pub)
        rope_found = len(rope_points_in_cam) > 0

        if rope_found:
            # These are in the tool frame of mujoco, which is not the same as the
            # tool from of the URDF, so we publish a new frame
            rope_points_in_tool = mj_transform_points(dcam_site, tool_site, rope_points_in_cam)
            cam2mj_tool_mat, cam2mj_tool_pos = rel_pos_mat(dcam_site, tool_site)
            cam2mj_tool_44 = pos_mat_to_44(cam2mj_tool_mat, cam2mj_tool_pos)
            cam2mj_tool_quat = xyzw_quat_from_matrix(cam2mj_tool_44)
            rr.log_points("grasping/rope_points_in_tool", rope_points_in_tool)
            plot_points_rviz(viz.markers_pub, rope_points_in_tool, idx=0, frame_id=mj_tool_site_frame,
                             label='rope points in tool', s=0.15, color='r')
            tfw.send_transform(cam2mj_tool_pos, cam2mj_tool_quat, camera_frame, mj_tool_site_frame)
            grasp_mat_in_tool, grasp_pos_in_tool = get_grasp_for_closest_point(rope_points_in_tool, grasp_pos_inset)
        else:
            # check whether CDCPD and mujoco mostly agree
            print("Using CDCPD")
            cdcpd_in_cam = val_cmd.get_cdcpd_np()
            zivid2tool = tfw.get_transform(tool_frame_name, "zivid_optical_frame")
            zivid2world = tfw.get_transform('world', "zivid_optical_frame")
            cdcpd_in_tool = cdcpd_in_cam @ zivid2tool[:3, :3].T + zivid2tool[:3, 3]
            cdcpd_in_world = cdcpd_in_cam @ zivid2world[:3, :3].T + zivid2world[:3, 3]
            cdcpd_mj_dist = np.linalg.norm(cdcpd_in_world - get_rope_points(phy), axis=-1)[21:].mean()
            if cdcpd_mj_dist > 0.07:
                print("CDCPD and mujoco are too far apart, skipping")
                continue
            # settle_after=False so we actually servo to where CDCPD says the rope is, not where mujoco says
            val_cmd.pull_rope_towards_cdcpd(phy, 500, settle_after=False)
            plot_points_rviz(viz.markers_pub, cdcpd_in_tool, idx=0, frame_id=tool_frame_name, label='cdcpd_in_tool',
                             color='b')
            grasp_mat_in_tool, grasp_pos_in_tool = get_grasp_cdcpd(cdcpd_in_tool, grasp_pos_inset)

        if prev_grasp_pos is None:
            prev_grasp_pos = grasp_pos_in_tool
            prev_grasp_mat = grasp_mat_in_tool
        elif np.linalg.norm(grasp_pos_in_tool - prev_grasp_pos) > 0.05:
            # reject outliers, use previous info
            print("Rejecting outlier")
            grasp_pos_in_tool = prev_grasp_pos
            grasp_mat_in_tool = prev_grasp_mat
        else:
            # smooth the grasp position to avoid jitter
            alpha = 0.8
            grasp_pos_in_tool = alpha * grasp_pos_in_tool + (1 - alpha) * prev_grasp_pos
            grasp_mat_in_tool = alpha * grasp_mat_in_tool + (1 - alpha) * prev_grasp_mat
            prev_grasp_mat = grasp_mat_in_tool
            prev_grasp_pos = grasp_pos_in_tool

        grasp_z_in_tool = grasp_mat_in_tool[:, 2]
        skeleton = make_ring_skeleton(grasp_pos_in_tool, -grasp_z_in_tool, radius, delta_angle=0.5)
        viz.lines(skeleton, "ring", 0, 0.007, 'g', frame_id=tool_frame_name)
        viz.arrow('grasp_z', grasp_pos_in_tool, grasp_z_in_tool * 0.1, 'm', frame_id=tool_frame_name)
        rr.log_line_strip("grasping/ring", skeleton)
        rr.log_point('grasping/pos', grasp_pos_in_tool, radius=0.006)
        rr.log_arrow('grasping/grasp_z', grasp_pos_in_tool, grasp_z_in_tool * 0.1, color=(1, 0, 1, 1.))

        v_in_tool_bfield = get_v_in_tool_bfield(skeleton, v_scale_bfield, max_v_norm)
        v_in_tool_direct = grasp_pos_in_tool * v_scale_direct
        v_in_tool = (v_in_tool_bfield + v_in_tool_direct) / 2

        v_in_world = tool2world_mat @ v_in_tool
        viz.arrow("v", tool_site_pos, v_in_world * 2, 'w')
        rr.log_arrow('grasping/lin_vel', np.zeros(3), v_in_tool * 2, color=(1, 1, 0, 1.))

        ctrl, w_in_tool = get_jacobian_ctrl(phy, tool_site, grasp_mat_in_tool, v_in_tool, joint_indices,
                                            w_scale=w_scale, jnt_lim_avoidance=jnt_lim_avoidance)

        # control gripper speed based on whether the tool has converged to the grasp pose
        r_xy = np.linalg.norm(grasp_pos_in_tool[:2])
        r_z = grasp_pos_in_tool[2]
        if r_xy < 0.035 and r_z < 0.007:
            desired_gripper_q = finger_q_closed + r_xy * 2
        else:
            desired_gripper_q = finger_q_open
        gripper_gripper_vel = gripper_kp * (desired_gripper_q - gripper_q)
        ctrl[-1] = gripper_gripper_vel
        rr.log_scalar('grasping/r_xy', r_xy, color=(1, 0, 1, 1.))
        rr.log_scalar('grasping/r_z', r_z, color=(1, 1, 1, 1.))
        rr.log_scalar('grasping/desired_gripper_q', desired_gripper_q, color=(0, 1, 0, 1.))
        rr.log_scalar('grasping/gripper_q', gripper_q, color=(0, 0, 1, 1.))

        # rescale to respect velocity limits
        ctrl = rescale_ctrl(phy, ctrl, act_indices)

        full_ctrl = np.zeros(phy.m.nu)
        full_ctrl[act_indices] = ctrl

        # testing in sim
        control_step(phy, full_ctrl, 0.6, val_cmd=val_cmd, slow=False)
        viz.viz(phy)
        match_mj_to_real(phy, val_cmd)

        if is_grasp_complete(gripper_q, desired_gripper_q, finger_q_closed) and rope_found:
            val_cmd.send_pos_command(val_cmd.get_latest_qpos_in_mj_order())
            print("Grasp successful!")
            success = True
            break

        last_t = t

    pipe.stop()

    # The real robot has moved whereas the mujoco robot has not,
    # so we need update mujoco to match the real robot state.
    val_cmd.update_mujoco_qpos(phy)

    return success


def pos_mat_to_44(mat, pos):
    """
    Convert a 3x3 rotation matrix and a 3x1 position vector to a 4x4 matrix

    Args:
        mat: 3x3 rotation matrix
        pos: 3x1 position vector

    Returns:
        4x4 matrix
    """
    m = np.eye(4)
    m[:3, :3] = mat
    m[:3, 3] = pos
    return m


def rel_pos_mat(a, b):
    b_mat_T = a.xmat.reshape(3, 3).T
    b_mat = b.xmat.reshape(3, 3)
    a2b_mat = b_mat_T @ b_mat
    a2b_pos = b_mat_T @ (b.xpos - a.xpos)
    return a2b_mat, a2b_pos


def is_grasp_complete(gripper_q, desired_gripper_q, finger_q_closed):
    return gripper_q < finger_q_closed + 0.05 and desired_gripper_q < finger_q_closed + 0.05


def read_and_segment(far_threshold, pipe, predictor: Predictor, camera_frame: str,
                     pc_pub: Optional[rospy.Publisher] = None,
                     rgb_pub: Optional[rospy.Publisher] = None,
                     mask_pub: Optional[rospy.Publisher] = None):
    frames = pipe.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    rgb_frame = aligned_frames.first(rs.stream.color)
    rgb = np.asanyarray(rgb_frame.get_data())
    depth_frame = aligned_frames.get_depth_frame()

    # Save for training
    from time import time
    from PIL import Image as PImage
    now = int(time())
    PImage.fromarray(rgb).save(f"imgs/rgb_{now}.png")

    if rgb_pub:
        rgb_msg = ros_numpy.msgify(Image, rgb, encoding='rgb8')
        rgb_pub.publish(rgb_msg)

    pc = rs.pointcloud()
    pc.map_to(rgb_frame)
    points_frame = pc.calculate(depth_frame)
    points_xyz = np.asanyarray(points_frame.get_vertices()).view(np.float32).reshape(-1, 3)
    points_rgb_uvs_float = np.asanyarray(points_frame.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    points_rgb_uvs_float = (points_rgb_uvs_float * rgb.shape[:2][::-1] + 0.5).astype(np.uint32)
    # remove points that are all zero or too far away
    point_dists = np.linalg.norm(points_xyz, axis=1)
    is_valid = ~np.all(points_xyz == 0, axis=1) & (point_dists < far_threshold)
    points_xyz = points_xyz[is_valid]
    points_rgb_uvs_float = points_rgb_uvs_float[is_valid]
    points_rgb = rgb[points_rgb_uvs_float[:, 1], points_rgb_uvs_float[:, 0]]

    seg_mask = get_mask(predictor, rgb)

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hsv_seg = (hsv[..., 0] < 25) & (hsv[..., 1] > 50)

    seg_mask = seg_mask & hsv_seg

    points_seg = seg_mask[points_rgb_uvs_float[:, 1], points_rgb_uvs_float[:, 0]]
    points_xyz_masked = points_xyz[np.where(points_seg)]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz_masked)
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.004)
    inlier_pcd, _ = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.2)
    points_xyz_filtered = np.asarray(inlier_pcd.points)

    if pc_pub:
        publish_pointcloud(camera_frame, pc_pub, points_rgb, points_xyz)

    if mask_pub:
        rgb[~seg_mask] = 255
        mask_msg = ros_numpy.msgify(Image, rgb, encoding='rgb8')
        mask_pub.publish(mask_msg)

    return points_xyz_filtered


def match_mj_to_real(phy: Physics, val_cmd):
    q_target = val_dedup(val_cmd.get_latest_qpos_in_mj_order())
    for i in range(4):
        q_current = get_q(phy)
        command = hp['joint_kp'] * (q_target - q_current)
        control_step(phy, command, sub_time_s=0.04)


def get_grasp_cdcpd(cdcpd_in_tool, grasp_pos_inset: float):
    cdcpd_pred_torch = torch.from_numpy(cdcpd_in_tool)

    # Take the closest point to the tool tip, which is where the grippers will close
    distances = np.linalg.norm(cdcpd_in_tool, axis=-1)
    distances_sorted_idx = np.argsort(distances)
    closest_idx = distances_sorted_idx[0]
    grasp_pos_in_tool = cdcpd_in_tool[closest_idx]
    next_to_closest = cdcpd_in_tool[closest_idx - 3]
    rope_x_in_tool = torch.from_numpy(grasp_pos_in_tool - next_to_closest)

    grasp_mat_in_tool = np.eye(3)

    grasp_mat_in_tool, grasp_pos_in_tool = opt_grasp(grasp_mat_in_tool, grasp_pos_in_tool, rope_x_in_tool,
                                                     grasp_pos_inset)

    return grasp_mat_in_tool, grasp_pos_in_tool


def get_grasp_for_closest_point(rope_points_in_tool, grasp_pos_inset: float):
    """

    Args:
        rope_points_in_tool: (n, 3)
        grasp_pos_inset: how far into the gripper from the tool tip to target the grasp
    """
    rope_points_in_tool_torch = torch.from_numpy(rope_points_in_tool)

    # Take the closest point to the tool tip, which is where the grippers will close
    distances = np.linalg.norm(rope_points_in_tool, axis=-1)
    closest_idx = np.argmin(distances)
    grasp_pos_in_tool = rope_points_in_tool[closest_idx]
    # grasp_pos_in_tool = np.mean(rope_points_in_tool, axis=-1)

    # use PCA to extract the long (x) direction of the rope
    _, _, V = torch.pca_lowrank(rope_points_in_tool_torch)
    rope_x_in_tool = V[:, 0]
    grasp_mat_in_tool = np.eye(3)

    grasp_mat_in_tool, grasp_pos_in_tool = opt_grasp(grasp_mat_in_tool, grasp_pos_in_tool, rope_x_in_tool,
                                                     grasp_pos_inset)

    return grasp_mat_in_tool, grasp_pos_in_tool


def opt_grasp(grasp_mat_in_tool, grasp_pos_in_tool, rope_x_in_tool, grasp_pos_inset):
    SO3 = SpecialOrthogonalGroup(3)
    manifold = SO3

    @pymanopt.function.pytorch(manifold)
    def cost(mat):
        candidate_grasp_x = mat[:, 0]
        x_align = -torch.abs(torch.dot(candidate_grasp_x, rope_x_in_tool))

        total_cost = sum([
            x_align,
        ])
        return total_cost

    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.SteepestDescent(max_iterations=15, min_step_size=5e-4, verbosity=0)
    result = optimizer.run(problem, initial_point=grasp_mat_in_tool)
    grasp_mat_in_tool = result.point
    grasp_pos_in_tool += grasp_mat_in_tool[:, 2] * grasp_pos_inset  # helps grasp deeper in the gripper
    return grasp_mat_in_tool, grasp_pos_in_tool


def get_v_in_tool_bfield(skeleton, bv_scale, max_v_norm):
    # b points in the right direction, but b gets bigger when you get closer, but we want it to get smaller
    # since we're doing everything in tool frame, the tool is always at the origin, so query the field at (0,0,0)
    b = skeleton_field_dir(skeleton, np.zeros([1, 3]))[0]
    b_normalized = b / np.linalg.norm(b)
    # v here means linear velocity, not to be confused with pixel column
    v_in_tool = b_normalized / np.linalg.norm(b) * bv_scale
    v_norm = np.linalg.norm(v_in_tool)
    if v_norm > max_v_norm:
        v_in_tool = v_in_tool / v_norm * max_v_norm
    return v_in_tool


def filter_by_volume(rope_points):
    # convert rope points to open3d pointcloud so we can do clustering and stuff
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rope_points)
    # cluster the points
    min_points = 50
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=min_points, print_progress=False))
    if len(labels) == 0:
        return rope_points

    max_label = labels.max()
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])
    # Score each cluster by the volume and aspect ratio of its bounding box
    best_cluster = rope_points
    best_cluster_pcd = None
    best_volume = 0
    for label in range(max_label + 1):
        cluster_mask = labels == label
        cluster_points = rope_points[cluster_mask]
        cluster_pcd = o3d.geometry.PointCloud()
        cluster_pcd.points = o3d.utility.Vector3dVector(cluster_points)
        try:
            bbox = cluster_pcd.get_oriented_bounding_box()
            bbox_volume = bbox.volume()
            if bbox_volume > best_volume:
                best_volume = bbox_volume
                best_cluster = cluster_points
                best_cluster_pcd = cluster_pcd
        except RuntimeError:
            pass  # Qhull precision error
    # o3d.visualization.draw_geometries([best_cluster_pcd])
    return best_cluster


def get_mask(predictor, rgb):
    predictions = predictor.predict(rgb, min_score_threshold=0.1)
    combined_mask = np.zeros(rgb.shape[:2], dtype=bool)
    for p in predictions:
        if p['class'] == 'red_cable':
            binary_mask = p['mask'] > 0.6
            combined_mask = combined_mask | binary_mask
    return combined_mask


def publish_pointcloud(camera_frame, pc_pub, points_rgb, points_xyz):
    r, g, b = points_rgb.T
    points_xyzrgb_rec = np.rec.fromarrays([*points_xyz.T, r, g, b], names='x,y,z,r,g,b')
    points_xyzrgb_rec = merge_rgb_fields(points_xyzrgb_rec)
    pc_msg = ros_numpy.msgify(PointCloud2, points_xyzrgb_rec, stamp=rospy.Time.now(), frame_id=camera_frame)
    pc_pub.publish(pc_msg)


def rescale_ctrl(phy, ctrl, act_indices=None):
    vmin = phy.m.actuator_ctrlrange[act_indices, 0]
    vmax = phy.m.actuator_ctrlrange[act_indices, 1]
    if np.any(ctrl > vmax):
        offending_joint = np.argmax(ctrl)
        ctrl = ctrl / np.max(ctrl) * vmax[offending_joint]
    if np.any(ctrl < vmin):
        offending_joint = np.argmin(ctrl)
        ctrl = ctrl / np.min(ctrl) * vmin[offending_joint]
    return ctrl
