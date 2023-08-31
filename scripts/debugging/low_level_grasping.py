from pathlib import Path
import open3d as o3d
from PIL import Image
from time import time
from time import perf_counter

import cv2
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import pymanopt
import pyrealsense2 as rs
import pysdf_tools
import rerun as rr
import torch
from arm_segmentation.predictor import Predictor
from matplotlib import cm
from pymanopt.manifolds import SpecialOrthogonalGroup

import ros_numpy
import rospy
from arc_utilities import ros_init
from arc_utilities.tf2wrapper import TF2Wrapper
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.jacobian_ctrl import get_jacobian_ctrl
from mjregrasping.movie import MjRGBD
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import mj_transform_points, np_wxyz_to_xyzw
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import control_step
from mjregrasping.rviz import plot_points_rviz
from mjregrasping.scenarios import val_untangle
from mjregrasping.sdf_autograd import point_to_idx as point_to_idx_torch
from mjregrasping.sdf_autograd import sdf_lookup
from mjregrasping.viz import make_viz, Viz
from mjregrasping.voxelgrid import point_to_idx as point_to_idx_np
from ros_numpy.point_cloud2 import merge_rgb_fields
from sensor_msgs.msg import PointCloud2


def batch_rotate_and_translate(points, mat, pos=None):
    new_p = (mat @ points.T).T
    if pos is not None:
        new_p += pos
    return new_p


def val_dedup(x):
    """ removes the duplicated gripper values, which are at indices 10 and 19 """
    return np.concatenate([x[:10], x[11:19], x[20:]])


@ros_init.with_ros("low_level_grasping")
def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    scenario = val_untangle
    scenario.xml_path = Path("models/real_scene.xml")
    scenario.obstacle_name = "obstacles"

    rr.init('low_level_grasping')
    rr.connect()

    viz: Viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)

    d = mujoco.MjData(m)
    phy = Physics(m, d, objects)

    val = RealValCommander(phy.o.robot)

    pc_pub = rospy.Publisher('grasp_pc', PointCloud2, queue_size=1)

    tool_idx = 0
    tool_site_name = phy.o.rd.tool_sites[tool_idx]
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
    res = 0.002

    serial_numbers = [
        '128422270394',  # left
        '126122270471',  # right
    ]
    config = rs.config()
    config.enable_device(serial_numbers[tool_idx])
    pipe = rs.pipeline()
    pipe.start(config)

    v_scale = 0.05
    w_scale = 0.1
    jnt_lim_avoidance = 0.01
    max_v_norm = 0.03
    gripper_kp = 0.2

    gripper_q_indices = [phy.m.actuator(a).trnid[0] for a in phy.o.rd.gripper_actuator_names]
    tool_frame_name = phy.o.rd.tool_sites[tool_idx]

    camera_name = phy.o.rd.camera_names[tool_idx]
    camera_site_name = f'{camera_name}_cam'
    camera_frame = camera_site_name

    predictor = Predictor("model.pth")
    tfw = TF2Wrapper()

    last_t = perf_counter()
    for idx in range(50):
        t = perf_counter()
        dt = t - last_t
        print(f"{dt=:.3f}")

        # Set mujoco state to match the real robot
        update_mujoco_qpos(phy, val)

        points_rgb, points_xyz, rgb = read_realsense(FAR_THRESHOLD, pipe)

        if len(points_xyz) == 0:
            print("No points found!")
            continue

        publish_pointcloud(camera_frame, pc_pub, points_rgb, points_xyz)

        combined_mask = get_mask(predictor, rgb)

        fig = plt.figure()
        plt.imshow(rgb)
        plt.imshow(combined_mask, alpha=0.3)
        viz.fig(fig)
        plt.close(fig)

        points_xyz_masked = simple_hsv_mask(points_rgb, points_xyz)
        rope_points_in_cam = points_xyz_masked

        rope_points_in_cam = filter_by_volume(rope_points_in_cam)

        dcam_site = phy.d.site(camera_site_name)

        tool2world_mat = tool_site.xmat.reshape(3, 3)
        tool_site_pos = tool_site.xpos

        rope_points_in_tool = mj_transform_points(dcam_site, tool_site, rope_points_in_cam)

        gripper_q = phy.d.qpos[gripper_q_indices[tool_idx]]

        grasp_mat_in_tool, grasp_pos_in_tool = get_best_grasp(rope_points_in_tool)

        radius = 0.01
        grasp_z_in_tool = grasp_mat_in_tool[:, 2]
        skeleton = make_ring_skeleton(grasp_pos_in_tool, -grasp_z_in_tool, radius, delta_angle=0.5)

        v_in_tool = get_v_in_tool(skeleton, v_scale, max_v_norm)
        v_in_world = tool2world_mat @ v_in_tool

        ctrl, w_in_tool = get_jacobian_ctrl(phy, tool_site, grasp_mat_in_tool, v_in_tool, joint_indices,
                                            w_scale=w_scale, jnt_lim_avoidance=jnt_lim_avoidance)

        # control gripper speed based on whether the tool has converged to the grasp pose
        lin_speed = np.linalg.norm(v_in_tool)
        ang_speed = np.linalg.norm(w_in_tool)
        gripper_q_mix = np.clip(12 * lin_speed + 0.3 * ang_speed, 0, 1)
        desired_gripper_q = gripper_q_mix * hp['finger_q_open'] + (1 - gripper_q_mix) * hp['finger_q_closed']
        gripper_gripper_vel = gripper_kp * (desired_gripper_q - gripper_q)
        ctrl[-1] = gripper_gripper_vel

        # rescale to respect velocity limits
        ctrl = rescale_ctrl(phy, ctrl, act_indices)

        full_ctrl = np.zeros(phy.m.nu)
        full_ctrl[act_indices] = ctrl

        # send commands to the robot
        val.send_vel_command(phy.m, full_ctrl)

        plot_points_rviz(viz.markers_pub, rope_points_in_cam, idx=0, frame_id=camera_frame, label='rope_in_cam', s=0.2)
        plot_points_rviz(viz.markers_pub, rope_points_in_tool, idx=0, frame_id=tool_site_name,
                         label='rope points in tool', s=0.1)
        viz.lines(skeleton, "ring", 0, 0.007, 'g', frame_id=tool_frame_name)
        viz.arrow("v", tool_site_pos, v_in_world, 'w')
        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, grasp_mat_in_tool.flatten())
        tfw.send_transform(grasp_pos_in_tool, np_wxyz_to_xyzw(q_wxyz), tool_frame_name, 'grasp_pose_in_tool')

        if is_grasp_complete(gripper_q, desired_gripper_q):
            val.stop()
            print("Grasp successful!")
            break

        last_t = t

    pipe.stop()
    val.stop()


def filter_by_volume(rope_points_in_cam):
    # convert rope_points_in_cam to open3d pointcloud so we can do clustering and stuff
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(rope_points_in_cam)
    # cluster the points
    min_points = 50
    labels = np.array(pcd.cluster_dbscan(eps=0.02, min_points=min_points, print_progress=False))
    max_label = labels.max()
    # colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    # colors[labels < 0] = 0
    # pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([pcd])
    # Score each cluster by the volume and aspect ratio of its bounding box
    best_cluster = None
    best_cluster_pcd = None
    best_volume = 0
    for label in range(max_label + 1):
        cluster_mask = labels == label
        cluster_points = rope_points_in_cam[cluster_mask]
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
    predictions = predictor.predict(rgb)
    combined_mask = np.zeros(rgb.shape[:2], dtype=bool)
    for p in predictions:
        if p['class'] == 'red_cable':
            binary_mask = p['mask'] > 0.5
            combined_mask = combined_mask | binary_mask
    return combined_mask


def publish_pointcloud(camera_frame, pc_pub, points_rgb, points_xyz):
    r, g, b = points_rgb.T
    points_xyzrgb_rec = np.rec.fromarrays([*points_xyz.T, r, g, b], names='x,y,z,r,g,b')
    points_xyzrgb_rec = merge_rgb_fields(points_xyzrgb_rec)
    pc_msg = ros_numpy.msgify(PointCloud2, points_xyzrgb_rec, stamp=rospy.Time.now(), frame_id=camera_frame)
    pc_pub.publish(pc_msg)


def read_realsense(FAR_THRESHOLD, pipe):
    frames = pipe.wait_for_frames()
    align = rs.align(rs.stream.color)
    aligned_frames = align.process(frames)
    rgb_frame = aligned_frames.first(rs.stream.color)
    rgb = np.asanyarray(rgb_frame.get_data())
    depth_frame = aligned_frames.get_depth_frame()
    # save for training
    # now = int(time())
    # Image.fromarray(rgb).save(f"imgs/rgb_{now}.png")
    pc = rs.pointcloud()
    pc.map_to(rgb_frame)
    points_frame = pc.calculate(depth_frame)
    points_xyz = np.asanyarray(points_frame.get_vertices()).view(np.float32).reshape(-1, 3)
    points_rgb_uvs_float = np.asanyarray(points_frame.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    points_rgb_uvs_float = (points_rgb_uvs_float * rgb.shape[:2][::-1] + 0.5).astype(np.uint32)
    # remove points that are all zero or too far away
    point_dists = np.linalg.norm(points_xyz, axis=1)
    is_valid = ~np.all(points_xyz == 0, axis=1) & (point_dists < FAR_THRESHOLD)
    points_xyz = points_xyz[is_valid]
    points_rgb_uvs_float = points_rgb_uvs_float[is_valid]
    points_rgb = rgb[points_rgb_uvs_float[:, 1], points_rgb_uvs_float[:, 0]]
    return points_rgb, points_xyz, rgb


def update_mujoco_qpos(phy, val):
    js = val.get_latest_joint_state()
    for name, pos in zip(js.name, js.position):
        joint = phy.m.joint(name)
        mj_qpos_idx = joint.qposadr[0]
        phy.d.qpos[mj_qpos_idx] = pos
    phy.d.act = val_dedup(js.position)
    mujoco.mj_forward(phy.m, phy.d)


def simple_hsv_mask(points_rgb, points_xyz):
    points_hsv = cv2.cvtColor(points_rgb[None], cv2.COLOR_RGB2HSV)[0]
    mask_indices = np.where((points_hsv[:, 0] < 10) & (points_hsv[:, 1] > 50))[0]
    points_xyz_masked = points_xyz[mask_indices]
    return points_xyz_masked


def get_best_grasp(rope_points_in_tool):
    rope_points_in_tool_torch = torch.from_numpy(rope_points_in_tool)

    # Take the closest point to the tool tip, which is where the grippers will close
    distances = np.linalg.norm(rope_points_in_tool, axis=-1)
    closest_idx = np.argmin(distances)
    grasp_pos_in_tool = rope_points_in_tool[closest_idx]

    # use PCA to extract the long (x) direction of the rope
    _, _, V = torch.pca_lowrank(rope_points_in_tool_torch)
    rope_x_in_tool = V[:, 0]
    grasp_mat_in_tool = np.eye(3)

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

    grasp_pos_in_tool += grasp_mat_in_tool[:, 2] * 0.01  # helps grasp deeper in the gripper

    return grasp_mat_in_tool, grasp_pos_in_tool


def mj_get_rgbd_and_rope_mask(d, hand_r, phy):
    rgb = hand_r.render(d).copy()
    depth = hand_r.render(d, depth=True).copy()
    seg = hand_r.render_with_flags(d, {mujoco.mjtRndFlag.mjRND_IDCOLOR: 1, mujoco.mjtRndFlag.mjRND_SEGMENT: 1})
    seg = seg[:, :, 0].copy()  # only the R component is used
    # This +1 converts geom id to seg ids, because seg id 0 means 'background' which means NO geom
    rope_mask = np.zeros_like(seg)
    for seg_id in phy.o.rope.geom_indices + 1:
        rope_body_mask = (seg == seg_id)
        rope_mask = rope_mask | rope_body_mask
    return depth, rgb, rope_mask


def get_v_in_tool(skeleton, v_scale, max_v_norm):
    # b points in the right direction, but b gets bigger when you get closer, but we want it to get smaller
    # since we're doing everything in tool frame, the tool is always at the origin, so query the field at (0,0,0)
    b = skeleton_field_dir(skeleton, np.zeros([1, 3]))[0]
    b_normalized = b / np.linalg.norm(b)
    # v here means linear velocity, not to be confused with pixel column
    v_in_tool = b_normalized / np.linalg.norm(b) * v_scale
    v_norm = np.linalg.norm(v_in_tool)
    if v_norm > max_v_norm:
        v_in_tool = v_in_tool / v_norm * max_v_norm
    return v_in_tool


def is_grasp_complete(gripper_q, desired_gripper_q):
    return gripper_q < hp['finger_q_closed'] * 1.2 and desired_gripper_q < hp['finger_q_closed'] * 1.2


def get_masked_points(rgbd: MjRGBD, depth: np.ndarray, mask: np.ndarray):
    us, vs = np.where(mask)
    depth = depth[us, vs]
    xs = depth * (vs - rgbd.cx) / rgbd.fpx
    ys = depth * (us - rgbd.cy) / rgbd.fpx
    zs = depth
    xyz_in_cam = np.stack([xs, ys, zs], axis=1)
    return xyz_in_cam


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


if __name__ == '__main__':
    main()
