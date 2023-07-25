from pathlib import Path

import mujoco
import numpy as np
import pymanopt
import pysdf_tools
import rerun as rr
import torch
from matplotlib import cm
from pymanopt.manifolds import SpecialOrthogonalGroup, Euclidean, Product
from scipy.linalg import logm, block_diag

from arc_utilities import ros_init
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.mjsaver import load_data_and_eq
from mjregrasping.move_to_joint_config import pid_to_joint_config, get_q
from mjregrasping.movie import MjRenderer, MjRGBD
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import mj_transform_points, np_wxyz_to_xyzw, transform_points
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.scenarios import val_untangle
from mjregrasping.sdf_autograd import sdf_lookup, point_to_idx
from mjregrasping.viz import make_viz, Viz
from sdf_tools.utils_3d import get_gradient


def batch_rotate_and_translate(points, mat, pos=None):
    new_p = (mat @ points.T).T
    if pos is not None:
        new_p += pos
    return new_p


@ros_init.with_ros("low_level_grasping")
def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    scenario = val_untangle

    rr.init('low_level_grasping')
    rr.connect()

    viz: Viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)

    d = mujoco.MjData(m)
    phy = Physics(m, d, objects)
    robot_q = np.array([
        0.1, 0.0,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.3,  # left gripper
        0.3, 0.0, 0, 0.0, np.pi / 2, -1.1, 0.3,  # right arm
        0.3,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q, sub_time_s=DEFAULT_SUB_TIME_S)
    robot_q = np.array([
        0.1, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.3,  # left gripper
        0.7, 0.0, 0, 0.0, 1.3, -1.1, 0.3,  # right arm
        0.3,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q, sub_time_s=DEFAULT_SUB_TIME_S)

    # d = load_data_and_eq(m, Path("states/grasp/1690310029.pkl"))
    # phy = Physics(m, d, objects)

    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    sdf_np = np.array(sdf.GetRawData(), dtype=np.float64)
    sdf_np = sdf_np.reshape([sdf.GetNumXCells(), sdf.GetNumYCells(), sdf.GetNumZCells()])
    sdf_origin = torch.tensor(sdf.GetOriginTransform().translation(), dtype=torch.float64)
    sdf_res = torch.tensor(sdf.GetResolution(), dtype=torch.float64)
    sdf_grad_np = get_gradient(sdf, dtype=np.float64)
    sdf_torch = torch.from_numpy(sdf_np)
    sdf_grad_torch = torch.from_numpy(sdf_grad_np)

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    tool_idx = 1
    tool_site_name = phy.o.rd.tool_sites[tool_idx]
    tool_site = phy.d.site(tool_site_name)

    w_scale = 0.2
    v_scale = 0.05
    gripper_kp = 1.0
    jnt_lim_avoidance = 0.1

    sdf_weight = 0.001
    d_rot_weight = 0.01
    rope_pos_weight = 0.1

    gripper_ctrl_indices = [phy.m.actuator(a).id for a in phy.o.rd.gripper_actuator_names]
    gripper_q_indices = [phy.m.actuator(a).trnid[0] for a in phy.o.rd.gripper_actuator_names]

    camera_name = phy.o.rd.camera_names[tool_idx]
    camera_site_name = f'{camera_name}_cam'
    hand_mcam = phy.m.camera(camera_name)
    hand_vcam = mujoco.MjvCamera()
    hand_vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    hand_vcam.fixedcamid = hand_mcam.id

    hand_r = MjRenderer(phy.m, cam=hand_vcam)
    hand_rgbd = MjRGBD(hand_mcam, hand_r)

    while True:
        rgb = hand_r.render(d).copy()
        depth = hand_r.render(d, depth=True).copy()

        seg = hand_r.render_with_flags(d, {mujoco.mjtRndFlag.mjRND_IDCOLOR: 1, mujoco.mjtRndFlag.mjRND_SEGMENT: 1})
        seg = seg[:, :, 0].copy()  # only the R component is used

        # not sure why this +1 works
        rope_mask = np.zeros_like(seg)
        for seg_id in phy.o.rope.geom_indices + 1:
            rope_body_mask = (seg == seg_id)
            rope_mask = rope_mask | rope_body_mask

        rr.log_image("img/rgb", rgb)
        rr.log_image("img/depth", np.clip(depth, 0, 1))
        rr.log_image("img/rope_mask", rope_mask * 255)

        # v in pixel space is x in camera space
        # u in pixel space is y in camera space
        rope_points_in_cam = get_masked_points(hand_rgbd, depth, rope_mask)
        bg_points_in_cam = get_masked_points(hand_rgbd, depth, 1 - rope_mask)

        dcam_site = phy.d.site(camera_site_name)
        cam2world_mat = dcam_site.xmat.reshape([3, 3])
        cam_pos_in_world = dcam_site.xpos[None]
        rope_points_in_world = batch_rotate_and_translate(rope_points_in_cam, cam2world_mat, cam_pos_in_world)
        bg_points_in_world = batch_rotate_and_translate(bg_points_in_cam, cam2world_mat, cam_pos_in_world)
        # viz.points('rope', rope_points_in_world, color='orange', radius=0.001)
        # viz.points('background', bg_points_in_world, color='black', radius=0.001)

        tool2world_mat = tool_site.xmat.reshape(3, 3)
        tool_site_pos = tool_site.xpos
        tool2world_mat_torch = torch.from_numpy(tool2world_mat)
        tool_site_pos_torch = torch.from_numpy(tool_site_pos)

        rope_points_in_tool = mj_transform_points(dcam_site, tool_site, rope_points_in_cam)
        bg_points_in_tool = mj_transform_points(dcam_site, tool_site, bg_points_in_cam)
        bg_points_in_tool_torch = torch.from_numpy(bg_points_in_tool)

        # plot_points_rviz(viz.markers_pub, rope_points_in_tool[::100], idx=0, frame_id='right_tool_site', label='rope points in tool')
        # viz.viz(phy)

        # Take the closest point to the tool tip, which is where the grippers will close
        distances = np.linalg.norm(rope_points_in_tool, axis=-1)
        closest_idx = np.argmin(distances)
        closest_xyz_in_tool = rope_points_in_tool[closest_idx]
        closest_xyz_in_tool_torch = torch.from_numpy(closest_xyz_in_tool)

        # use PCA to extract the long (x) direction of the rope
        _, _, V = torch.pca_lowrank(torch.from_numpy(rope_points_in_tool))
        rope_x_in_tool = V[:, 0]

        # robot_q[-1] -= np.deg2rad(10)
        # pid_to_joint_config(phy, viz, robot_q, sub_time_s=DEFAULT_SUB_TIME_S)
        gripper_q = phy.d.qpos[gripper_q_indices[tool_idx]]
        # TODO: generalize getting a set of surface points for the gripper given tool_idx
        finger_tips_in_world = []
        for g_name in np.array(phy.o.rd.allowed_robot_collision_geoms_names).reshape(2, -1)[tool_idx]:
            finger_tips_in_world.append(phy.d.geom(g_name).xpos)
        finger_tips_in_world = np.array(finger_tips_in_world)
        finger_tips_in_tool = transform_points(np.eye(3), np.zeros(3), tool_site.xmat.reshape(3, 3), tool_site.xpos,
                                               finger_tips_in_world)
        finger_tips_in_tool_torch = torch.from_numpy(finger_tips_in_tool)
        viz.points(f'finger tips', finger_tips_in_tool, 'y', frame_id='right_tool_site', radius=0.02)
        viz.viz(phy)

        # Create a manifold that is 3D Euclidean space plus SO(3) rotations
        SO3 = SpecialOrthogonalGroup(3)
        R3 = Euclidean(3)
        manifold = Product([R3, SO3])

        @pymanopt.function.pytorch(manifold)
        def cost(pos, mat):
            d_to_rope = torch.linalg.norm(pos - closest_xyz_in_tool_torch) * rope_pos_weight
            d_rot = torch.linalg.norm(torch.eye(3) - mat) * d_rot_weight
            new_finger_tips_in_tool = batch_rotate_and_translate(finger_tips_in_tool_torch, mat, pos)
            new_finger_tips_in_world = batch_rotate_and_translate(new_finger_tips_in_tool, tool2world_mat_torch,
                                                                  tool_site_pos_torch)
            # TODO: do a bilinear interpolation to get a more accurate SDF value given a low res grid?
            #  or just make the grid a lower resolution?
            tip_dists = sdf_lookup(sdf_torch, sdf_grad_torch, sdf_origin, sdf_res, new_finger_tips_in_world)
            d_to_contact = torch.exp(-tip_dists.min()) * sdf_weight
            candidate_grasp_x = mat[:, 0]
            x_align = -torch.abs(torch.dot(candidate_grasp_x, rope_x_in_tool))

            # visualize the candidate grasp
            # if not (torch.isnan(pos).any() or torch.isnan(mat).any()):
            #     q_wxyz = np.zeros(4)
            #     mujoco.mju_mat2Quat(q_wxyz, mat.flatten().detach().numpy())
            #     viz.tfw.send_transform(pos, np_wxyz_to_xyzw(q_wxyz), 'right_tool_site', 'candidate_grasp')
            #     contact_color = cm.RdYlGn(np.clip(d_to_contact.detach().numpy(), 0, 1))
            #     viz.points(f'finger tips', new_finger_tips_in_world.detach().numpy(), contact_color, radius=0.02)
            #     try:
            #         sdf_grads = sdf_grad_torch[torch.unbind(point_to_idx(new_finger_tips_in_world, sdf_origin, sdf_res), 1)]
            #         for i, grad_i in enumerate(sdf_grads):
            #             viz.arrow(f'sdf_grad{i}', new_finger_tips_in_world[i], grad_i * sdf_weight, 'w')
            #     except IndexError:
            #         pass
            #     viz.viz(phy)

            total_cost = sum([
                d_to_rope,
                d_rot,
                d_to_contact,
                x_align,
            ])
            return total_cost

        problem = pymanopt.Problem(manifold, cost)

        optimizer = pymanopt.optimizers.SteepestDescent(max_iterations=25, min_step_size=1e-3)
        result = optimizer.run(problem, initial_point=(closest_xyz_in_tool, np.eye(3)))
        best_pos, best_mat = result.point

        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, best_mat.flatten())
        viz.tfw.send_transform(best_pos, np_wxyz_to_xyzw(q_wxyz), 'right_tool_site', 'candidate_grasp')
        viz.viz(phy)

        rope_points = get_rope_points(phy)
        # define the coordinate frame of where we want to grasp the rope
        # rope_idx = 4
        # rope_body = phy.d.body(phy.o.rope.body_indices[rope_idx])
        # rope_grasp_z = np.array([0.0, 0, -1.0])
        # rope_grasp_z = rope_grasp_z / np.linalg.norm(rope_grasp_z)
        # rope_grasp_x = rope_body.xmat.reshape(3, 3)[:, 0]
        # rope_grasp_y = np.cross(rope_grasp_z, rope_grasp_x)
        # rope_grasp_mat = np.stack([rope_grasp_x, rope_grasp_y, rope_grasp_z], axis=1)

        grasp_point_in_tool = cam2tool_mat @ closest_xyz_in_cam
        radius = 0.01
        skeleton = make_ring_skeleton(grasp_point_in_tool, -grasp_z_in_tool, radius, delta_angle=0.5)
        viz.lines(skeleton, "ring", 0, 0.003, 'g')

        tool_z = tool2world_mat[:, 2]
        # b points in the right direction, but b gets bigger when you get closer, but we want it to get smaller
        b = skeleton_field_dir(skeleton, tool_site_pos[None])[0]
        b_normalized = b / np.linalg.norm(b)
        # v here means linear velocity, not to be confused with pixel column
        v_in_tool = b_normalized / np.linalg.norm(b) * v_scale
        v_norm = np.linalg.norm(v_in_tool)
        if v_norm > 0.01:
            v_in_tool = v_in_tool / v_norm * 0.01

        grasp_point_in_world = tool2world_mat @ grasp_point_in_tool
        viz.arrow("tool_z", tool_site_pos, tool_z, 'm')
        viz.arrow("grasp_x", grasp_point_in_world, grasp_x_in_world, 'r')
        viz.arrow("grasp_y", grasp_point_in_world, grasp_y_in_world, 'g')
        viz.arrow("grasp_z", grasp_point_in_world, grasp_z_in_world, 'b')

        v_in_world = tool2world_mat @ v_in_tool
        viz.arrow("v", tool_site_pos, v_in_world, 'w')

        # Now compute the angular velocity using matrix logarithm
        # https://youtu.be/WHn9xJl43nY?t=150
        W_in_tool = logm(grasp_mat_in_tool).real
        w_in_tool = np.array([W_in_tool[2, 1], W_in_tool[0, 2], W_in_tool[1, 0]]) * w_scale

        twist_in_tool = np.concatenate([v_in_tool, w_in_tool])

        Jp = np.zeros((3, phy.m.nv))
        Jr = np.zeros((3, phy.m.nv))
        mujoco.mj_jacSite(phy.m, phy.d, Jp, Jr, tool_site.id)
        J_base = np.concatenate((Jp, Jr), axis=0)
        J_base = J_base[:, phy.m.actuator_trnid[:, 0]]
        # Transform J from base from to gripper frame
        J_gripper = block_diag(tool2world_mat.T, tool2world_mat.T) @ J_base
        J_pinv = np.linalg.pinv(J_gripper)
        # use null-space projection to avoid joint limits
        zero_vels = -get_q(phy) * jnt_lim_avoidance

        ctrl = J_pinv @ twist_in_tool + (np.eye(phy.m.nu) - J_pinv @ J_gripper) @ zero_vels

        # grippers
        lin_speed = np.linalg.norm(v_in_tool)
        ang_speed = np.linalg.norm(w_in_tool)
        gripper_q_mix = 10 * lin_speed + 0.25 * ang_speed
        desired_gripper_q = gripper_q_mix * hp['finger_q_open'] + (1 - gripper_q_mix) * hp['finger_q_closed']
        gripper_gripper_vel = gripper_kp * (desired_gripper_q - gripper_q)
        ctrl[gripper_ctrl_indices[tool_idx]] = gripper_gripper_vel

        # rescale to respect velocity limits
        vmin = phy.m.actuator_ctrlrange[:, 0]
        vmax = phy.m.actuator_ctrlrange[:, 1]
        if np.any(ctrl > vmax):
            offending_joint = np.argmax(ctrl)
        ctrl = ctrl / np.max(ctrl) * vmax[offending_joint]
        if np.any(ctrl < vmin):
            offending_joint = np.argmin(ctrl)
        ctrl = ctrl / np.min(ctrl) * vmin[offending_joint]

        control_step(phy, ctrl, 0.02)

        viz.viz(phy)
        viz.lines(skeleton, "ring", 0, 0.003, 'g')


def get_masked_points(rgbd: MjRGBD, depth: np.ndarray, mask: np.ndarray):
    us, vs = np.where(mask)
    depth = depth[us, vs]
    xs = depth * (vs - rgbd.cx) / rgbd.fpx
    ys = depth * (us - rgbd.cy) / rgbd.fpx
    zs = depth
    xyz_in_cam = np.stack([xs, ys, zs], axis=1)
    return xyz_in_cam


if __name__ == '__main__':
    main()
