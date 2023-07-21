import mujoco
import numpy as np
import rerun as rr
from scipy.linalg import logm, block_diag

from arc_utilities import ros_init
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.move_to_joint_config import pid_to_joint_config, get_q
from mjregrasping.movie import MjRenderer, MjRGBD
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.params import hp
from mjregrasping.physics import Physics
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.rviz import plot_points_rviz
from mjregrasping.scenarios import val_untangle
from mjregrasping.viz import make_viz, Viz


@ros_init.with_ros("low_level_grasping")
def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)
    scenario = val_untangle

    rr.init('low_level_grasping')
    rr.connect()

    viz: Viz = make_viz(scenario)

    m = mujoco.MjModel.from_xml_path(str(scenario.xml_path))
    d = mujoco.MjData(m)
    objects = MjObjects(m, scenario.obstacle_name, scenario.robot_data, scenario.rope_name)
    phy = Physics(m, d, objects)

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    tool_idx = 1
    tool_site_name = phy.o.rd.tool_sites[tool_idx]
    tool_site = phy.d.site(tool_site_name)

    w_scale = 0.2
    v_scale = 0.05
    gripper_kp = 1.0
    jnt_lim_avoidance = 0.1

    robot_q = np.array([
        0.1, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.3,  # left gripper
        0.8, 0.0, 0, 0.0, np.pi / 2, -1.1, np.pi / 2,  # right arm
        0.3,  # right gripper
    ])
    pid_to_joint_config(phy, viz, robot_q, sub_time_s=DEFAULT_SUB_TIME_S)
    gripper_ctrl_indices = [phy.m.actuator(a).id for a in phy.o.rd.gripper_actuator_names]
    gripper_q_indices = [phy.m.actuator(a).trnid[0] for a in phy.o.rd.gripper_actuator_names]

    camera_name = "left_hand"
    camera_site_name = f'{camera_name}_cam'
    left_hand_mcam = phy.m.camera(camera_name)
    left_hand_vcam = mujoco.MjvCamera()
    left_hand_vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    left_hand_vcam.fixedcamid = left_hand_mcam.id

    left_hand_r = MjRenderer(phy.m, cam=left_hand_vcam)
    left_hand_rgbd = MjRGBD(left_hand_mcam, left_hand_r)

    while True:
        rgb = left_hand_r.render(d).copy()
        depth = left_hand_r.render(d, depth=True).copy()

        seg = left_hand_r.render_with_flags(d, {mujoco.mjtRndFlag.mjRND_IDCOLOR: 1, mujoco.mjtRndFlag.mjRND_SEGMENT: 1})
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
        rope_us, rope_vs = np.where(rope_mask)
        rope_depth = depth[rope_us, rope_vs]

        rope_uvs = np.stack([rope_us, rope_vs, np.ones_like(rope_us)], axis=0)
        np.linalg.inv(left_hand_rgbd.K) @ rope_uvs

        rope_xs = rope_depth * (rope_vs - left_hand_rgbd.cx) / left_hand_rgbd.fpx
        rope_ys = rope_depth * (rope_us - left_hand_rgbd.cy) / left_hand_rgbd.fpx
        rope_zs = rope_depth
        xyz_in_cam = np.stack([rope_xs, rope_ys, rope_zs], axis=1)

        dcam_site = phy.d.site(camera_site_name)
        xyz_in_world = dcam_site.xpos[None] + xyz_in_cam @ dcam_site.xmat.reshape([3, 3]).T
        plot_points_rviz(viz.markers_pub, xyz_in_world, color='orange', frame_id='world', idx=0, label='rope_xyz_world')

        # TODO: detect a supporting plane, and use that axis if there is one, else use vector from the camera to rope?
        # define the coordinate frame of where we want to grasp the rope
        radius = 0.01
        rope_idx = 4
        rope_body = phy.d.body(phy.o.rope.body_indices[rope_idx])
        rope_grasp_z = np.array([0.0, 0, -1.0])
        rope_grasp_z = rope_grasp_z / np.linalg.norm(rope_grasp_z)
        rope_grasp_x = rope_body.xmat.reshape(3, 3)[:, 0]
        rope_grasp_y = np.cross(rope_grasp_z, rope_grasp_x)
        rope_grasp_mat = np.stack([rope_grasp_x, rope_grasp_y, rope_grasp_z], axis=1)
        rope_points = get_rope_points(phy)
        rope_point = rope_points[rope_idx]
        skeleton = make_ring_skeleton(rope_point, -rope_grasp_z, radius, delta_angle=0.5)

        tool_site_pos = tool_site.xpos
        tool_mat = tool_site.xmat.reshape(3, 3)
        tool_z = tool_mat[:, 2]
        # b points in the right direction, but b gets bigger when you get closer, but we want it to get smaller
        b = skeleton_field_dir(skeleton, tool_site_pos[None])[0]
        b_normalized = b / np.linalg.norm(b)
        v = b_normalized / np.linalg.norm(b) * v_scale
        v_norm = np.linalg.norm(v)
        if v_norm > 0.01:
            v = v / v_norm * 0.01
        viz.arrow("tool_z", tool_site_pos, tool_z, 'm')
        viz.arrow("grasp_x", rope_point, rope_grasp_x, 'r')
        viz.arrow("grasp_y", rope_point, rope_grasp_y, 'g')
        viz.arrow("grasp_z", rope_point, rope_grasp_z, 'b')

        viz.arrow("v", tool_site_pos, v, 'w')

        # Now compute the angular velocity using matrix logarithm
        # https://youtu.be/WHn9xJl43nY?t=150
        # First transform the grasp matrix to the gripper frame
        grasp_mat_in_tool = tool_mat.T @ rope_grasp_mat
        W_in_gripper = logm(grasp_mat_in_tool).real
        w_in_gripper = np.array([W_in_gripper[2, 1], W_in_gripper[0, 2], W_in_gripper[1, 0]]) * w_scale

        v_in_gripper = tool_mat.T @ v
        twist_in_gripper = np.concatenate([v_in_gripper, w_in_gripper])

        Jp = np.zeros((3, phy.m.nv))
        Jr = np.zeros((3, phy.m.nv))
        mujoco.mj_jacSite(phy.m, phy.d, Jp, Jr, tool_site.id)
        J_base = np.concatenate((Jp, Jr), axis=0)
        J_base = J_base[:, phy.m.actuator_trnid[:, 0]]
        # Transform J from base from to gripper frame
        J_gripper = block_diag(tool_mat.T, tool_mat.T) @ J_base
        J_pinv = np.linalg.pinv(J_gripper)
        # use null-space projection to avoid joint limits
        zero_vels = -get_q(phy) * jnt_lim_avoidance

        ctrl = J_pinv @ twist_in_gripper + (np.eye(phy.m.nu) - J_pinv @ J_gripper) @ zero_vels

        # grippers
        current_gripper_q = phy.d.qpos[gripper_q_indices[tool_idx]]
        lin_speed = np.linalg.norm(v_in_gripper)
        ang_speed = np.linalg.norm(w_in_gripper)
        gripper_q_mix = 10 * lin_speed + 0.25 * ang_speed
        desired_gripper_q = gripper_q_mix * hp['finger_q_open'] + (1 - gripper_q_mix) * hp['finger_q_closed']
        gripper_gripper_vel = gripper_kp * (desired_gripper_q - current_gripper_q)
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


if __name__ == '__main__':
    main()
