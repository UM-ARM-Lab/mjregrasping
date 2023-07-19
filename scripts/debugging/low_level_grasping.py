import mujoco
import transformations
from scipy.linalg import logm, block_diag
import numpy as np
import rerun as rr

from arc_utilities import ros_init
from mjregrasping.goal_funcs import get_rope_points
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.ik import full_jacobian
from mjregrasping.move_to_joint_config import pid_to_joint_config
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import angle_between
from mjregrasping.physics import Physics
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
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

    rope_points = get_rope_points(phy)
    rope_idx = 4
    rope_point = rope_points[rope_idx]
    rope_body = phy.d.body(phy.o.rope.body_indices[rope_idx])
    # TODO: detect a supporting plane, and use that axis if there is one, else use vector from the camera to rope?

    # define the coordinate frame of where we want to grasp the rope
    rope_grasp_z = np.array([0.0, 0, -1.0])
    rope_grasp_z = rope_grasp_z / np.linalg.norm(rope_grasp_z)
    rope_grasp_x = rope_body.xmat.reshape(3, 3)[:, 0]
    rope_grasp_y = np.cross(rope_grasp_z, rope_grasp_x)
    rope_grasp_mat = np.stack([rope_grasp_x, rope_grasp_y, rope_grasp_z], axis=1)

    radius = 0.02
    skeleton = make_ring_skeleton(rope_point, -rope_grasp_z, radius, delta_angle=0.5)
    tool_idx = 1
    tool_site_name = phy.o.rd.tool_sites[tool_idx]
    tool_site = phy.d.site(tool_site_name)
    gripper_body = phy.d.body(phy.o.rd.tool_bodies[1])

    w_scale = 0.1
    v_scale = 0.02
    gripper_kp = 1.0

    robot_q = np.array([
        0.1, 0.4,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.3,  # left gripper
        0.6, 0.0, 0, 0.0, np.pi / 2, -0.8, np.pi / 2,  # right arm
        0.3,  # right gripper
    ])
    robot_q = np.array([
        0.0, 0.1,  # torso
        -0.4, 0.3, -0.3, 0.5, 0, 0, 0,  # left arm
        0.3,  # left gripper
        np.pi/2, 0.0, 0, 0.0, 0, 0, 0,  # right arm
        0.3,  # right gripper
    ])
    # pid_to_joint_config(phy, viz, robot_q, sub_time_s=DEFAULT_SUB_TIME_S)
    gripper_ctrl_indices = [phy.m.actuator(a).id for a in phy.o.rd.gripper_actuator_names]
    gripper_q_indices = [phy.m.actuator(a).trnid[0] for a in phy.o.rd.gripper_actuator_names]
    start_gripper_pos = tool_site.xpos.copy()

    viz.viz(phy)
    viz.lines(skeleton, "ring", 0, 0.003, 'g')

    while True:
        tool_site_pos = tool_site.xpos
        gripper_body_pos = gripper_body.xpos
        gripper_mat = tool_site.xmat.reshape(3, 3)
        gripper_z = gripper_mat[:, 2]
        # b points in the right direction, but b gets bigger when you get closer, but we want it to get smaller
        b = skeleton_field_dir(skeleton, tool_site_pos[None])[0]
        b_normalized = b / np.linalg.norm(b)
        v = b_normalized / np.linalg.norm(b) * v_scale
        v_norm = np.linalg.norm(v)
        if v_norm > 0.01:
            v = v / v_norm * 0.01
        viz.arrow("gripper_z", tool_site_pos, gripper_z, 'm')
        viz.arrow("grasp_x", rope_point, rope_grasp_x, 'r')
        viz.arrow("grasp_y", rope_point, rope_grasp_y, 'g')
        viz.arrow("grasp_z", rope_point, rope_grasp_z, 'b')

        # First transform the grasp matrix to the gripper frame
        grasp_mat_in_gripper = rope_grasp_mat.T @ gripper_mat
        z_error = angle_between(rope_grasp_z, gripper_z)
        v_mix = z_error / np.pi
        start_v = start_gripper_pos - gripper_body_pos
        v = v_mix * start_v + (1 - v_mix) * v
        v = start_v
        # viz.arrow("v", tool_site_pos, v, 'w')

        # Now compute the angular velocity using matrix logarithm
        # https://youtu.be/WHn9xJl43nY?t=150
        W_in_gripper = logm(grasp_mat_in_gripper).real
        w_in_gripper = np.array([W_in_gripper[2, 1], W_in_gripper[0, 2], W_in_gripper[1, 0]]) * w_scale
        # Now transform w to base/world frame
        w = gripper_mat @ w_in_gripper

        twist = np.concatenate([v, w])

        # J_base = full_jacobian(phy, gripper_body.id, ee_offset=np.array([0, 0, 0.181]))
        J_base = full_jacobian(phy, gripper_body.id, ee_offset=np.zeros(3))
        viz.arrow(f'Jp', gripper_body_pos, J_base[:3, 0], 'r')
        # for i, Jp_i in enumerate(J_base.T):
        #     viz.arrow(f'Jp', gripper_body_pos, Jp_i[:3], 'r')
        # Transform J from base from to gripper frame
        # J_gripper = block_diag(gripper_mat.T, gripper_mat.T) @ J_base
        # print(J_base[:, 0], J_base[:, 1])
        # ctrl = np.linalg.pinv(J_gripper) @ np.concatenate([np.zeros(3), np.array([1.0, 0, 0])])

        # project the rotation into the null space of position. in other words, find the joint velocities that
        # cause the gripper to rotate without moving
        # J_pos = J[:3, :]
        # ctrl = (np.eye(phy.m.nu) - np.linalg.pinv(J_pos) @ J_pos) @ rot_ctrl
        # J_pos @ rot_ctrl

        # # grippers
        # current_gripper_q = phy.d.qpos[gripper_q_indices[tool_idx]]
        # lin_speed = np.linalg.norm(v)
        # ang_speed = np.linalg.norm(w)
        # gripper_desired_open = lin_speed > 0.001 or ang_speed > 0.01
        # desired_gripper_q = 0.3 if gripper_desired_open else 0.05
        # gripper_gripper_vel = gripper_kp * (desired_gripper_q - current_gripper_q)
        # ctrl[gripper_ctrl_indices[tool_idx]] = gripper_gripper_vel
        #
        # # rescale to respect velocity limits
        # vmin = phy.m.actuator_ctrlrange[:, 0]
        # vmax = phy.m.actuator_ctrlrange[:, 1]
        # if np.any(ctrl > vmax):
        #     offending_joint = np.argmax(ctrl)
        #     ctrl = ctrl / np.max(ctrl) * vmax[offending_joint]
        # if np.any(ctrl < vmin):
        #     offending_joint = np.argmin(ctrl)
        #     ctrl = ctrl / np.min(ctrl) * vmin[offending_joint]

        ctrl = np.zeros(phy.m.nu)
        ctrl[0] = 0.4
        control_step(phy, ctrl, 0.05)

        viz.viz(phy)
        viz.lines(skeleton, "ring", 0, 0.003, 'g')


if __name__ == '__main__':
    main()
