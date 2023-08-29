from pathlib import Path

import mujoco
import numpy as np
import pymanopt
import pysdf_tools
import rerun as rr
import torch
from matplotlib import cm
from pymanopt.manifolds import SpecialOrthogonalGroup

from arc_utilities import ros_init
from mjregrasping.homotopy_utils import make_ring_skeleton, skeleton_field_dir
from mjregrasping.jacobian_ctrl import get_jacobian_ctrl
from mjregrasping.mjsaver import load_data_and_eq, save_data_and_eq
from mjregrasping.movie import MjRenderer, MjRGBD
from mjregrasping.mujoco_objects import MjObjects
from mjregrasping.my_transforms import mj_transform_points, np_wxyz_to_xyzw, transform_points
from mjregrasping.params import hp
from mjregrasping.physics import Physics, rescale_ctrl
from mjregrasping.real_val import RealValCommander
from mjregrasping.rollout import control_step, DEFAULT_SUB_TIME_S
from mjregrasping.scenarios import val_untangle, setup_untangle
from mjregrasping.sdf_autograd import point_to_idx as point_to_idx_torch
from mjregrasping.sdf_autograd import sdf_lookup
from mjregrasping.viz import make_viz, Viz
from mjregrasping.voxelgrid import point_to_idx as point_to_idx_np
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
    setup_untangle(phy, viz)
    # from mjregrasping.mjsaver import save_data_and_eq
    # save_data_and_eq(phy, Path(f"states/grasp/on_surface_right.pkl"))

    # d = load_data_and_eq(m, path=Path("states/grasp/on_surface_left.pkl"))
    # d = load_data_and_eq(m, path=Path("states/grasp/free_space_right.pkl"))
    # d = load_data_and_eq(m, path=Path("states/grasp/on_surface_right.pkl"))
    tool_idx = 0
    phy = Physics(m, d, objects)

    val = RealValCommander(phy.o.robot)

    # TODO this is too slow. We either need to use a lower res grid or do something completely different.
    #  what if we just used the BG points and instead of computing a full SDF, we just say that gradient points towards
    #  the camera for any points "behind" an observed BG point. This basically grids up the R3 into "cones"?
    sdf = pysdf_tools.SignedDistanceField.LoadFromFile(str(scenario.sdf_path))
    sdf_np = np.array(sdf.GetRawData(), dtype=np.float64)
    sdf_np = sdf_np.reshape([sdf.GetNumXCells(), sdf.GetNumYCells(), sdf.GetNumZCells()])
    print("Computing gradient...")
    sdf_grad_np = get_gradient(sdf, dtype=np.float64)
    sdf_origin_np = sdf.GetOriginTransform().translation().astype(np.float64)
    sdf_res_np = sdf.GetResolution()

    mujoco.mj_forward(phy.m, phy.d)
    viz.viz(phy)

    tool_site_name = phy.o.rd.tool_sites[tool_idx]
    tool_site = phy.d.site(tool_site_name)

    v_scale = 0.05
    max_v_norm = 0.03
    gripper_kp = 5.0

    gripper_ctrl_indices = [phy.m.actuator(a).id for a in phy.o.rd.gripper_actuator_names]
    gripper_q_indices = [phy.m.actuator(a).trnid[0] for a in phy.o.rd.gripper_actuator_names]
    tool_frame_name = f'{phy.o.rd.tool_sites[tool_idx]}_site'

    camera_name = phy.o.rd.camera_names[tool_idx]
    camera_site_name = f'{camera_name}_cam'
    hand_mcam = phy.m.camera(camera_name)
    hand_vcam = mujoco.MjvCamera()
    hand_vcam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    hand_vcam.fixedcamid = hand_mcam.id

    hand_r = MjRenderer(phy.m, cam=hand_vcam)
    hand_rgbd = MjRGBD(hand_mcam, hand_r)

    t = 0
    while True:
        # TODO: use learned instance segmentation here
        depth, rgb, rope_mask = mj_get_rgbd_and_rope_mask(d, hand_r, phy)

        rr.log_image("img/rgb", rgb)
        rr.log_image("img/depth", np.clip(depth, 0, 1))
        rr.log_image("img/rope_mask", rope_mask * 255)

        # v in pixel space is x in camera space
        # u in pixel space is y in camera space
        rope_points_in_cam = get_masked_points(hand_rgbd, depth, rope_mask)

        dcam_site = phy.d.site(camera_site_name)

        tool2world_mat = tool_site.xmat.reshape(3, 3)
        tool_site_pos = tool_site.xpos

        rope_points_in_tool = mj_transform_points(dcam_site, tool_site, rope_points_in_cam)
        from mjregrasping.rviz import plot_points_rviz
        plot_points_rviz(viz.markers_pub, rope_points_in_tool[::100], idx=0, frame_id='right_tool_site',
                         label='rope points in tool', s=0.1)
        viz.viz(phy)

        gripper_q = phy.d.qpos[gripper_q_indices[tool_idx]]
        # TODO: generalize getting a set of surface points for the gripper given tool_idx
        finger_tips_in_world = []
        for g_name in np.array(phy.o.rd.allowed_robot_collision_geoms_names).reshape(2, -1)[tool_idx]:
            finger_tips_in_world.append(phy.d.geom(g_name).xpos)
        finger_tips_in_world = np.array(finger_tips_in_world)
        finger_tips_in_tool = transform_points(np.eye(3), np.zeros(3), tool_site.xmat.reshape(3, 3), tool_site.xpos,
                                               finger_tips_in_world)

        if t % 5 == 0:
            from time import perf_counter
            t0 = perf_counter()
            grasp_mat_in_tool, grasp_pos_in_tool = get_best_grasp(finger_tips_in_tool, tool2world_mat, tool_site_pos,
                                                                  rope_points_in_tool, sdf_np, sdf_grad_np,
                                                                  sdf_origin_np, sdf_res_np, tool_frame_name,
                                                                  phy=None, viz=None)
            print(f'Computing grasp took {perf_counter() - t0:.3f} seconds')

        radius = 0.01
        grasp_z_in_tool = grasp_mat_in_tool[:, 2]
        skeleton = make_ring_skeleton(grasp_pos_in_tool, -grasp_z_in_tool, radius, delta_angle=0.5)

        v_in_tool = get_v_in_tool(max_v_norm, skeleton, v_scale)
        v_in_world = tool2world_mat @ v_in_tool

        ctrl, w_in_tool = get_jacobian_ctrl(phy, tool_site, grasp_mat_in_tool, v_in_tool)

        # grippers
        lin_speed = np.linalg.norm(v_in_tool)
        ang_speed = np.linalg.norm(w_in_tool)
        gripper_q_mix = np.clip(12 * lin_speed + 0.3 * ang_speed, 0, 1)
        desired_gripper_q = gripper_q_mix * hp['finger_q_open'] + (1 - gripper_q_mix) * hp['finger_q_closed']
        gripper_gripper_vel = gripper_kp * (desired_gripper_q - gripper_q)
        ctrl[gripper_ctrl_indices[tool_idx]] = gripper_gripper_vel

        # rescale to respect velocity limits
        ctrl = rescale_ctrl(phy, ctrl)

        control_step(phy, ctrl, 0.02)

        # send commands to the robot
        val.send_vel_command(phy.m, ctrl)

        viz.lines(skeleton, "ring", 0, 0.007, 'g', frame_id=tool_frame_name)
        viz.arrow("v", tool_site_pos, v_in_world, 'w')
        q_wxyz = np.zeros(4)
        mujoco.mju_mat2Quat(q_wxyz, grasp_mat_in_tool.flatten())
        viz.tfw.send_transform(grasp_pos_in_tool, np_wxyz_to_xyzw(q_wxyz), tool_frame_name, 'grasp_pose_in_tool')
        viz.viz(phy)

        if is_grasp_complete(gripper_q, desired_gripper_q):
            print("Grasp successful!")
            break

    val.stop()


def get_best_grasp(finger_tips_in_tool, tool2world_mat, tool_site_pos, rope_points_in_tool, sdf_np, sdf_grad_np,
                   sdf_origin_np, sdf_res_np, tool_frame_name, phy=None, viz=None):
    # Hyperparameters
    sdf_weight = 0.02
    sdf_exp = 50
    align_weight = 0.1

    # Context needed to compute the cost
    finger_tips_in_tool_torch = torch.from_numpy(finger_tips_in_tool)
    tool2world_mat_torch = torch.from_numpy(tool2world_mat)
    tool_site_pos_torch = torch.from_numpy(tool_site_pos)
    sdf_torch = torch.from_numpy(sdf_np)
    sdf_grad_torch = torch.from_numpy(sdf_grad_np)
    sdf_origin = torch.tensor(sdf_origin_np, dtype=torch.float64)
    sdf_res = torch.tensor(sdf_res_np, dtype=torch.float64)
    rope_points_in_tool_torch = torch.from_numpy(rope_points_in_tool)

    # Take the closest point to the tool tip, which is where the grippers will close
    distances = np.linalg.norm(rope_points_in_tool, axis=-1)
    closest_idx = np.argmin(distances)
    grasp_pos_in_tool = rope_points_in_tool[closest_idx]
    grasp_pos_in_tool_torch = torch.from_numpy(grasp_pos_in_tool)

    # use PCA to extract the long (x) direction of the rope
    _, _, V = torch.pca_lowrank(rope_points_in_tool_torch)
    rope_x_in_tool = V[:, 0]
    grasp_mat_in_tool = np.eye(3)

    # Outer loop, we adjust the closest_xyz_in_tool_torch
    for _ in range(3):
        # Inner loop, optimize orientation. Using pymanopt to optimize both did not work well for me
        # Create a manifold that is 3D Euclidean space plus SO(3) rotations
        SO3 = SpecialOrthogonalGroup(3)
        manifold = SO3

        @pymanopt.function.pytorch(manifold)
        def cost(mat):
            new_finger_tips_in_tool = batch_rotate_and_translate(finger_tips_in_tool_torch, mat,
                                                                 grasp_pos_in_tool_torch)
            new_finger_tips_in_world = batch_rotate_and_translate(new_finger_tips_in_tool, tool2world_mat_torch,
                                                                  tool_site_pos_torch)
            # TODO: do a bilinear interpolation to get a more accurate SDF value given a low res grid?
            #  or just make the grid a lower resolution?
            tip_dists = sdf_lookup(sdf_torch, sdf_grad_torch, sdf_origin, sdf_res, new_finger_tips_in_world)
            d_to_contact = torch.exp(-sdf_exp * tip_dists).max() * sdf_weight
            candidate_grasp_x = mat[:, 0]
            x_align = -torch.abs(torch.dot(candidate_grasp_x, rope_x_in_tool)) * align_weight

            # visualize the candidate grasp
            if viz and phy and not (torch.isnan(mat).any()):
                q_wxyz = np.zeros(4)
                mujoco.mju_mat2Quat(q_wxyz, mat.flatten().detach().numpy())
                viz.tfw.send_transform(grasp_pos_in_tool, np_wxyz_to_xyzw(q_wxyz), tool_frame_name, 'candidate_grasp')
                contact_color = cm.RdYlGn(np.clip(1 - d_to_contact.detach().numpy() * 0.3, 0, 1))
                viz.points(f'finger tips', new_finger_tips_in_world.detach().numpy(), contact_color, radius=0.02)
                try:
                    sdf_grads = sdf_grad_torch[
                        torch.unbind(point_to_idx_torch(new_finger_tips_in_world, sdf_origin, sdf_res), 1)]
                    for i, grad_i in enumerate(sdf_grads):
                        viz.arrow(f'sdf_grad{i}', new_finger_tips_in_world[i], grad_i * sdf_weight, 'w')
                except IndexError:
                    pass
                viz.viz(phy)

            total_cost = sum([
                d_to_contact,
                x_align,
            ])
            return total_cost

        problem = pymanopt.Problem(manifold, cost)
        verbosity = 2 if viz else 0
        optimizer = pymanopt.optimizers.SteepestDescent(max_iterations=10, min_step_size=1e-3, verbosity=verbosity)
        result = optimizer.run(problem, initial_point=grasp_mat_in_tool)
        grasp_mat_in_tool = result.point

        new_finger_tips_in_tool = batch_rotate_and_translate(finger_tips_in_tool, grasp_mat_in_tool, grasp_pos_in_tool)
        new_finger_tips_in_world = batch_rotate_and_translate(new_finger_tips_in_tool, tool2world_mat, tool_site_pos)
        # Try to move the rope deeper in between the fingers of the gripper so the grasp is more stable
        good_grasp_weight = 0.01
        good_grasp_grad = grasp_mat_in_tool[:, 2] * good_grasp_weight
        # But also don't move out of penetration if the SDF value is low
        sdf_indices = tuple(point_to_idx_np(new_finger_tips_in_world, sdf_origin_np, sdf_res_np).T)
        tip_dists = sdf_np[sdf_indices]
        min_tip_dist_i = tip_dists.argmin()
        min_tip_dist = tip_dists.min()
        sdf_grad_in_world = sdf_grad_np[sdf_indices][min_tip_dist_i] * min_tip_dist
        sdf_grad_in_tool = tool2world_mat.T @ sdf_grad_in_world
        if min_tip_dist <= sdf_res_np:
            grasp_mat_in_tool += sdf_grad_in_tool
        else:
            grasp_pos_in_tool += good_grasp_grad

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


def get_v_in_tool(max_v_norm, skeleton, v_scale):
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
    return gripper_q < hp['finger_q_closed'] * 1.15 and desired_gripper_q < hp['finger_q_closed'] * 1.15


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
