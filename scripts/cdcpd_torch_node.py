import argparse

import torch
from cdcpd_torch.core.deformable_object_configuration import RopeConfiguration
from cdcpd_torch.core.tracking_map import TrackingMap
from cdcpd_torch.core.visibility_prior import visibility_prior
from cdcpd_torch.data_utils.types.grippers import GrippersInfo, GripperInfoSingle
from cdcpd_torch.modules.cdcpd_module_arguments import CDCPDModuleArguments
from cdcpd_torch.modules.cdcpd_network import CDCPDModule
from cdcpd_torch.modules.cdcpd_parameters import CDCPDParamValues
from cdcpd_torch.modules.post_processing.configuration import PostProcConfig, PostProcModuleChoice

import rospy
from visualization_msgs.msg import MarkerArray


# from gnn.mab import KFMANDB
# from gnn.online_model import ModelSelect, get_model
# from point_env.point_model import CCGP
# from pytorch_mppi import mppi
# from replay import goal_publish
# from rope_utils.val import velocity_control
# from rope_utils.visual_prior import get_point_cloud


def cdcpd_helper(tracking_map, cdcpd_module, depth, mask, intrinsic, grip_points, rope_points_in_world, K):
    verts = tracking_map.form_vertices_cloud()
    verts_np = verts.detach().numpy()
    Y_emit_prior = visibility_prior(verts_np, depth, mask, intrinsic[:3, :3], K)
    Y_emit_prior = torch.from_numpy(Y_emit_prior.reshape(-1, 1)).to(torch.double)

    grasped_points = GrippersInfo()
    for idx, point in grip_points:
        grasped_points.append(
            GripperInfoSingle(fixed_pt_pred=point, grasped_vertex_idx=idx),
        )
    model_input = CDCPDModuleArguments(tracking_map, rope_points_in_world, gripper_info=grasped_points)
    model_input.set_Y_emit_prior(Y_emit_prior)

    cdcpd_module_arguments = cdcpd_module(model_input)

    post_processing_out = cdcpd_module_arguments.get_Y_opt()
    tracking_map.update_def_obj_vertices(post_processing_out)

    verts = tracking_map.form_vertices_cloud()
    cur_state_estimate = verts.detach().numpy().T

    return cur_state_estimate


NUM_TRACKED_POINTS = 25
MAX_ROPE_LENGTH = .768

USE_DEBUG_MODE = False

ALPHA = 0.5
BETA = 1.0
LAMBDA = 1.0
K = 100.0
ZETA = 50.0
OBSTACLE_COST_WEIGHT = 0.02
FIXED_POINTS_WEIGHT = 100.0
OBJECTIVE_VALUE_THRESHOLD = 1.0
MIN_DISTANCE_THRESHOLD = 0.04


class CDCPDTorchNode:

    def __init__(self):
        pass

    def cb(self, msg):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--length_mult', type=float, default=1)
    parser.add_argument('--model', type=str, default='NN')
    parser.add_argument('--kink', action='store_true')
    parser.add_argument('--use_kink_model', action='store_true')
    parser.add_argument('--ukf', action='store_true')
    parser.add_argument('--batched_gp', action='store_true')
    parser.add_argument('--lc', type=int, default=2)
    parser.add_argument('--mab_opt', type=float, default=-1)
    parser.add_argument('--cdcpd', action='store_true', help='Enable CDCPD for perception')
    parser.add_argument('--gpis', action='store_true', help='Enable gpis')

    args = parser.parse_args()

    rospy.init_node("cdcpd_torch_node")

    cdcpd_pred_pub = rospy.Publisher('cdcpd_pred', MarkerArray, queue_size=10)

    device = 'cuda'
    rope_start_pos = torch.tensor(rope_start_pos).to(device)
    rope_end_pos = torch.tensor(rope_end_pos).to(device)
    grip_constraints = [(0, rope_start_pos), (24, rope_end_pos)]

    def_obj_config = RopeConfiguration(NUM_TRACKED_POINTS, MAX_ROPE_LENGTH, rope_start_pos,
                                       rope_end_pos)
    def_obj_config.initialize_tracking()

    tracking_map = TrackingMap()
    tracking_map.add_def_obj_configuration(def_obj_config)

    postproc_config = PostProcConfig(module_choice=PostProcModuleChoice.PRE_COMPILED)

    param_vals = CDCPDParamValues()
    param_vals.alpha_.val = ALPHA
    param_vals.beta_.val = BETA
    param_vals.lambda_.val = LAMBDA
    param_vals.zeta_.val = ZETA
    param_vals.obstacle_cost_weight_.val = OBSTACLE_COST_WEIGHT
    param_vals.obstacle_constraint_min_dist_threshold.val = MIN_DISTANCE_THRESHOLD

    grasped_points = GrippersInfo()
    for idx, point in grip_constraints:
        grasped_points.append(
            GripperInfoSingle(fixed_pt_pred=point, grasped_vertex_idx=idx),
        )

    cdcpd_module = CDCPDModule(deformable_object_config_init=def_obj_config,
                               param_vals=param_vals,
                               postprocessing_option=postproc_config,
                               debug=USE_DEBUG_MODE, device=device)
    cdcpd_module = cdcpd_module.eval()

    current_estimate = cdcpd_helper(tracking_map, cdcpd_module, depth, mask, intrinsic, grip_constraints,
                                    previous_estimate, K)
    plot_rope_rviz(cdcpd_pred_pub, cur_state_estimate, 333, 'cdcpd_pred', color='r', frame_id='world', s=2)


if __name__ == "__main__":
    main()
