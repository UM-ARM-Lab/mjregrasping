import torch
import pysdf_tools


def point_to_idx(points, origin_point, res):
    # round helps with stupid numerics issues
    return torch.round((points - origin_point) / res).long()


class SDFLookup(torch.autograd.Function):

    @staticmethod
    def forward(ctx, sdf: torch.tensor, sdf_grad: torch.tensor, origin: torch.tensor, res: torch.tensor, points):
        """

        Args:
            ctx:
            sdf: [x, y, z]
            sdf_grad: [x, y, z]
            origin: [3] the xyz position of cell index [0,0,0]
            res: float, in meters
            points: [n, 3]

        Variable naming:
            _v means valid, as in not out-of-bounds, and _i means index. oob means out of bounds, ib means in bounds
        """
        indices = point_to_idx(points, origin, res)
        shape = torch.tensor(sdf.sdf.shape[1:], dtype=sdf.sdf.dtype, device=sdf.sdf.device)
        is_oob = torch.any((indices < 0) | (indices >= shape), dim=-1)
        batch_oob_i, time_oob_i = torch.where(is_oob)
        batch_ib_i, time_ib_i = torch.where(~is_oob)
        ib_indices = indices[batch_ib_i, time_ib_i]
        oob_d = 999
        oob_distances_flat = torch.ones_like(batch_oob_i, dtype=points.dtype) * oob_d
        ib_x_i, ib_y_i, ib_z_i = torch.unbind(ib_indices, dim=-1)
        ib_zeros_i = torch.zeros_like(ib_x_i)
        ib_distances_flat = sdf.sdf[ib_zeros_i, ib_x_i, ib_y_i, ib_z_i]
        distances = torch.zeros([points.shape[0], points.shape[1]]).to(points)
        distances[batch_oob_i, time_oob_i] = oob_distances_flat
        distances[batch_ib_i, time_ib_i] = ib_distances_flat
        ctx.save_for_backward(points, grad, batch_ib_i, time_ib_i, batch_oob_i, time_oob_i, ib_indices)
        return distances

    @staticmethod
    def backward(ctx, grad_output):
        points, sdf_grad, batch_ib_i, time_ib_i, batch_oob_i, time_oob_i, ib_indices = ctx.saved_tensors
        oob_gradients_flat = torch.zeros([batch_oob_i.shape[0], 3], dtype=points.dtype, device=points.device)
        ib_x_i, ib_y_i, ib_z_i = torch.unbind(ib_indices, dim=-1)
        ib_zeros_i = torch.zeros_like(ib_x_i)
        ib_gradients_flat = sdf_grad[ib_zeros_i, ib_x_i, ib_y_i, ib_z_i]
        gradients = torch.zeros([points.shape[0], points.shape[1], 3]).to(points)
        gradients[batch_oob_i, time_oob_i] = oob_gradients_flat
        gradients[batch_ib_i, time_ib_i] = ib_gradients_flat
        return None, grad_output.unsqueeze(-1) * gradients


sdf_lookup = SDFLookup.apply
