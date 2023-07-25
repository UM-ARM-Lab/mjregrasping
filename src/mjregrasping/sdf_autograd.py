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
        shape = torch.tensor(sdf.shape, dtype=sdf.dtype, device=sdf.device)
        is_oob = torch.any((indices < 0) | (indices >= shape), dim=-1)
        batch_oob_i = torch.where(is_oob)[0]
        batch_ib_i = torch.where(~is_oob)[0]
        ib_indices = indices[batch_ib_i]
        oob_d = sdf.max()
        oob_distances_flat = torch.ones_like(batch_oob_i, dtype=points.dtype) * oob_d
        ib_x_i, ib_y_i, ib_z_i = torch.unbind(ib_indices, dim=-1)
        ib_distances_flat = sdf[ib_x_i, ib_y_i, ib_z_i]
        distances = torch.zeros(points.shape[0]).to(points)
        distances[batch_oob_i] = oob_distances_flat
        distances[batch_ib_i] = ib_distances_flat
        ctx.save_for_backward(sdf_grad, origin, res, points, batch_ib_i, batch_oob_i, ib_indices)
        return distances

    @staticmethod
    def backward(ctx, grad_output):
        sdf_grad, origin, res, points, batch_ib_i, batch_oob_i, ib_indices = ctx.saved_tensors
        oob_gradients_flat = torch.zeros([batch_oob_i.shape[0], 3], dtype=points.dtype, device=points.device)
        ib_x_i, ib_y_i, ib_z_i = torch.unbind(ib_indices, dim=-1)
        ib_gradients_flat = sdf_grad[ib_x_i, ib_y_i, ib_z_i]
        gradients = torch.zeros([points.shape[0], 3]).to(points)
        gradients[batch_oob_i] = oob_gradients_flat
        gradients[batch_ib_i] = ib_gradients_flat
        return None, None, None, None, grad_output.unsqueeze(-1) * gradients


sdf_lookup = SDFLookup.apply
