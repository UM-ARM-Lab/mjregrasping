import torch


def normalized(x):
    return x / torch.linalg.norm(x, dim=-1, keepdims=True)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    # https://stackoverflow.com/questions/2827393
    """
    v1_u = normalized(v1)
    v2_u = normalized(v2)
    return torch.arccos(torch.clip(torch.sum(v1_u * v2_u, dim=-1), -1.0, 1.0))


def alignment(v1: torch.tensor, v2: torch.tensor):
    """ same as angle_between but considers opposite directions as aligned """
    return min(angle_between(v1, v2), angle_between(v1, -v2))


def pairwise_squared_distances(a: torch.tensor, b: torch.tensor):
    """
    Adapted from https://github.com/ClayFlannigan/icp
    Computes pairwise distances between to sets of points

    Args:
        a: [b, ..., n, k]
        b:  [b, ..., m, k]

    Returns: [b, ..., n, m]

    """
    a_s = torch.sum(a.square(), dim=-1, keepdim=True)  # [b, ..., n, 1]
    b_s = torch.sum(b.square(), dim=-1, keepdim=True)  # [b, ..., m, 1]
    dist = a_s - 2 * a @ torch.transpose(b, -1, -2) + torch.transpose(b_s, -1, -2)  # [b, ..., n, m]
    return dist
