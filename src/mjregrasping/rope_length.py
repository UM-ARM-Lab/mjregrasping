def get_rope_length(phy):
    rope_length = phy.m.geom(phy.o.rope.geom_indices[0]).size[1] * len(phy.o.rope.geom_indices) * 2
    return rope_length
