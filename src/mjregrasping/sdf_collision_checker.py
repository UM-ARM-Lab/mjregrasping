import pysdf_tools

from mjregrasping.homotopy_checker import CollisionChecker, AllowablePenetration


class SDFCollisionChecker(CollisionChecker):

    def __init__(self, sdf: pysdf_tools.SignedDistanceField):
        super().__init__()
        self.sdf = sdf

    def is_collision(self, point, allowable_penetration: AllowablePenetration = AllowablePenetration.NONE):
        if allowable_penetration == AllowablePenetration.NONE:
            d = 0
        elif allowable_penetration == AllowablePenetration.HALF_CELL:
            d = self.sdf.GetResolution() / 2
        elif allowable_penetration == AllowablePenetration.FULL_CELL:
            d = self.sdf.GetResolution()
        else:
            raise NotImplementedError(allowable_penetration)
        sdf_value = self.sdf.GetValueByCoordinates(*point)[0]
        in_collision = sdf_value < -d
        return in_collision

    def get_resolution(self):
        return self.sdf.GetResolution()
