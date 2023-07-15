import pysdf_tools

from mjregrasping.homotopy_checker import CollisionChecker


class SDFCollisionChecker(CollisionChecker):

    def __init__(self, sdf: pysdf_tools.SignedDistanceField):
        super().__init__()
        self.sdf = sdf

    def is_collision(self, point):
        sdf_value = self.sdf.GetValueByCoordinates(*point)[0]
        in_collision = sdf_value < -self.get_resolution() / 2
        return in_collision

    def get_resolution(self):
        return self.sdf.GetResolution()
