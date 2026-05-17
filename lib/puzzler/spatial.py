import numpy as np
import puzzbin

class NearestPointImage:

    def __init__(self, data, radius=0):
        bbox = np.array((np.min(data,axis=0)-radius, np.max(data,axis=0)+radius+1))
        self.ll = bbox[0]
        self.ur = bbox[1]
        self.data = data
        self.image = puzzbin.compute_nearest_point_image(bbox, data)[1]

    def query(self, points, distance_upper_bound=None):
        # see also puzzler.align.DistanceImage.query()
        
        h, w = self.image.shape

        rows = np.int32(np.clip(points[:,1] - self.ll[1], a_min=0, a_max=h-1))
        cols = np.int32(np.clip(points[:,0] - self.ll[0], a_min=0, a_max=w-1))

        indices = self.image[rows, cols]
        dist = np.linalg.norm(points - self.data[indices], axis=1)
        return dist, indices
