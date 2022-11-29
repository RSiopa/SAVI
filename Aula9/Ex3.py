#!/usr/bin/env python3

from copy import deepcopy
from matplotlib import cm
import numpy as np
import open3d as o3d

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [6.5291471481323242, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.53799727367026606, -0.75773014686234363, 0.36932906473676308],
                "lookat": [-3.889649208321611, -2.5907102147187757, 2.5114310558736914],
                "up": [-0.18430194401548972, 0.32180358757709787, 0.92869545301709133],
                "zoom": 0.32119999999999999
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


class PlaneDetection:
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r, g, b):
        self.inlier_cloud.paint_uniform_color([r, g, b])

    def segment(self, distance_threshold=0.25, ransac_n=3, num_iterations=100):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) + ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0'
        return text


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    point_cloud_original = o3d.io.read_point_cloud('factory_without_ground.ply')

    number_of_planes = 6
    colormap = cm.Pastel1(list(range(0, number_of_planes)))

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    point_cloud = deepcopy(point_cloud_original)
    planes = []
    while True:     # Run consecutive plane detections

        plane = PlaneDetection(point_cloud)     # Create a new plane instance
        point_cloud = plane.segment()   # New point cloud are the outliers of this iteration
        print(plane)

        # Colorization using a colormap
        idx_color = len(planes)
        color = colormap[idx_color, 0:3]
        plane.colorizeInliers(r=color[0], g=color[1], b=color[2])

        planes.append(plane)

        if len(planes) >= number_of_planes:
            break

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    # Create a list of entities to draw
    entities = [x.inlier_cloud for x in planes]
    entities.append(point_cloud)

    o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])


if __name__ == "__main__":
    main()
