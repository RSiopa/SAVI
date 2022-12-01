#!/usr/bin/env python3

from copy import deepcopy
from more_itertools import locate
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

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    point_cloud = deepcopy(point_cloud_original)

    # Downsampling using voxel grid filter
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.1)

    # Clustering
    cluster_idxs = list(point_cloud_downsampled.cluster_dbscan(eps=0.45, min_points=50, print_progress=True))

    print(cluster_idxs)

    possible_values = list(set(cluster_idxs))
    possible_values.remove(-1)

    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points

    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))

    cloud_building = point_cloud_downsampled.select_by_index(largest_idxs)
    cloud_others = point_cloud_downsampled.select_by_index(largest_idxs, invert=True)

    cloud_others.paint_uniform_color([0, 0, 1.0])

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    # Create a list of entities to draw
    entities = []
    entities.append(point_cloud_downsampled)

    o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])

    o3d.io.write_point_cloud('factory_isolated.ply', cloud_building, write_ascii=False, compressed=False, print_progress=False)


if __name__ == "__main__":
    main()
