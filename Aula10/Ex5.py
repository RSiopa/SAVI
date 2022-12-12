#!/usr/bin/env python3
import math
from copy import deepcopy
from matplotlib import cm
import numpy as np
import open3d as o3d
from more_itertools import locate

from Aula10.point_cloud_processing import PointCloudProcessing

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [0.5, 0.53416627014546436, 0.5],
                "boundingbox_min": [-0.61673348350192037, -0.54458965365665124, -0.29940674528176447],
                "field_of_view": 60.0,
                "front": [0.70305531581859215, 0.46472232998964957, 0.53828094793352199],
                "lookat": [0.2463415273490199, 0.13213349546420794, 0.18646573490019594],
                "up": [-0.45615466607246913, -0.28598699046111098, 0.84269470266954372],
                "zoom": 0.67999999999999994
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    p = PointCloudProcessing()
    p.loadPointCloud('data/scene.ply')

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    p.preProcess(voxel_size=0.01)

    p.transform(-108, 0, 0, 0, 0, 0)
    p.transform(0, 0, -37, 0, 0, 0)
    p.transform(0, 0, 0, -0.85, -1.10, 0.35)

    p.crop(-0.9, -0.9, -0.3, 0.9, 0.9, 0.4)

    outliers = p.findPlane()

    # Clustering
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.05, min_points=60, print_progress=True))

    print(cluster_idxs)

    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1)

    number_of_objects = len(object_idxs)
    colormap = cm.Pastel1(list(range(0, number_of_objects)))

    objects = []
    for object_idx in object_idxs:

        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))

        object_points = outliers.select_by_index(object_point_idxs)

        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points
        d['color'] = colormap[object_idx, 0:3]
        d['points'].paint_uniform_color(d['color'])
        objects.append(d)

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    p.inliers.paint_uniform_color([0, 1, 1])
    entities = [p.pcd]

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    for object in objects:
        entities.append(object['points'])

    o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])


if __name__ == "__main__":
    main()
