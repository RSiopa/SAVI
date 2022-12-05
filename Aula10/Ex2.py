#!/usr/bin/env python3
import math
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
                "boundingbox_max": [2.611335277557373, 1.2635015249252319, 3.83980393409729],
                "boundingbox_min": [-2.5246021747589111, -1.5300980806350708, -1.4928504228591919],
                "field_of_view": 60.0,
                "front": [0.81445302111169859, -0.57953694477437501, -0.028340889957977806],
                "lookat": [0.043366551399230957, -0.13329827785491943, 1.1734767556190491],
                "up": [-0.54472968813936007, -0.74688752288557203, -0.38135101287062273],
                "zoom": 0.69999999999999996
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    point_cloud_original = o3d.io.read_point_cloud('data/scene.ply')
    down_point_cloud = point_cloud_original.voxel_down_sample(voxel_size=0.02)

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    down_point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    down_point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 0.]))


    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    entities = [down_point_cloud]

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., -0.13, 1.3]))
    R = frame.get_rotation_matrix_from_xyz((np.pi/1.6, 0, -np.pi/1.25))
    frame.rotate(R)
    entities.append(frame)

    o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])


if __name__ == "__main__":
    main()
