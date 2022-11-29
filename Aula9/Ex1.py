#!/usr/bin/env python3

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


def main():

    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    # ply_point_cloud = o3d.data.PLYPointCloud()
    point_cloud = o3d.io.read_point_cloud('Factory/factory.ply')
    o3d.visualization.draw_geometries([point_cloud], zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    # -----------------------------------------------------
    # Termination
    # -----------------------------------------------------


if __name__ == "__main__":
    main()
