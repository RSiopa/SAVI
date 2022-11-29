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

    point_cloud = o3d.io.read_point_cloud('Factory/factory.ply')

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    print('Starting plane detection')
    plane_model, inlier_idxs = point_cloud.segment_plane(distance_threshold=0.3, ransac_n=3, num_iterations=100)
    [a, b, c, d] = plane_model
    print()

    inlier_cloud = point_cloud.select_by_index(inlier_idxs)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = point_cloud.select_by_index(inlier_idxs, invert=True)

    # -----------------------------------------------------
    # Visualization
    # -----------------------------------------------------

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], zoom=view['trajectory'][0]['zoom'],
                                      front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])

    o3d.io.write_point_cloud('factory_without_ground.ply', outlier_cloud, write_ascii=False, compressed=False, print_progress=False)


if __name__ == "__main__":
    main()
