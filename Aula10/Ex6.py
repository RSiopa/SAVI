#!/usr/bin/env python3
import math
from copy import deepcopy
from matplotlib import cm
import numpy as np
import open3d as o3d
from more_itertools import locate
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
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

def draw_registration_result(source, target, transformation):
    source_temp = deepcopy(source)
    target_temp = deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                      zoom=0.4459,
                                      front=[0.9288, -0.2951, -0.2242],
                                      lookat=[1.6784, 2.0612, 1.4451],
                                      up=[-0.3402, -0.9189, -0.1996])


def main():
    # -----------------------------------------------------
    # Initialization
    # -----------------------------------------------------

    p = PointCloudProcessing()
    p.loadPointCloud('data/scene.ply')

    # -----------------------------------------------------
    # Execution
    # -----------------------------------------------------

    p.preProcess(voxel_size=0)

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
        d['center'] = d['points'].get_center()
        objects.append(d)

    cereal_box_model = o3d.io.read_point_cloud('data/cereal_box_2_2_40.pcd')

    for object in objects:
        print("Apply point-to-point ICP to object " + str(object['idx']))

        trans_init = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        reg_p2p = o3d.pipelines.registration.registration_icp(cereal_box_model, object['points'], 2, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())

        print(reg_p2p.inlier_rmse)
        object['rmse'] = reg_p2p.inlier_rmse

        # draw_registration_result(cereal_box_model, object['points'], reg_p2p.transformation)

    minimum_rmse = 10e8
    cereal_box_idx = None

    for object_idx, object in enumerate(objects):
        if object['rmse'] < minimum_rmse:
            minimum_rmse = object['rmse']
            cereal_box_idx = object_idx

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

    # Make a more complex open3D window to show object labels on top of 3d

    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Open3D - 3D Text", 1024, 768)
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0, 0, 0, 1])
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 5 * w.scaling

    for entity_idx, entity in enumerate(entities):
        widget3d.scene.add_geometry("Entity" + str(entity_idx), entity, mat)

    for object_idx, object in enumerate(objects):
        label_pos = [object['center'][0], object['center'][1], object['center'][2] + 0.15]

        label_text = object['idx']
        if object_idx ==:


        label = widget3d.add_3d_label(label_pos, object['idx'])
        label.color = gui.Color(object['color'][0], object['color'][1], object['color'][2])
        label.scale = 2

    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()

    # o3d.visualization.draw_geometries(entities, zoom=view['trajectory'][0]['zoom'],
    #                                   front=view['trajectory'][0]['front'], lookat=view['trajectory'][0]['lookat'],
    #                                   up=view['trajectory'][0]['up'])


if __name__ == "__main__":
    main()
