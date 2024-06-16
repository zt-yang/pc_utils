import os
from os.path import isfile, join, dirname
import open3d as o3d
import numpy as np


def hex_to_pcd_color(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))

HEX_RED = 'e74c3c'
HEX_ORANGE = 'e67e22'
HEX_YELLOW = 'f1c40f'
HEX_GREEN = '2ecc71'
HEX_BLUE = '3498db'
HEX_PURPLE = '9b59b6'
HEX_MIDNIGHT = '34495e'
RAINBOW = [HEX_RED, HEX_ORANGE, HEX_YELLOW, HEX_GREEN, HEX_BLUE, HEX_PURPLE, HEX_MIDNIGHT]
COLORS = [HEX_RED, HEX_BLUE, HEX_YELLOW, HEX_PURPLE, HEX_ORANGE, HEX_MIDNIGHT, HEX_GREEN]
RAINBOW = [hex_to_pcd_color(color) for color in RAINBOW]
COLORS = [hex_to_pcd_color(color) for color in COLORS]


def similarity_to_colors(numbers, color1=HEX_BLUE, color2=HEX_RED):
    # Convert hex colors to RGB
    rgb1 = tuple(int(color1[i:i + 2], 16) for i in (0, 2, 4))
    rgb2 = tuple(int(color2[i:i + 2], 16) for i in (0, 2, 4))

    # convert np array of numbers in [-1, 1] to colors in [color1, color2] RGB without for loop
    colors = (np.array(rgb1) * (1 - numbers) + np.array(rgb2) * (numbers + 1)) / 2
    return colors.astype(int)


############################################################


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def get_indices_in_bound(all_points, min_bound, max_bound):
    mask_x = (all_points[:, 0] >= min_bound[0]) & (all_points[:, 0] <= max_bound[0])
    mask_y = (all_points[:, 1] >= min_bound[1]) & (all_points[:, 1] <= max_bound[1])
    mask_z = (all_points[:, 2] >= min_bound[2]) & (all_points[:, 2] <= max_bound[2])
    mask = mask_x & mask_y & mask_z
    ind_points = np.where(mask)[0]
    print(f'in range points: {len(ind_points)} / {len(all_points)}')
    return ind_points


def get_scene_bounds(pcd, source_name):
    ## TODO: need a more principled way of getting the bounding box for trained scenes
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()

    if source_name == 'chairs_full':
        min_bound[0] = -max_bound[0]
        min_bound[2] = -max_bound[2]
        max_bound[1] = -min_bound[1]
        min_bound[2] /= 2
        max_bound[2] /= 2
        min_bound[1] /= 3
        max_bound[1] /= 3
        min_bound[0] /= 3
        max_bound[0] /= 3

    if source_name == 'office_3':
        # min_bound[2] = -1.6
        max_bound[2] = -0.2
        min_bound[1] = -7

    return min_bound, max_bound


def get_scene_bounding_box(pcd, cut_z: bool = False, verbose: bool = False):
    min_bound = pcd.get_min_bound()
    max_bound = pcd.get_max_bound()
    if verbose:
        print(f'min_bound: {min_bound}, max_bound: {max_bound}')

    if cut_z:
        max_bound[2] = 0.1
        min_bound[2] = -2.5

    ## crop in a bounding box
    R = np.identity(3)
    center = (min_bound + max_bound) / 2.0
    extent = max_bound - min_bound
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return obb


def adjust_floor_by_ransac(pcd: o3d.geometry.PointCloud, pcd_subset: o3d.geometry.PointCloud = None,
                           distance_threshold: float = 0.01, ransac_n: int = 300, num_iterations: int = 1000,
                           visualize: bool = True, return_R: bool = False):
    """ Detect the floor plane using RANSAC.
    Args:
        pcd: an open3d point cloud.
        distance_threshold: maximum distance a point can have to an estimated plane to be considered an inlier.
        ransac_n: the number of points that are randomly sampled to estimate a plane.
        num_iterations: how often a random plane is sampled and verified.
        visualize: whether to visualize the point cloud.

    Returns:
        Tuple[numpy.ndarray[numpy.float64[4, 1]], List[int]]

    Source:
        https://stackoverflow.com/questions/62596854/aligning-a-point-cloud-with-the-floor-plane-using-open3d
    """

    def vector_angle(u, v):
        return np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))

    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    def get_floor_plane(pc):
        plane_model, inliers = pc.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=ransac_n,
                                                num_iterations=num_iterations)
        return plane_model

    # Get the plane equation of the floor → ax+by+cz+d = 0
    floor = get_floor_plane(pcd_subset) if pcd_subset is not None else get_floor_plane(pcd)
    a, b, c, d = floor

    # Calculate rotation angle between plane normal & z-axis
    plane_normal = tuple(floor[:3])
    z_axis = (0, 0, 1)

    rotation_angle = vector_angle(plane_normal, z_axis)
    # print(f'rotation angle: {rotation_angle}')

    R = rotation_matrix_from_vectors(np.asarray(plane_normal), np.asarray(z_axis))

    if return_R:
        return R

    pcds = [pcd, pcd_subset] if pcd_subset is not None else [pcd]
    for pc in pcds:
        pc.translate((0, -d / c, 0))
        pc.rotate(R, center=(0, 0, 0))

    ## adjust the subset of points
    if pcd_subset is not None:
        obb = get_scene_bounding_box(pcd_subset, cut_z=True)
        pcd_subset = pcd_subset.crop(obb)

    if visualize:
        visualize_pcd_in_world(pcd_subset)
    return pcd, pcd_subset, R


def visualize_pcd_in_world(pcd, title=''):
    if len(title) > 0:
        print(title)
    obb = get_scene_bounding_box(pcd)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd, draw_aabb(obb), world_frame])


def visualize_final_pcds(pcds, obb, crop=True, boxes=[], background=None, point_triad=None,
                         screenshot_path=None, width=3940, height=2160):
    geometries = []

    if background is not None:
        background.paint_uniform_color([0.8, 0.8, 0.8])
        pcds.append(background)

    for pc in pcds:
        if crop:
            pc_cropped = pc.crop(obb)
            # pc_cropped = pc_cropped.voxel_down_sample(voxel_size=0.05)
            geometries.append(pc_cropped)
        else:
            geometries.append(pc)

    geometries.append(draw_aabb(obb))
    geometries.extend([draw_aabb(box) for box in boxes])

    if point_triad is None:
        point_triad = [0, 0, 0]
    point_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=point_triad)
    geometries.append(point_frame)

    if screenshot_path is not None:
        view_point_path = screenshot_path.replace('point_cloud_', 'view_point_').replace('.png', '.json')
        if not isfile(view_point_path):
            save_view_point(geometries, view_point_path, width=width, height=height)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height)

        for geometry in geometries:
            vis.add_geometry(geometry)
        vis.poll_events()

        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(view_point_path)
        ctr.convert_from_pinhole_camera_parameters(param)
        vis.update_renderer()

        vis.capture_screen_image(screenshot_path)
        vis.destroy_window()
        print(f'\t... screenshot saved to: {screenshot_path}')
    else:
        o3d.visualization.draw_geometries(geometries)


def save_view_point(geometries, filename, width=1920, height=1080):
    reference_view_point_path = join(dirname(filename), 'view_point.json')

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)
    for geometry in geometries:
        vis.add_geometry(geometry)

    if isfile(reference_view_point_path):
        ctr = vis.get_view_control()
        param = o3d.io.read_pinhole_camera_parameters(reference_view_point_path)
        ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()  # user changes the view and press "q" to terminate

    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename, param)
    if not isfile(reference_view_point_path):
        o3d.io.write_pinhole_camera_parameters(reference_view_point_path, param)

    vis.destroy_window()


def draw_aabb(obb):
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=obb.get_min_bound(), max_bound=obb.get_max_bound())
    bbox.color = (1, 0, 0)
    return bbox


def rot2euler(R):
    # from scipy.spatial.transform import Rotation
    # r = Rotation.from_matrix(R)
    # return r.as_euler('xyz', degrees=False)
    beta = -np.arcsin(R[2, 0])
    alpha = np.arctan2(R[2, 1]/np.cos(beta), R[2, 2]/np.cos(beta))
    gamma = np.arctan2(R[1, 0]/np.cos(beta), R[0, 0]/np.cos(beta))
    return np.array((alpha, beta, gamma))
