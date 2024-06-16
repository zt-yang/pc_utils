from os.path import abspath

from pcd_utils import *

pcd_path = abspath('data/point_cloud.ply')


def create_pcd(points, indices: list = None, show: bool = True, use_ransac: bool = True):

    """ build scene point cloud """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.points)
    pcd.colors = o3d.utility.Vector3dVector(points.colors)  ## color is float64 array of shape (n, 3), range [0, 1]
    ## pcd.colors = o3d.utility.Vector3dVector(feats)
    visualize_pcd_in_world(pcd, title="\nOriginal Point Cloud")

    """ filter bad training points """
    all_points = np.asarray(pcd.points)
    min_bound, max_bound = get_scene_bounds(pcd, source_name='chairs_full')
    ind_points = get_indices_in_bound(all_points, min_bound, max_bound)
    pcd_subset = pcd.select_by_index(ind_points)
    visualize_pcd_in_world(pcd_subset, title="\nCropped Point Cloud")

    """ adjust floor using subset of pcd """
    if use_ransac:
        pcd, pcd_subset, R1 = adjust_floor_by_ransac(pcd, pcd_subset, visualize=False)
        pcd, pcd_subset, R2 = adjust_floor_by_ransac(pcd, pcd_subset, visualize=False)
        # rotation = rot2euler(R1) ## R2 @ R1
        # print('-' * 50+f'\n\nrotation: {rotation} \n\n'+'-' * 50)

    """ crop in a bounding box """
    obb = get_scene_bounding_box(pcd_subset, verbose=True)
    ind_points = obb.get_point_indices_within_bounding_box(pcd.points)

    """ remove outliers """
    cl, ind_floating = pcd.remove_radius_outlier(nb_points=60, radius=0.02)
    # display_inlier_outlier(pcd, ind_floating)

    ind = list(set(ind_points) - set(ind_floating))
    print(f'original: {all_points.shape[0]} x in range: {len(ind_points)}  '
          f'- floating: {len(ind_floating)} = remaining: {len(ind)}')

    ## filter specific objects
    if indices is not None:
        ind = list(set(ind).intersection(set(indices)))

    if show:
        print("\nProcessed Point Cloud")
        pc = pcd.select_by_index(ind)
        background = pcd.select_by_index(ind, invert=True)
        visualize_final_pcds([pc], obb, crop=True, background=background)


if __name__ == '__main__':
    points = o3d.io.read_point_cloud(pcd_path)

    original_points = np.asarray(points.points)

    create_pcd(points, indices=None, show=True, use_ransac=True)
