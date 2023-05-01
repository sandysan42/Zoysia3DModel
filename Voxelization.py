import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage

def voxelize(pcd,voxel_size = 0.003,visualize=False):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxel_size)
    if visualize == True:
        o3d.visualization.draw_geometries([voxel_grid])

    return voxel_grid, len(voxel_grid.get_voxels())

def voxelize_fill(pcd,voxel_size = 0.003,visualize=False):
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,
                                                            voxel_size=voxel_size)
    voxel_list = voxel_grid.get_voxels()
    o3d_vol = len(voxel_grid.get_voxels())
    model_vol = 0
    filled_vol = 0

    max_level = 0

    for j in range(len(voxel_list)):
        a = voxel_list[j].grid_index[2]
        if a > max_level:
            max_level = a

    for i in range(max_level+1):
        level_list = []
        for j in range(len(voxel_list)):
            if voxel_list[j].grid_index[2] == i:
                level_list.append(voxel_list[j])
        if len(level_list) != 0:
            level_18 = level_list
            test = []
            for k in range(len(level_18)):
                val = list(level_18[k].grid_index[:2])
                test.append(val)
            test = np.asarray(test)

            matrix = np.zeros((max(test[:,0]+1),max(test[:,1]+1)),dtype=int)
            matrix[test[:,0],test[:,1]] = 1

            #Fill matrix using scipy ndimage library
            filled_matrix = ndimage.binary_fill_holes(matrix).astype(int)

            if visualize == True:
                fig = plt.figure()
                ax1 = fig.add_subplot(121)
                ax1.imshow(matrix)
                ax2 = fig.add_subplot(122)
                ax2.imshow(filled_matrix)
                plt.show()

            count = np.count_nonzero(matrix)
            model_vol += count

            filled_count = np.count_nonzero(filled_matrix)
            filled_vol += filled_count

    return o3d_vol,model_vol,filled_vol