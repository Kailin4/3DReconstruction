import numpy as np
import open3d as o3d
import os

# Load data
input_path="./"
output_path="out/"
if not os.path.exists(output_path):
    os.mkdir(output_path)
dataname="sample_w_normals.xyz"
point_cloud= np.loadtxt(dataname,skiprows=1)
print(point_cloud)

def create_mesh(point_cloud):
    # initializing point clouds used to test
    pcd = o3d.geometry.PointCloud() # initialize point cloud
    # print(point_cloud[0,:3], point_cloud[0,3:6]/255, point_cloud[0,:], len(point_cloud[0,:]))
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])


    # Compute meshes using poisson reconstruction
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=5, width=2, scale=1.1, linear_fit=False)[0]
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # Export meshes to visualize in meshlab
    o3d.io.write_triangle_mesh(output_path+"ori.ply", poisson_mesh)
    o3d.io.write_triangle_mesh(output_path+"p_mesh_c.ply", p_mesh_crop)

create_mesh(point_cloud) 
# in reference to https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba#:~:text=5-Step%20Guide%20to%20generate%203D%20meshes%20from%20point,the%20pcd%20point%20cloud.%20...%20More%20items...%20
