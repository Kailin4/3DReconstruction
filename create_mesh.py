import numpy as np
import open3d as o3d
import os

# poisson creating mesh
def pr(point_cloud):
    # initializing point clouds used to test
    pcd = o3d.geometry.PointCloud() # initialize point cloud
    
    # print(point_cloud[0,:3], point_cloud[0,3:6]/255, point_cloud[0,:], len(point_cloud[0,:]))
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])


    # Compute meshes using poisson reconstruction
    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
    # print(poisson_mesh, type(poisson_mesh))
    bbox = pcd.get_axis_aligned_bounding_box()
    p_mesh_crop = poisson_mesh.crop(bbox)

    # Export meshes to visualize in meshlab
    o3d.io.write_triangle_mesh(output_path+"ori_100_pr.ply", poisson_mesh)

# ball pivoting algorithm
def bpa(point_cloud):
    # initializing point clouds used to test
    pcd = o3d.geometry.PointCloud() # initialize point cloud
    
    # print(point_cloud[0,:3], point_cloud[0,3:6]/255, point_cloud[0,:], len(point_cloud[0,:]))
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:,3:6]/255)
    pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])

    # compute radius
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 3 * avg_dist


    # Compute meshes 
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd,o3d.utility.DoubleVector([radius, radius * 2]))
    # dec_mesh = mesh.simplify_quadric_decimation(100000)
    # Export meshes to visualize in meshlab
    print(bpa_mesh, bpa_mesh.get_surface_area())
    o3d.io.write_triangle_mesh(output_path+"ori_100_bpa_1218.ply", bpa_mesh)

def remove_from_set(pt_set, pt):
    if pt in pt_set:
        pt_set.remove(pt)


def get_edge(coord_array, not_visted, vertices_face_match, visited_vertex):
    rm_pt = not_visted.pop()
    calc_arry = coord_array- rm_pt
    calc_arry= np.sum(calc_arry[:,0:3]*calc_arry[:,0:3], axis=1)

    my_idx = np.argmin(calc_arry)
    calc_arry[my_idx]=10000000000000000000000000000000

    min_idx = np.argmin(calc_arry)
    # print(calc_arry[min_idx])
    calc_arry[min_idx]=100000000000000000000000000
    # print(calc_arry[min_idx])
    sec_min_idx = np.argmin(calc_arry)
    # print(calc_arry,my_idx, min_idx, sec_min_idx, calc_arry[sec_min_idx])
    while sec_min_idx == min_idx:
        calc_arry[sec_min_idx]=100000000000000000000000000
        sec_min_idx = np.argmin(calc_arry)
    # print(calc_arry[min_idx])
    sec_min_idx = np.argmin(calc_arry)
    # print(min_idx, sec_min_idx)

    front = [(tuple(coord_array[min_idx]), tuple(coord_array[sec_min_idx]))]
    # print(tmp_edge)
    
    remove_from_set(not_visted, front[0][0])
    remove_from_set(not_visted, front[0][1])
    remove_from_set(not_visted, rm_pt)

    return front, not_visted

def find_Vertice(visited_edge, edge, coord_array, not_visted, front, faces_3d, vertices_face_match, seed_trig, faces_3d_dict, vertice_dict):
    # check if the edge has been examined or not

    # find new vertice for next triangle
    u,v = edge[0:2]
    # if (u,v) in vertice_dict:
    #     # print((u,v), len(vertice_dict[(u, v)]), vertice_dict[(u, v)])
    #     if len(vertice_dict[(u, v)])>=2:
    #         front.remove(edge)
    #         return visited_edge, front, faces_3d, vertice_dict
    # else:
    #     vertice_dict[(u, v)] = []

    if seed_trig == True:
        u,v= edge
    
        # print('edge,', u.v)
        sum_arr = np.sum((coord_array- u)[:,0:3]*(coord_array- u)[:,0:3], axis=1) + np.sum((coord_array-v)[:,0:3]*(coord_array-v)[:,0:3], axis=1) 
        min_idx = np.argmin(sum_arr)
        min_coord = vertices_face_match[min_idx]

        while min_coord == u or min_coord == v:
            sum_arr[min_idx]=1000000000000000
            min_idx = np.argmin(sum_arr)
            min_coord = vertices_face_match[min_idx]

    else:
        u,v, opo_coord = edge
        
        # print('edge,', u.v)
        sum_arr = np.sum((coord_array- u)[:,0:3]*(coord_array- u)[:,0:3], axis=1) + np.sum((coord_array-v)[:,0:3]*(coord_array-v)[:,0:3], axis=1) 
        min_idx = np.argmin(sum_arr)
        min_coord = vertices_face_match[min_idx]

        while min_coord == u or min_coord == v or min_coord == opo_coord:
            sum_arr[min_idx]=1000000000000000
            min_idx = np.argmin(sum_arr)
            min_coord = vertices_face_match[min_idx]
        
    # if len(vertice_dict[(u, v)]) ==1 and min_coord==vertice_dict[(u, v)][0]:
    #     sum_arr[min_idx]=1000000000000000
    #     min_idx = np.argmin(sum_arr)
    #     min_coord = vertices_face_match[min_idx]

    new_vertice = tuple(coord_array[min_idx])

    if (min_idx, vertices_face_match.index(v), vertices_face_match.index(u)) in faces_3d_dict:
        front.remove(edge)
        return visited_edge, front, faces_3d, vertice_dict
    # vertice_dict[(u, v)].append(new_vertice)
    # vertice_dict[(v, u)] = vertice_dict[(u, v)]

    faces_3d.append([min_idx, vertices_face_match.index(v), vertices_face_match.index(u)])
    faces_3d_dict.add((min_idx, vertices_face_match.index(v), vertices_face_match.index(u)))
    # faces_3d.append([3, min_idx, vertices_face_match.index(u), vertices_face_match.index(u)])

    # delete vertice if it has not been visited 
    remove_from_set(not_visted, new_vertice)
    
    # update the queue, to include new edges 
    if seed_trig:
        front.remove((u,v))
        front += [(u,new_vertice,v), (v,new_vertice, u)]
    else:
        front.remove((u,v,opo_coord)) 
        front += [(u,new_vertice, v), (v,new_vertice, u)]

    # if (u,new_vertice) not in vertice_dict:
    #     vertice_dict[(u,new_vertice)]=[]
    # if (v,new_vertice) not in vertice_dict:
    #     vertice_dict[(v,new_vertice)]=[]
    # vertice_dict[(u,new_vertice)].append(v)
    # vertice_dict[(v,new_vertice)].append(u)
    
    return visited_edge, front, faces_3d, vertice_dict


def get_trg(point_cloud):
    # all points' coordinate help to keep track of while loop condition
    not_visted=set(tuple(map(tuple, point_cloud[:,0:3])))
    vertices_3d = point_cloud[:,0:3]
    vertices_face_match = list(map(tuple, point_cloud[:,0:3]))

    # coordinate array of the point cloud
    coord_array = point_cloud[:,0:3]
    color_array = point_cloud[:,3:6]/255

    # initialization  
    visited_edge = set()
    visited_vertex=set()
    
    faces_3d = []
    faces_3d_dict=set()
    vertice_dict={}
    front, not_visted = get_edge(coord_array, not_visted, vertices_face_match, visited_vertex)

    seed_trig=True

    while len(not_visted)>0:
        front, not_visted=get_edge(coord_array, not_visted, vertices_face_match, visited_vertex)
        seed_trig=True
        while len(front)>0:
            edge = front[0]
            # print(edge)
            visited_edge, front, faces_3d, vertice_dict=find_Vertice(visited_edge, edge, coord_array, not_visted, front, faces_3d, vertices_face_match, seed_trig, faces_3d_dict, vertice_dict)
            seed_trig=False
    edges_3d = list(visited_edge)
    return color_array, vertices_3d, faces_3d, edges_3d


# write the mesh into 'ply' file to be able to display 
def create_ply(color_array, vertices_3d, faces_3d, edges_3d):
    faces_3d=np.asarray(faces_3d)
    o3d.geometry.TriangleMesh.create_coordinate_frame()

    vertices_3d=o3d.utility.Vector3dVector(vertices_3d)
    color_array=o3d.utility.Vector3dVector(color_array)
    faces_3d=o3d.utility.Vector3iVector(faces_3d)
    mesh=o3d.geometry.TriangleMesh(vertices_3d,faces_3d)
    mesh.vertex_colors=color_array
    # mesh.paint_uniform_color(np.array([0.0,0.1,0.2]))
    print(mesh, 'surface area=', mesh.get_surface_area())
    o3d.io.write_triangle_mesh(output_path+"ori_mesh_our_1218.ply", mesh)
    return True

def run_our_alg(point_cloud):
    color_array,vertices_3d, faces_3d, edges_3d = get_trg(point_cloud)
    create_ply(color_array, vertices_3d, faces_3d, edges_3d)

def run_algo(run, point_cloud):
    print(run)
    for i in run:
        if i == 'ark':
            start = time.time()
            run_our_alg(point_cloud)
            tri = time.time()-start
            print(i, tri)
        elif i == 'bpa':
            start = time.time()
            bpa(point_cloud)
            tri = time.time()-start
            print(i, tri)
        elif i == 'pr':
            start = time.time()
            pr(point_cloud)
            tri = time.time()-start
            print(i, tri)
        else:
            print('not implemented')

import time
# Load data
input_path="./"
output_path="out/"
if not os.path.exists(output_path):
    os.mkdir(output_path)

# input data filename this is the sample point cloud we obtained from online
# dataname="sample_w_normals.xyz" 
# point_cloud= np.loadtxt(dataname,skiprows=1)
# point_cloud=point_cloud[0:-1:100,:]
# np.savetxt('sample_500.xyz', point_cloud)


#our generated sample cloud
dataname="pc.npy" 
point_cloud= np.load(dataname)
print(point_cloud, len(point_cloud))

run = ['ark', 'bpa', 'pr'] # include the algorithm you want to run in the list 
run_algo(run, point_cloud)


