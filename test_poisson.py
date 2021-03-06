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
point_cloud=point_cloud[0:-1:100,:]
# np.savetxt('sample_500.xyz', point_cloud)


# dataname="sample_500.xyz"
# point_cloud= np.loadtxt(dataname,skiprows=1)

# dataname="pc.npy"
# point_cloud= np.load(dataname)
print(point_cloud, len(point_cloud))

# poisson creating mesh
def create_mesh(point_cloud):
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
    o3d.io.write_triangle_mesh(output_path+"ori_100_poi_1218.ply", poisson_mesh)
    o3d.io.write_triangle_mesh(output_path+"p_mesh_c_100_poi_1218.ply", p_mesh_crop)

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

def get_edge(coord_array, not_visted, vertice_face_match):
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

    tmp_edge = [(tuple(coord_array[min_idx]), tuple(coord_array[sec_min_idx]))]
    # print(tmp_edge)
    
    remove_from_set(not_visted, tmp_edge[0][0])
    remove_from_set(not_visted, tmp_edge[0][1])
    remove_from_set(not_visted, rm_pt)

    return tmp_edge, not_visted

def remove_from_set(pt_set, pt):
    if pt in pt_set:
        pt_set.remove(pt)

def find_Vertice(visited_edge, edge, coord_array, not_visted, tmp_edge, faces_3d, vertices_face_match):
    if len(tmp_edge)<=1:
        tmp_edge=[]
    else:
        tmp_edge=tmp_edge[1::]
    
    # check if the edge has been examined or not
    if edge in visited_edge:
        return visited_edge, tmp_edge, faces_3d
    # find new vertice for next triangle

    visited_edge.add(edge)
    u,v = edge
    # print('edge,', u.v)
    sum_arr = np.sum((coord_array- u)[:,0:3]*(coord_array- u)[:,0:3], axis=1) + np.sum((coord_array-v)[:,0:3]*(coord_array-v)[:,0:3], axis=1) 
    min_idx = np.argmin(sum_arr)
    min_coord = vertices_face_match[min_idx]
    # print(u,v,min_coord)
    # print(min_idx, min_coord)
    if min_coord == u:
        sum_arr[min_idx]=1000000000000000
        min_idx = np.argmin(sum_arr)
        min_coord = vertices_face_match[min_idx]

    if min_coord == v:
        sum_arr[min_idx]=1000000000000000
        min_idx = np.argmin(sum_arr)

    new_vertice = tuple(coord_array[min_idx])
    # print(123, min_idx, vertices_face_match.index(u), vertices_face_match.index(u))
    faces_3d.append([min_idx, vertices_face_match.index(v), vertices_face_match.index(u)])
    # faces_3d.append([3, min_idx, vertices_face_match.index(u), vertices_face_match.index(u)])

    # delete vertice if it has not been visited 
    remove_from_set(not_visted, new_vertice)

    # update the queue, to include new edges 
    tmp_edge += [(u,new_vertice), (v,new_vertice)]
    return visited_edge, tmp_edge, faces_3d


def get_trg(point_cloud):
    # all points' coordinate help to keep track of while loop condition
    not_visted=set(tuple(map(tuple, point_cloud[:,0:3])))
    vertices_3d = point_cloud[:,0:3]
    vertices_face_match = list(map(tuple, point_cloud[:,0:3]))

    # coordinate array of the point cloud
    coord_array = point_cloud[:,0:3]
    color_array = point_cloud[:,3:6]

    # initialization  
    visited_edge = set()
    faces_3d = []


    while len(not_visted)>0:
        tmp_edge, not_visted = get_edge(coord_array, not_visted, vertices_face_match)
        while len(tmp_edge)>0:
            edge = tmp_edge[0]
            visited_edge, tmp_edge, faces_3d=find_Vertice(visited_edge, edge, coord_array, not_visted, tmp_edge, faces_3d, vertices_face_match)
    edges_3d = list(visited_edge)
    return vertices_3d, faces_3d, edges_3d



def get_start_edge(coord_array, not_visted, vertices_face_match):
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

    tmp_edge = [(tuple(coord_array[min_idx]), tuple(coord_array[sec_min_idx]))]
    # print(tmp_edge)
    
    remove_from_set(not_visted, tmp_edge[0][0])
    remove_from_set(not_visted, tmp_edge[0][1])
    remove_from_set(not_visted, rm_pt)

    return tmp_edge, not_visted

def get_edge_1(coord_array, not_visted, vertices_face_match, front, visited_vertex):
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

def find_Vertice_1(visited_edge, edge, coord_array, not_visted, front, faces_3d, vertices_face_match, seed_trig, faces_3d_dict, vertice_dict):
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
    # print(123, min_idx, vertices_face_match.index(u), vertices_face_match.index(u))

    

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

def is_valid():

    pass


def get_trg_1(point_cloud):
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
    front, not_visted = get_start_edge(coord_array, not_visted, vertices_face_match)

    seed_trig=True

    while len(not_visted)>0:
        # tmp_edge, not_visted = get_edge_1(coord_array, not_visted, vertices_face_match)
        front, not_visted=get_edge_1(coord_array, not_visted, vertices_face_match, front, visited_vertex)
        seed_trig=True
        # print(len(not_visted))
        while len(front)>0:
            edge = front[0]
            print(edge)
            visited_edge, front, faces_3d, vertice_dict=find_Vertice_1(visited_edge, edge, coord_array, not_visted, front, faces_3d, vertices_face_match, seed_trig, faces_3d_dict, vertice_dict)
            seed_trig=False
            # print(len(front))
    edges_3d = list(visited_edge)
    return color_array, vertices_3d, faces_3d, edges_3d


def create_ply(color_array, vertices_3d, faces_3d, edges_3d):
    # import PyMesh_0.python.PyMesh as pymesh
    faces_3d=np.asarray(faces_3d)
    # pymesh.meshio.save_mesh_raw('out_mesh.ply', vertices_3d, faces_3d)
    print(vertices_3d[faces_3d[0]], vertices_3d[faces_3d[2]], vertices_3d[faces_3d[1]])
    o3d.geometry.TriangleMesh.create_coordinate_frame()

    vertices_3d=o3d.utility.Vector3dVector(vertices_3d)
    color_array=o3d.utility.Vector3dVector(color_array)
    faces_3d=o3d.utility.Vector3iVector(faces_3d)
    mesh=o3d.geometry.TriangleMesh(vertices_3d,faces_3d)
    mesh.vertex_colors=color_array
    # mesh.paint_uniform_color(np.array([0.0,0.1,0.2]))
    print(mesh, mesh.get_surface_area())
    o3d.io.write_triangle_mesh(output_path+"ori_mesh_our_1218.ply", mesh)
    return True


import time
start = time.time()
# create_mesh(point_cloud) # in reference to https://towardsdatascience.com/5-step-guide-to-generate-3d-meshes-from-point-clouds-with-python-36bad397d8ba#:~:text=5-Step%20Guide%20to%20generate%203D%20meshes%20from%20point,the%20pcd%20point%20cloud.%20...%20More%20items...%20
poi = time.time()-start
start = time.time()

# vertices_3d, faces_3d, edges_3d = get_trg(point_cloud)
# color_array,vertices_3d, faces_3d, edges_3d = get_trg_1(point_cloud)
# tri = time.time()-start
# create_ply(color_array, vertices_3d, faces_3d, edges_3d)
# print(poi, tri)

bpa(point_cloud)

