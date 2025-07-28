import numpy as np
import trimesh
import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev
import os
import os.path as osp
import potpourri3d as pp3d
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components



def extract_connected_seam_subset(vertices, faces, vertex_labels, seam, label):
    """
    提取 seam 中属于 label 的最大连通子集。

    参数:
    - vertices: (N, 3) 顶点数组
    - faces: (M, 3) 面索引
    - vertex_labels: (N,) 每个点的标签
    - seam: List[int]，边界点索引
    - label: int，要处理的 label 编号

    返回:
    - List[int]: seam 中属于最大连通部分的点
    """
    # 取出 label 对应的全部顶点索引
    v_mask = (vertex_labels == label)
    v_indices = np.where(v_mask)[0]

    # 找出完全属于该 label 的三角形
    f_mask = np.all(np.isin(faces, v_indices), axis=1)
    f_sub = faces[f_mask]

    # 构建邻接图
    I, J = [], []
    for tri in f_sub:
        for i in range(3):
            a, b = tri[i], tri[(i + 1) % 3]
            I += [a, b]
            J += [b, a]

    if not I:
        print(f"❌ label {label} 没有合法面")
        return []

    adj = coo_matrix((np.ones(len(I)), (I, J)), shape=(len(vertices), len(vertices)))
    n_comp, labels_cc = connected_components(adj)

    # 找出最大连通块编号
    label_counts = np.bincount(labels_cc[seam])
    largest_cc = np.argmax(label_counts)

    # 返回 seam 中在该连通块上的点
    connected_seam = [v for v in seam if labels_cc[v] == largest_cc]
    return connected_seam


def find_adjacent_label_pairs(vertex_labels, faces):
    adjacent_pairs = set()
    for f in faces:
        vl = vertex_labels[f]
        labels = set(vl)
        if len(labels) == 2:
            a, b = sorted(labels)
            adjacent_pairs.add((a, b))
    return sorted(adjacent_pairs)

def extract_seam_pairs(vertex_labels, faces, target_label_pair):
    """
    提取两个 label 接触处的 stitching seam 对。

    参数:
    - vertex_labels: (N,) 每个顶点的标签
    - faces: (M, 3) 三角形面数组
    返回:
    - seam_A: List[int] 属于 label A 的边界点
    - seam_B: List[int] 属于 label B 的边界点
    """
    label_a, label_b = target_label_pair
    seam_A = set()
    seam_B = set()

    for f in faces:
        vl = vertex_labels[f]  # 这个三角形的3个顶点的label
        labels_in_face = set(vl)

        if label_a in labels_in_face and label_b in labels_in_face:
            # 说明是连接这两个 label 的边界面
            for i in range(3):
                vi, vj = f[i], f[(i + 1) % 3]
                li, lj = vertex_labels[vi], vertex_labels[vj]

                if {li, lj} == {label_a, label_b}:
                    # vi 属于一个 label，vj 属于另一个 label
                    if li == label_a:
                        seam_A.add(vi)
                        seam_B.add(vj)
                    else:
                        seam_A.add(vj)
                        seam_B.add(vi)

    return list(seam_A), list(seam_B)


def get_furthest_cross_pair(vertices, seam_A, seam_B):
    A_pts = vertices[seam_A]
    B_pts = vertices[seam_B]    
    D = cdist(A_pts, B_pts)
    i, j = np.unravel_index(np.argmax(np.triu(D, 1)), D.shape)
    return seam_A[i], seam_B[j]


def find_geodesic_path(verts, faces, start_vert, end_vert):
    path_solver = pp3d.EdgeFlipGeodesicSolver(verts, faces)
    path_pts = path_solver.find_geodesic_path(start_vert, end_vert,max_iterations=1)
    print(f"path_pts shape: {path_pts.shape}")
    return path_pts

def convert_path_coords_to_vertex_indices(vertices, path_coords, threshold=1e-6):
    """
    Convert path coordinates to vertex indices by finding the nearest vertices.
    
    Parameters:
    - vertices: (N, 3) vertex array
    - path_coords: (M, 3) path coordinate array
    - threshold: distance threshold for considering a match
    
    Returns:
    - List[int]: vertex indices that are closest to path coordinates
    """    
    # Calculate distances between all path points and all vertices
    distances = cdist(path_coords, vertices)
    
    # Find the closest vertex for each path point
    closest_vertices = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    
    # Filter out path points that are too far from any vertex
    valid_mask = min_distances < threshold
    valid_vertices = closest_vertices[valid_mask]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_vertices = []
    for v in valid_vertices:
        if v not in seen:
            seen.add(v)
            unique_vertices.append(v)
    
    print(f"Original path points: {len(path_coords)}")
    print(f"Valid matches within threshold: {np.sum(valid_mask)}")
    print(f"Unique vertices: {len(unique_vertices)}")
    
    return unique_vertices

# def find_geodesic_path_2(verts, faces, seam_A):
#     path_solver = pp3d.EdgeFlipGeodesicSolver(verts, faces)
#     path_pts = path_solver.find_geodesic_path_poly(seam_A)
#     print(f"path_pts:{path_pts}")
#     return path_pts


def analyze_mesh(path):
    # 加载 mesh
    mesh = trimesh.load(path, process=True)

    # 是否 watertight（闭合）
    print(f"🧵 is_watertight: {mesh.is_watertight}")

    # 拆分为多个连通部分
    components = mesh.split(only_watertight=False)  # 设置为 True 会忽略非闭合部分

    print(f"🧩 Number of connected components: {len(components)}")

    for i, c in enumerate(components):
        print(f"  Component {i}:")
        print(f"    Faces: {len(c.faces)}")
        print(f"    Vertices: {len(c.vertices)}")
        print(f"    Watertight: {c.is_watertight}")

    return mesh, components



# ---------- 配置 ----------
LABEL_FILE = "../outputs/labels/vertex_labels_full.npy"
MESH_FILE = "../outputs/colored_meshes_outputs/dress_colored_vert.ply"

# ---------- 加载数据 ----------

# 先加载原始网格和标签
mesh_raw = trimesh.load(MESH_FILE, process=False)
vertex_labels_raw = np.load(LABEL_FILE)

# 然后加载处理后的网格
mesh = trimesh.load(MESH_FILE, process=True)

# 找到处理后网格顶点与原始顶点的对应关系
from scipy.spatial.distance import cdist
distances = cdist(mesh.vertices, mesh_raw.vertices)
closest_indices = np.argmin(distances, axis=1)

# 根据对应关系调整标签
vertex_labels = vertex_labels_raw[closest_indices]

print(f"原始顶点数: {len(mesh_raw.vertices)}")
print(f"处理后顶点数: {len(mesh.vertices)}")
print(f"总label顶点数: {vertex_labels.shape[0]}")
print(f"标签范围: {np.min(vertex_labels)} - {np.max(vertex_labels)}")
print(f"唯一标签: {np.unique(vertex_labels)}")
# 使用方法
mesh, components = analyze_mesh(MESH_FILE)
# ---------- 提取每个 label 的子网格 ----------
label_set = np.unique(vertex_labels)
label_submeshes = {}

vertices = mesh.vertices
faces = mesh.faces

adj_pairs = find_adjacent_label_pairs(vertex_labels, faces)

for label_a, label_b in adj_pairs:
    seam_A, seam_B = extract_seam_pairs(vertex_labels, faces, (label_a, label_b))
    print(f"Labels {label_a}-{label_b}: seam_A={len(seam_A)}, seam_B={len(seam_B)}")



seam_A, seam_B = extract_seam_pairs(vertex_labels, faces, (0, 1))
vertices_new = list(set(seam_A).union(seam_B))


v_start, v_end = get_furthest_cross_pair(vertices, seam_A, seam_B)
print(f"v_start:{v_start}, label={vertex_labels[v_start]}, coord={vertices[v_start]}")
print(f"v_end:{v_end}, label={vertex_labels[v_end]}, coord={vertices[v_end]}")


g_p = find_geodesic_path(vertices, faces, v_start, v_end)
g_p = np.array(g_p, dtype=int)



def export_geodesic_obj(vertices, faces, path_idx, output_path="geodesic_path_colored.obj"):
    """
    导出带彩色信息的 .obj 文件（用 vertex color 模拟显示）
    """
    N = len(vertices)
    colors = np.tile([200, 200, 200, 255], (N, 1)).astype(np.uint8)  # 默认灰色 + alpha=255

    colors[path_idx] = [255, 0, 0, 255]  # 红色路径

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=True)
    mesh.export(output_path)
    print(f"✅ geodesic 路径导出为 {output_path}")



# # --- 导出结果 ---
export_geodesic_obj(vertices, faces, g_p, output_path="geodesic_path_colored.ply")

