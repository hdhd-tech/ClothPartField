import numpy as np
import trimesh
import os
from scipy.spatial.distance import cdist
import networkx as nx
from collections import defaultdict


def find_adjacent_label_pairs(vertex_labels, faces):
    adjacent_pairs = set()
    for f in faces:
        labels = set(vertex_labels[f])
        if len(labels) == 2:
            adjacent_pairs.add(tuple(sorted(labels)))
    return sorted(adjacent_pairs)

def extract_seam_pairs(vertex_labels, faces, target_label_pair):
    label_a, label_b = target_label_pair
    seam_A, seam_B = set(), set()

    for f in faces:
        vl = vertex_labels[f]
        if label_a in vl and label_b in vl:
            for i in range(3):
                vi, vj = f[i], f[(i + 1) % 3]
                li, lj = vertex_labels[vi], vertex_labels[vj]

                if {li, lj} == {label_a, label_b}:
                    if li == label_a:
                        seam_A.add(vi)
                        seam_B.add(vj)
                    else:
                        seam_A.add(vj)
                        seam_B.add(vi)

    return list(seam_A), list(seam_B)

def get_furthest_cross_pair(vertices, seam_A, seam_B):
    A_pts, B_pts = vertices[seam_A], vertices[seam_B]
    D = cdist(A_pts, B_pts)
    i, j = np.unravel_index(np.argmax(D), D.shape)
    return seam_A[i], seam_B[j]

def mesh_to_graph(vertices, faces):
    G = nx.Graph()
    for face in faces:
        for i in range(3):
            u, v = face[i], face[(i + 1) % 3]
            dist = np.linalg.norm(vertices[u] - vertices[v])
            G.add_edge(u, v, weight=dist)
    return G

def shortest_path_on_mesh(G, start, end):
    return nx.shortest_path(G, source=start, target=end, weight='weight')

# method to find the open curve for the sleeves and collar
def find_boundary_vertices(faces):
    #if the edge is only in one face, it is a boundary edge
    edge_count = {}
    for f in faces:
        for i in range(3):
            edge = tuple(sorted((f[i], f[(i + 1) % 3])))
            if edge not in edge_count:
                edge_count[edge] = 0
            edge_count[edge] += 1

    # Boundary edges are those that appear only once
    boundary_edges = {edge for edge, count in edge_count.items() if count == 1}
    return boundary_edges

def get_label_boundary_info(vertex_labels, faces, target_label, min_edge_count=5):
    """
    为指定的label获取边界信息（基于边而不是顶点）
    返回字典，键为相邻的label值（开放边为-1），值为边的列表
    """
    # 收集边界信息
    boundary_info = defaultdict(set)  # 使用set避免重复
    
    # 首先找开放边（boundary edges）
    boundary_edges = find_boundary_vertices(faces)
    for edge in boundary_edges:
        v1, v2 = edge
        l1, l2 = vertex_labels[v1], vertex_labels[v2]
        # 如果这条边的两个顶点都属于目标label，则是开放边
        if l1 == target_label and l2 == target_label:
            boundary_info[-1].add(edge)
    
    # 然后找与其他label相交的边，严格按照extract_seam_pairs的逻辑
    for f in faces:
        vl = vertex_labels[f]  # 这个面的三个顶点的label
        # 只有当这个面包含target_label且包含其他label时才处理
        if target_label in vl:
            # 检查这个面的每条边
            for i in range(3):
                vi, vj = f[i], f[(i + 1) % 3]
                li, lj = vertex_labels[vi], vertex_labels[vj]
                
                # 严格条件：这条边必须恰好连接target_label和另一个不同的label
                if li != lj and {li, lj} == {target_label, lj if li == target_label else li} and (li == target_label or lj == target_label):
                    edge = tuple(sorted([vi, vj]))
                    other_label = lj if li == target_label else li
                    # 确保other_label不是target_label
                    if other_label != target_label:
                        boundary_info[other_label].add(edge)
    
    # 转换为list并过滤数量少的边
    filtered_boundary_info = {}
    for label_key, edges_set in boundary_info.items():
        edges_list = list(edges_set)
        if len(edges_list) >= min_edge_count:
            filtered_boundary_info[label_key] = edges_list
        else:
            print(f"Label {target_label} -> {label_key}: 过滤掉 {len(edges_list)} 条边（少于{min_edge_count}）")
    
    return filtered_boundary_info
    
    # 转换为list并过滤数量少的边
    filtered_boundary_info = {}
    for label_key, edges_set in boundary_info.items():
        edges_list = list(edges_set)
        if len(edges_list) >= min_edge_count:
            filtered_boundary_info[label_key] = edges_list
        else:
            print(f"Label {target_label} -> {label_key}: 过滤掉 {len(edges_list)} 条边（少于{min_edge_count}）")
    
    return filtered_boundary_info

def process_all_labels_boundary(vertex_labels, faces, min_edge_count=5):
    """
    处理所有label的边界信息
    """
    unique_labels = np.unique(vertex_labels)
    all_boundaries = {}
    
    for label in unique_labels:
        print(f"\n处理Label {label}...")
        boundary_info = get_label_boundary_info(vertex_labels, faces, label, min_edge_count)
        all_boundaries[label] = boundary_info
        
        # 创建边界标签列表用于debug显示
        boundary_labels_list = []
        edge_to_neighbor = {}
        
        for neighbor_label, edges in boundary_info.items():
            for edge in edges:
                edge_to_neighbor[edge] = neighbor_label
        
        # 按边的第一个顶点索引排序，创建对应的标签列表
        sorted_edges = sorted(edge_to_neighbor.keys(), key=lambda x: x[0])
        boundary_labels_list = [edge_to_neighbor[edge] for edge in sorted_edges]
        
        print(f"Label {label} 边界信息:")
        for neighbor_label, edges in boundary_info.items():
            if neighbor_label == -1:
                print(f"  开放边: {len(edges)} 条边")
            else:
                print(f"  与Label {neighbor_label}相交: {len(edges)} 条边")
        
        # 打印边界标签列表
        if boundary_labels_list:
            print(f"  边界标签列表 ({len(boundary_labels_list)} 条边): {boundary_labels_list}")
        else:
            print(f"  边界标签列表: []")
    
    return all_boundaries



LABEL_FILE = "../outputs/labels/vertex_labels_full.npy"
MESH_FILE = "../outputs/colored_meshes_outputs/dress_colored_vert.ply"
SAVE_DIR = "outputs/tmp"
os.makedirs(SAVE_DIR, exist_ok=True)

vertex_labels = np.load(LABEL_FILE)
mesh = trimesh.load(MESH_FILE, process=True)
vertices, faces = mesh.vertices, mesh.faces

print("开始处理所有版片的边界信息...")
print(f"总共有 {len(np.unique(vertex_labels))} 个版片")

# 处理所有label的边界信息
all_boundaries = process_all_labels_boundary(vertex_labels, faces, min_edge_count=5)

# 保存每个label的边界信息
for label, boundary_info in all_boundaries.items():
    save_path = os.path.join(SAVE_DIR, f"label_{label}_boundaries.npy")
    np.save(save_path, boundary_info)
    print(f"已保存Label {label}的边界信息到: {save_path}")

# 可选：可视化某个特定label的边界
def visualize_label_boundaries(mesh, vertices, label, boundary_info, vertex_labels):
    """可视化特定label的边界"""
    import matplotlib.pyplot as plt
    
    scene = trimesh.Scene(mesh)
    
    # 使用matplotlib的colormap
    cmap = plt.cm.tab10  #viridis, plasma, Set1等
    
    # 获取所有label的范围来标准化颜色
    all_labels = np.unique(vertex_labels)
    max_label = max(all_labels)
    
    for neighbor_label, boundary_edges in boundary_info.items():
        if len(boundary_edges) > 0:
            # 从边中提取顶点来创建线段
            edge_points = []
            for edge in boundary_edges:
                v1, v2 = edge
                edge_points.extend([vertices[v1], vertices[v2]])
            
            if len(edge_points) > 0:
                edge_points = np.array(edge_points).reshape(-1, 2, 3)
                
                # 创建线段来显示边界边
                for i, line_seg in enumerate(edge_points):
                    line = trimesh.load_path(line_seg)
                    
                    if neighbor_label == -1:
                        # 开放边用特殊颜色（红色）
                        line.colors = [255, 0, 0, 255]
                        name = f"开放边_{i}"
                    else:
                        # 根据label值自动分配颜色，基于实际的label范围
                        color_normalized = neighbor_label / max_label if max_label > 0 else 0
                        rgba = cmap(color_normalized)
                        # 转换为0-255范围的整数
                        color_255 = [int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255), 255]
                        line.colors = color_255
                        name = f"与Label {neighbor_label}相交_{i}"
                    
                    scene.add_geometry(line, node_name=name)
    
    print(f"显示Label {label}的边界信息")
    scene.show()

# 示例：可视化Label 0的边界
# if 0 in all_boundaries:
#     print("\n可视化Label 0的边界...")
#     visualize_label_boundaries(mesh, vertices, 0, all_boundaries[0], vertex_labels)
