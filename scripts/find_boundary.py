import numpy as np
import trimesh

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

    参数:p
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


LABEL_FILE = "outputs/labels/vertex_labels_full.npy"
MESH_FILE = "outputs/colored_meshes/dress_colored_vert.ply"
SAVE_DIR = "outputs/tmp"

os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- 加载数据 ----------
vertex_labels = np.load(LABEL_FILE)
mesh = trimesh.load(MESH_FILE, process=True) # (M, 3)
# 使用方法
mesh, components = analyze_mesh(mesh)
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