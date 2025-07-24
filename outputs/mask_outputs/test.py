import numpy as np
import trimesh
import matplotlib.pyplot as plt

# === 1. 读取数据 ===
mesh = trimesh.load("../dress.obj", process=False)
vertices = mesh.vertices
faces = mesh.faces

vertex_labels = np.load("vertex_labels.npy")
assert vertex_labels.shape[0] == vertices.shape[0], "顶点数不匹配！"

# === 2. 给每个 label 分配一种颜色 ===
label_ids = np.unique(vertex_labels)
label_ids = label_ids[label_ids != -1]  # 不包含 -1
num_labels = len(label_ids)

# 给每个 label 分配一个颜色 (循环颜色表)
cmap = plt.get_cmap("tab20")
label_to_color = {
    label: (np.array(cmap(i % 20)[:3]) * 255).astype(np.uint8)
    for i, label in enumerate(label_ids)
}

# === 3. 构造颜色数组 ===
vertex_colors = np.zeros((vertices.shape[0], 3), dtype=np.uint8)

for i in range(vertices.shape[0]):
    lbl = vertex_labels[i]
    if lbl == -1:
        vertex_colors[i] = [150, 150, 150]  # 灰色
    else:
        vertex_colors[i] = label_to_color.get(lbl, [0, 255, 0])  # 若出错设为绿色

# === 4. 保存为 .ply ===
colored_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
colored_mesh.export("colored_dress.ply")
print("保存成功：colored_dress.ply")
