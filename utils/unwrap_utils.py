#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按顶点标签分割子网格并分别做 LSCM UV 展开，
使用原始网格顶点顺序以确保 labels 长度一致。
结合了主要连通分量提取功能以确保UV展开成功。
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import trimesh
import igl

def load_full_mesh(path: Path):
    """加载完整网格，不拆分，直接返回 V (#V×3), F (#F×3)."""
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh.merge_vertices()  # 如果你之前也做了 merge，保留这里
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)

def extract_submesh(V, F, labels, target_label):
    """
    从全网格中提取属于 target_label 的子网格，
    返回 V_sub, F_sub, 以及原顶点索引 used_vids。
    """
    # 只选那些三个顶点都属于同一 label 的面
    mask = np.all(labels[F] == target_label, axis=1)
    F_sel = F[mask]
    if len(F_sel) == 0:
        return None, None, None
    used_vids = np.unique(F_sel)
    old2new = {old: new for new, old in enumerate(used_vids)}
    V_sub = V[used_vids]
    F_sub = np.array([[old2new[v] for v in face] for face in F_sel], dtype=np.int32)
    return V_sub, F_sub, used_vids

def get_main_component(V, F):
    """
    从网格中提取主要连通分量，确保UV展开能够成功。
    """
    # 创建临时网格
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    
    # 分割为连通分量
    parts = mesh.split(only_watertight=False)
    
    if len(parts) == 1:
        # 如果只有一个分量，直接返回
        main = parts[0]
    else:
        # 选择面数最多的分量作为主要分量
        main = max(parts, key=lambda m: len(m.faces))
    
    # 合并顶点
    main.merge_vertices()
    
    return np.asarray(main.vertices), np.asarray(main.faces, dtype=np.int32)

def parameterise(V: np.ndarray, F: np.ndarray):
    """使用 LSCM 展开 UV，返回 (#V×2) 的 UV 坐标。"""
    boundary = igl.boundary_loop(F)
    if len(boundary) < 2:
        raise RuntimeError("Mesh has no boundary loop; can't unwrap a closed surface.")
    
    B = V[boundary]
    d2 = ((B[:, None] - B[None, :]) ** 2).sum(-1)
    i, j = np.unravel_index(d2.argmax(), d2.shape)
    b = np.array([boundary[i], boundary[j]], dtype=np.int32)
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])
    
    result = igl.lscm(V, F, b, bc)
    return result[0] if isinstance(result, tuple) else result

def visualize_uv(uv_coords: np.ndarray, F: np.ndarray, title: str = None):
    """绘制 UV 网格，返回 matplotlib Figure。"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # 绘制边
    edges = []
    for face in F:
        for k in range(3):
            edges.append([uv_coords[face[k]], uv_coords[face[(k+1)%3]]])
    
    ax.add_collection(LineCollection(edges, colors="black", linewidths=0.5, alpha=0.7))
    ax.scatter(uv_coords[:,0], uv_coords[:,1], c="red", s=1, alpha=0.8)
    ax.set_aspect("equal")
    
    if title:
        ax.set_title(title)
    
    ax.set_xlabel("U")
    ax.set_ylabel("V")
    ax.grid(True, alpha=0.3)
    
    # 设置合适的显示范围
    padding = 0.1
    u_min, u_max = uv_coords[:,0].min(), uv_coords[:,0].max()
    v_min, v_max = uv_coords[:,1].min(), uv_coords[:,1].max()
    ax.set_xlim(u_min - padding, u_max + padding)
    ax.set_ylim(v_min - padding, v_max + padding)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(
        description="按顶点标签分割子网格并分别展开 UV，"
                    "使用原始网格顶点顺序来匹配 labels。"
    )
    parser.add_argument("input", type=Path, help="输入 PLY 文件")
    parser.add_argument("--labels", type=Path, required=True, help="顶点标签 .npy 文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="结果保存目录（默认：./outputs）",
    )
    args = parser.parse_args()

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading full mesh from {args.input} …")
    V, F = load_full_mesh(args.input)
    print(f"Full mesh: {len(V)} vertices, {len(F)} faces")

    # 加载标签
    labels = np.load(args.labels)
    if len(labels) != len(V):
        raise ValueError(f"标签长度 {len(labels)} 与顶点数 {len(V)} 不匹配！")
    print(f"Loaded labels: {len(labels)} entries; distinct labels: {np.unique(labels)}")

    # 处理每个标签
    for lbl in np.unique(labels):
        print(f"\n[Label {lbl}] 开始处理...")
        
        # 提取子网格
        V_sub, F_sub, used_vids = extract_submesh(V, F, labels, lbl)
        if V_sub is None:
            print(f"[Label {lbl}] 无对应面，跳过")
            continue

        print(f"[Label {lbl}] 提取到子网格: {len(V_sub)} 顶点, {len(F_sub)} 面片")

        # 获取主要连通分量
        try:
            V_main, F_main = get_main_component(V_sub, F_sub)
            print(f"[Label {lbl}] 主要连通分量: {len(V_main)} 顶点, {len(F_main)} 面片")
        except Exception as e:
            print(f"[Label {lbl}] 提取主要连通分量失败: {e}")
            continue

        # 保存子网格 PLY
        mesh_sub = trimesh.Trimesh(vertices=V_main, faces=F_main, process=False)
        ply_path = args.output_dir / f"part_{lbl}.ply"
        mesh_sub.export(ply_path)
        print(f"[Label {lbl}] 保存 PLY: {ply_path}")

        # 展开 UV
        try:
            uv = parameterise(V_main, F_main)
            print(f"[Label {lbl}] UV 展开成功")
            
            # 可视化并保存 UV 图
            fig = visualize_uv(uv, F_main, title=f"UV Label {lbl}")
            img_path = args.output_dir / f"uv_label{lbl}.png"
            fig.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"[Label {lbl}] 保存 UV 图: {img_path}")
            
        except Exception as e:
            print(f"[Label {lbl}] UV 展开失败: {e}")
            continue

    print(f"\n所有处理完成！结果保存在 {args.output_dir}")

if __name__ == "__main__":
    main()