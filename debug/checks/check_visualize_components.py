#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将基于label提取的连通分量保存为带颜色的PLY文件，方便在3D查看器中查看
"""

import argparse
from pathlib import Path
import numpy as np
import trimesh
import igl

def load_full_mesh(path: Path):
    """加载完整网格，不拆分，直接返回 V (#V×3), F (#F×3)."""
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(mesh.dump())
    mesh.merge_vertices()
    return np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.int32)

def extract_submesh(V, F, labels, target_label):
    """
    从全网格中提取属于 target_label 的子网格，
    返回 V_sub, F_sub, 以及原顶点索引 used_vids。
    """
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
    返回主要连通分量的顶点和面，以及这些顶点在输入V中的索引
    """
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)
    parts = mesh.split(only_watertight=False)
    
    if len(parts) == 1:
        main = parts[0]
    else:
        main = max(parts, key=lambda m: len(m.faces))
    
    main.merge_vertices()
    
    # 找到主要分量的顶点在原始V中的索引
    main_vertices = np.asarray(main.vertices)
    main_faces = np.asarray(main.faces, dtype=np.int32)
    
    # 通过最近邻匹配找到对应关系
    from scipy.spatial.distance import cdist
    distances = cdist(main_vertices, V)
    vertex_mapping = np.argmin(distances, axis=1)
    
    return main_vertices, main_faces, vertex_mapping

def generate_distinct_colors(n):
    """生成n个视觉上区分度高的颜色"""
    if n <= 10:
        # 使用tab10调色板
        import matplotlib.pyplot as plt
        colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n]
    else:
        # 使用HSV颜色空间生成更多颜色
        import matplotlib.colors as mcolors
        hues = np.linspace(0, 1, n, endpoint=False)
        colors = []
        for h in hues:
            colors.append(mcolors.hsv_to_rgb([h, 0.8, 0.9]))
        colors = np.array(colors)
    
    # 转换为0-255范围的整数
    return (colors[:, :3] * 255).astype(np.uint8)

def create_combined_mesh_with_colors(V, F, labels, output_dir):
    """
    创建一个包含所有连通分量的彩色网格文件
    """
    unique_labels = np.unique(labels)
    colors = generate_distinct_colors(len(unique_labels))
    
    # 存储所有分量的顶点、面和颜色
    all_vertices = []
    all_faces = []
    all_colors = []
    
    vertex_offset = 0
    component_info = []
    
    print("处理各个连通分量...")
    
    for i, lbl in enumerate(unique_labels):
        print(f"处理标签 {lbl}...")
        
        # 提取子网格
        V_sub, F_sub, used_vids = extract_submesh(V, F, labels, lbl)
        if V_sub is None:
            print(f"标签 {lbl} 无对应面，跳过")
            continue
        
        try:
            # 获取主要连通分量
            V_main, F_main, vertex_mapping = get_main_component(V_sub, F_sub)
            
            # 获取在原始网格中的实际顶点坐标
            original_positions = V[used_vids[vertex_mapping]]
            
            # 添加顶点
            all_vertices.append(original_positions)
            
            # 添加面（需要调整索引）
            faces_adjusted = F_main + vertex_offset
            all_faces.append(faces_adjusted)
            
            # 为所有顶点添加相同的颜色
            vertex_colors = np.tile(colors[i], (len(original_positions), 1))
            all_colors.append(vertex_colors)
            
            # 更新偏移量
            vertex_offset += len(original_positions)
            
            # 记录信息
            center = np.mean(original_positions, axis=0)
            component_info.append({
                'label': lbl,
                'vertices': len(original_positions),
                'faces': len(F_main),
                'center': center,
                'color': colors[i],
                'bbox_min': np.min(original_positions, axis=0),
                'bbox_max': np.max(original_positions, axis=0)
            })
            
            print(f"标签 {lbl}: {len(original_positions)} 顶点, {len(F_main)} 面")
            
        except Exception as e:
            print(f"标签 {lbl} 处理失败: {e}")
            continue
    
    if not all_vertices:
        print("没有成功处理任何连通分量！")
        return
    
    # 合并所有数据
    combined_vertices = np.vstack(all_vertices)
    combined_faces = np.vstack(all_faces)
    combined_colors = np.vstack(all_colors)
    
    print(f"\n合并结果: {len(combined_vertices)} 顶点, {len(combined_faces)} 面")
    
    # 创建trimesh对象
    mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        vertex_colors=combined_colors,
        process=False
    )
    
    # 保存PLY文件
    ply_path = output_dir / "components_colored.ply"
    mesh.export(ply_path)
    print(f"保存彩色网格: {ply_path}")
    
    # 保存组件信息
    info_path = output_dir / "components_info.txt"
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write("连通分量信息统计\n")
        f.write("=" * 50 + "\n")
        f.write(f"总计: {len(component_info)} 个连通分量\n")
        f.write(f"总顶点数: {len(combined_vertices)}\n")
        f.write(f"总面片数: {len(combined_faces)}\n\n")
        
        for info in component_info:
            f.write(f"标签 {info['label']}:\n")
            f.write(f"  顶点数: {info['vertices']}\n")
            f.write(f"  面片数: {info['faces']}\n")
            f.write(f"  颜色 (RGB): {info['color']}\n")
            f.write(f"  中心点: ({info['center'][0]:.3f}, {info['center'][1]:.3f}, {info['center'][2]:.3f})\n")
            f.write(f"  包围盒: [{info['bbox_min'][0]:.3f}, {info['bbox_min'][1]:.3f}, {info['bbox_min'][2]:.3f}] - ")
            f.write(f"[{info['bbox_max'][0]:.3f}, {info['bbox_max'][1]:.3f}, {info['bbox_max'][2]:.3f}]\n")
            f.write("\n")
    
    print(f"保存组件信息: {info_path}")
    
    return component_info

def create_individual_colored_meshes(V, F, labels, output_dir):
    """
    为每个连通分量创建单独的彩色PLY文件
    """
    unique_labels = np.unique(labels)
    colors = generate_distinct_colors(len(unique_labels))
    
    individual_dir = output_dir / "individual_components"
    individual_dir.mkdir(exist_ok=True)
    
    print("创建单独的连通分量文件...")
    
    for i, lbl in enumerate(unique_labels):
        print(f"处理标签 {lbl}...")
        
        # 提取子网格
        V_sub, F_sub, used_vids = extract_submesh(V, F, labels, lbl)
        if V_sub is None:
            continue
        
        try:
            # 获取主要连通分量
            V_main, F_main, vertex_mapping = get_main_component(V_sub, F_sub)
            
            # 获取在原始网格中的实际顶点坐标
            original_positions = V[used_vids[vertex_mapping]]
            
            # 为所有顶点添加相同的颜色
            vertex_colors = np.tile(colors[i], (len(original_positions), 1))
            
            # 创建网格
            mesh = trimesh.Trimesh(
                vertices=original_positions,
                faces=F_main,
                vertex_colors=vertex_colors,
                process=False
            )
            
            # 保存单独的PLY文件
            ply_path = individual_dir / f"component_label_{lbl}.ply"
            mesh.export(ply_path)
            print(f"保存: {ply_path}")
            
        except Exception as e:
            print(f"标签 {lbl} 处理失败: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(
        description="将基于label提取的连通分量保存为带颜色的PLY文件"
    )
    parser.add_argument("input", type=Path, help="输入 PLY 文件")
    parser.add_argument("--labels", type=Path, required=True, help="顶点标签 .npy 文件")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("colored_components"),
        help="结果保存目录（默认：./colored_components）",
    )
    parser.add_argument(
        "--individual",
        action="store_true",
        help="同时创建每个连通分量的单独PLY文件"
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

    # 创建合并的彩色网格
    print("\n创建合并的彩色网格...")
    component_info = create_combined_mesh_with_colors(V, F, labels, args.output_dir)
    
    # 如果指定了individual参数，创建单独的文件
    if args.individual:
        print("\n创建单独的连通分量文件...")
        create_individual_colored_meshes(V, F, labels, args.output_dir)
    
    print(f"\n处理完成！结果保存在 {args.output_dir}")
    print(f"主要输出文件: {args.output_dir / 'components_colored.ply'}")
    print("你可以用任何支持PLY格式的3D查看器打开这个文件，比如:")
    print("- MeshLab")
    print("- Blender") 
    print("- CloudCompare")
    print("- 或者在线查看器")

if __name__ == "__main__":
    main()