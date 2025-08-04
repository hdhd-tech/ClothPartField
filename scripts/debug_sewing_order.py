import numpy as np
import trimesh
import matplotlib.pyplot as plt
from typedef import ColorPanel
import os
from make_spec_copy import load_data      # 假设翻转代码写在 make_spec.py 的 load_data()

panels, sewing_pairs = load_data()   # 这里的 panels 已经翻转

# 把 panels 和 sewing_pairs 丢给新的分析函数

def analyze_sewing_pairs(panels, sewing_pairs, length_tol=2):
    """
    Analyze sewing pairs for ordering / length / wrap-around issues.

    Parameters
    ----------
    panels : List[ColorPanel]
        内存中的面片对象（已做翻转等修改）。
    sewing_pairs : List[Tuple]
        [(s0, s1), ...] 形式的缝合对。函数 *不会* 再从磁盘读取。
    length_tol : int, default=2
        允许的顶点数差异容忍值。
    """
    print("=== SEWING PAIRS ANALYSIS ===")

    # 建立快速索引：color_id -> panel
    id2panel = {p.color_id: p for p in panels}

    problematic_pairs = []

    for i, pair in enumerate(sewing_pairs):
        s0, s1 = pair
        pid0, start0, end0, rev0 = s0
        pid1, start1, end1, rev1 = s1

        print(f"\n--- Analyzing Pair {i} ---")
        print(f"Panel {pid0}: edges {start0}-{end0} (reverse: {rev0})")
        print(f"Panel {pid1}: edges {start1}-{end1} (reverse: {rev1})")

        # ---------- 1) wrap-around 检测 ----------
        if end0 < start0:
            print(f"⚠️  Panel {pid0} has wrap-around: {start0} -> {end0}")
            problematic_pairs.append((i, pid0, "wrap-around"))

        if end1 < start1:
            print(f"⚠️  Panel {pid1} has wrap-around: {start1} -> {end1}")
            problematic_pairs.append((i, pid1, "wrap-around"))

        # ---------- 2) 计算边段顶点数量 ----------
        bnd_len0 = len(id2panel[pid0].boundary_vertices)
        bnd_len1 = len(id2panel[pid1].boundary_vertices)

        len0 = (end0 - start0) if end0 >= start0 else (bnd_len0 - start0 + end0)
        len1 = (end1 - start1) if end1 >= start1 else (bnd_len1 - start1 + end1)

        print(f"Edge lengths: Panel {pid0}: {len0}, Panel {pid1}: {len1}")

        if abs(len0 - len1) > length_tol:
            print(f"⚠️  Edge length mismatch: {len0} vs {len1}")
            problematic_pairs.append(
                (i, f"panels {pid0}-{pid1}", "length mismatch")
            )

        if i == 7 or (pid0 == 4 or pid1 == 4):
            print(f"\n--- Analyzing Pair {i} ---")
            print(f"Panel {pid0}: edges {start0}-{end0} (reverse: {rev0})")
            print(f"Panel {pid1}: edges {start1}-{end1} (reverse: {rev1})")
    return sewing_pairs, problematic_pairs


def get_boundary_length(panel_id):
    """Get the boundary length for a panel"""
    folder = "../outputs"
    boundary_file = f"{folder}/sewing_pairs/boundary_vertices_{panel_id}.txt"
    if os.path.exists(boundary_file):
        boundary_vertices = np.loadtxt(boundary_file, dtype=int)
        return len(boundary_vertices)
    return 0

def visualize_panel_boundaries(panels):
    """Visualize panel boundaries to understand the ordering"""
    print("\n=== PANEL BOUNDARY VISUALIZATION ===")
    
    folder = "../outputs"
    panel_files = [f"{folder}/mesh_panel/dress_colored_vert_b_panel_{i}.ply" for i in range(8)]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, ply_file in enumerate(panel_files):
        if not os.path.exists(ply_file):
            continue
            
        # Load panel mesh
        mesh = trimesh.load(ply_file, process=False)
        boundary_vertices = np.loadtxt(f"{folder}/sewing_pairs/boundary_vertices_{i}.txt", dtype=int)
        
        # Get boundary coordinates
        boundary_coords = mesh.vertices[boundary_vertices]
        
        # Plot boundary
        ax = axes[i]
        ax.plot(boundary_coords[:, 0], boundary_coords[:, 1], 'b-', linewidth=2, label='Boundary')
        ax.scatter(boundary_coords[0, 0], boundary_coords[0, 1], c='red', s=50, label='Start')
        ax.scatter(boundary_coords[-1, 0], boundary_coords[-1, 1], c='green', s=50, label='End')
        
        # Add vertex indices for first few points
        for j in range(min(10, len(boundary_coords))):
            ax.annotate(f'{j}', (boundary_coords[j, 0], boundary_coords[j, 1]), 
                       fontsize=8, color='red')
        
        ax.set_title(f'Panel {i} Boundary')
        ax.legend()
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig("../outputs/panel_boundaries.png", dpi=150, bbox_inches='tight')
    print("Saved: outputs/panel_boundaries.png")

def fix_sewing_pairs(sewing_pairs, problematic_pairs):
    """Suggest fixes for problematic sewing pairs"""
    print("\n=== SUGGESTED FIXES ===")
    
    fixed_pairs = sewing_pairs.copy()
    
    for issue_idx, panel_info, issue_type in problematic_pairs:
        pair = sewing_pairs[issue_idx]
        s0, s1 = pair
        
        print(f"\nIssue in pair {issue_idx}: {issue_type}")
        print(f"Original: {pair}")
        
        if issue_type == "wrap-around":
            # Fix wrap-around by reversing the edge direction
            if "wrap-around" in str(panel_info) and isinstance(panel_info, int):
                # Find which segment has wrap-around and fix it
                panel0_id, start0, end0, reverse0 = s0
                panel1_id, start1, end1, reverse1 = s1
                
                if panel_info == panel0_id and end0 < start0:
                    # Fix panel 0 wrap-around
                    boundary_len = get_boundary_length(panel0_id)
                    new_start0 = end0
                    new_end0 = start0
                    new_reverse0 = not reverse0
                    s0 = (panel0_id, new_start0, new_end0, new_reverse0)
                    
                elif panel_info == panel1_id and end1 < start1:
                    # Fix panel 1 wrap-around
                    boundary_len = get_boundary_length(panel1_id)
                    new_start1 = end1
                    new_end1 = start1
                    new_reverse1 = not reverse1
                    s1 = (panel1_id, new_start1, new_end1, new_reverse1)
                
                fixed_pair = (s0, s1)
                fixed_pairs[issue_idx] = fixed_pair
                print(f"Fixed:    {fixed_pair}")
    
    return fixed_pairs

def save_fixed_sewing_pairs(fixed_pairs):
    """Save the fixed sewing pairs"""
    folder = "../outputs"
    output_file = f"{folder}/sewing_pairs/dress_colored_vert_b_sewing_pairs_fixed.txt"
    
    with open(output_file, 'w') as f:
        for pair in fixed_pairs:
            f.write(str(pair) + '\n')
    
    print(f"\nFixed sewing pairs saved to: {output_file}")
    print("\nTo use the fixed pairs, modify make_spec.py line 59:")
    print(f'    sewing_file = f"{folder}/sewing_pairs/dress_colored_vert_b_sewing_pairs_fixed.txt"')

def check_edge_directions(panels, sewing_pairs):

    
    for i, pair in enumerate(sewing_pairs):
        s0, s1 = pair
        panel0_id, start0, end0, reverse0 = s0
        panel1_id, start1, end1, reverse1 = s1
        
        print(f"\nPair {i}: Panel {panel0_id} <-> Panel {panel1_id}")
        
        # Load meshes
        mesh0 = panels[panel0_id].mesh
        mesh1 = panels[panel1_id].mesh
        
        boundary0 = panels[panel0_id].boundary_vertices
        boundary1 = panels[panel1_id].boundary_vertices
        
        # Get edge coordinates
        if end0 >= start0:
            edge0_vertices = boundary0[start0:end0+1]
        else:
            edge0_vertices = np.concatenate([boundary0[start0:], boundary0[:end0+1]])
            
        if end1 >= start1:
            edge1_vertices = boundary1[start1:end1+1]
        else:
            edge1_vertices = np.concatenate([boundary1[start1:], boundary1[:end1+1]])
        
        edge0_coords = mesh0.vertices[edge0_vertices]
        edge1_coords = mesh1.vertices[edge1_vertices]
        
        print(f"  Edge 0 length: {len(edge0_coords)}, Edge 1 length: {len(edge1_coords)}")
        
        # Check if edges should be reversed
        if len(edge0_coords) == len(edge1_coords):
            # Compare distances with and without reversal
            dist_normal = np.mean(np.linalg.norm(edge0_coords - edge1_coords, axis=1))
            dist_reversed = np.mean(np.linalg.norm(edge0_coords - edge1_coords[::-1], axis=1))
            
            print(f"  Distance (normal): {dist_normal:.4f}")
            print(f"  Distance (reversed): {dist_reversed:.4f}")
            
            if dist_reversed < dist_normal:
                print(f"  ⚠️  Edge directions might be wrong - consider reversing one edge")

if __name__ == "__main__":
    # Run full analysis
    panels, sewing_pairs = load_data()   
    analyze_sewing_pairs(panels, sewing_pairs)
    visualize_panel_boundaries(panels)
    check_edge_directions(panels, sewing_pairs)
    
    print("\n=== SUMMARY ===")
    print("1. Check outputs/panel_boundaries.png to visualize panel ordering")
    print("2. If wrap-around issues found, use the fixed sewing pairs file")
    print("3. Check edge direction analysis for potential reversals needed") 