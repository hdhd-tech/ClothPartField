import sys
import os
from pathlib import Path

# Add the path that contains the pygarment directory

garmentcode_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../GarmentCode"))
if garmentcode_root not in sys.path:
    sys.path.insert(0, garmentcode_root)


import trimesh
import matplotlib.pyplot as plt
from pygarment.garmentcode.panel import Panel
from pygarment.garmentcode.component import Component
from pygarment.garmentcode.edge import EdgeSequence, CurveEdge
from pygarment.garmentcode.connector import StitchingRule
from pygarment.data_config import Properties
import pygarment as pyg
from scipy.spatial.transform import Rotation as R
from typedef import ColorPanel
import numpy as np
from typing import List, Tuple
import json
from collections import defaultdict
from bezier_fitting import fit_bezier_curve


def load_data():
    folder = "../outputs"
    ply_files = [f"{folder}/mesh_panel/dress_colored_vert_b_panel_{i}.ply" for i in range(15)]
    # read all panels
    panels = []
    for i, ply_file in enumerate(ply_files):
        if not os.path.exists(ply_file):
            print(f"Warning: Panel file {ply_file} does not exist, skipping...")
            continue
        try:
            mesh = trimesh.load(ply_file, process=False)
            boundary_vertices = np.loadtxt(f"{folder}/sewing_pairs/boundary_vertices_{i}.txt", dtype=int)
            boundary_edges = np.loadtxt(f"{folder}/sewing_pairs/boundary_edges_{i}.txt", dtype=int)
            panels.append(
                ColorPanel(
                    color_id=i,
                    boundary_vertices=boundary_vertices,
                    boundary_edges=boundary_edges,
                    mesh=mesh,
                )
            )
            print(f"Successfully loaded panel {i}")
        except Exception as e:
            print(f"Error loading panel {i}: {e}")

    print(f"Total panels loaded: {len(panels)}")
    print(f"Panel color_ids: {[p.color_id for p in panels]}")

    # read sewing pairs
    sewing_pairs = []
    sewing_file = f"{folder}/sewing_pairs/dress_colored_vert_b_sewing_pairs.txt"
    if not os.path.exists(sewing_file):
        print(f"Warning: Sewing pairs file {sewing_file} does not exist")
        return panels, sewing_pairs
        
    for line_num, line in enumerate(open(sewing_file)):
        try:
            # TODO: is eval unsafe?
            pair = eval(line.strip())
            sewing_pairs.append(pair)
            print(f"Sewing pair {line_num}: panel IDs {pair[0][0]} and {pair[1][0]}")
        except Exception as e:
            print(f"Error parsing sewing pair line {line_num}: {e}")
    
    print(f"Total sewing pairs: {len(sewing_pairs)}")
    return panels, sewing_pairs
    

def uv_unfold(panel):
    import igl
    import numpy as np

    # Get mesh data
    V = panel.mesh.vertices
    F = panel.mesh.faces

    print(f"V shape: {V.shape}")
    print(f"F shape: {F.shape}")

    # Find boundary vertices for LSCM constraints
    boundary = igl.boundary_loop(F)
    print(f"boundary shape: {boundary.shape}")

    # Fix two boundary points for LSCM (arbitrary initial placement)
    # Convert to int64 to match faces type
    b = np.array([boundary[0], boundary[len(boundary) // 2]], dtype=np.int64)
    bc = np.array([[0.0, 0.0], [1.0, 0.0]])  # Temporary fixed positions

    print(f"b shape: {b.shape}, dtype: {b.dtype}")
    print(f"bc shape: {bc.shape}")
    print(f"F dtype: {F.dtype}")

    # Compute LSCM parameterization
    uv, _ = igl.lscm(V, F, b, bc)
    print(f"uv shape after LSCM: {uv.shape}")

    if uv.shape[1] != 2:
        print("ERROR: UV should have 2 columns!")
        return None, None

    # Now fit the UV to match original 3D position
    uv_scaled, rotation, translation, rmse_error = fit_uv_to_original_position(uv, V, F)

    print(f"rotation: {rotation}")
    print(f"translation: {translation}")

    print(f"rmse_error: {rmse_error:.4f}")

    return uv_scaled, rotation, translation


def fit_uv_to_original_position(uv, vertices_3d, faces):
    """
    Fits a similarity transform (scale, 3D rotation, translation) that maps 2D UV coordinates
    to 3D vertices as closely as possible.

    Returns:
        uv_scaled: Transformed UV coordinates in 3D (N x 3)
        R_lifted: 3D rotation matrix (3 x 3) applied to lifted UVs
        t: Translation vector (3,)
        rmse: Root mean square error between transformed UVs and 3D vertices
    """
    print("UV shape:", uv.shape)
    print("3D shape:", vertices_3d.shape)
    print("Same number of points:", uv.shape[0] == vertices_3d.shape[0])

    N = uv.shape[0]
    assert uv.shape[0] == vertices_3d.shape[0]
    print(f"Fitting UVs to {N} vertices...")
    # Center both sets
    uv_mean = uv.mean(axis=0)
    v3d_mean = vertices_3d.mean(axis=0)

    uv_centered = uv - uv_mean
    v3d_centered = vertices_3d - v3d_mean

    # Optimal scale (scalar)
    # sample faces to find scale
    n_samples = 100
    sample_faces = np.random.choice(np.arange(len(faces)), n_samples, replace=True)
    scale = 0
    for _f in sample_faces:
        v0, v1, v2 = faces[_f]
        xyz0 = vertices_3d[v0]
        xyz1 = vertices_3d[v1]
        xyz2 = vertices_3d[v2]
        uv0 = uv[v0]
        uv1 = uv[v1]
        uv2 = uv[v2]
        scale += np.sqrt(np.sum((xyz0 - xyz1) ** 2) / np.sum((uv0 - uv1) ** 2))
        scale += np.sqrt(np.sum((xyz1 - xyz2) ** 2) / np.sum((uv1 - uv2) ** 2))
        scale += np.sqrt(np.sum((xyz2 - xyz0) ** 2) / np.sum((uv2 - uv0) ** 2))
    scale /= n_samples * 3

    # Apply scale to UVs
    uv_scaled_2d = scale * uv_centered

    # Lift to 3D
    uv_scaled_3d = np.hstack([uv_scaled_2d, np.zeros((N, 1))])

    # Compute cross-covariance matrix
    C = v3d_centered.T @ uv_scaled_3d

    # SVD for rotation
    U, _, Vt = np.linalg.svd(C)
    R = U @ Vt

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # Compute translation
    t = np.mean(vertices_3d - (R @ uv_scaled_3d.T).T, axis=0)

    # Final reconstruction
    uv_reconstructed_3d = (R @ uv_scaled_3d.T).T + t

    # RMSE
    rmse = np.sqrt(np.mean(np.sum((uv_reconstructed_3d - vertices_3d) ** 2, axis=1)))

    return uv_scaled_2d, R, t, rmse  # uv_scaled in 2D space


def _extract_segments(panels: List[ColorPanel], sewing_pairs: List[Tuple]):
    panel_to_segments = defaultdict(list)
    
    # Create mapping from color_id to panel
    id_to_panel = {panel.color_id: panel for panel in panels}
    print(f"Available panels: {len(panels)} with IDs: {[p.color_id for p in panels]}")
    
    # Filter sewing pairs to only include valid panel IDs
    valid_sewing_pairs = []
    for i, pair in enumerate(sewing_pairs):
        s0, s1 = pair
        if s0[0] in id_to_panel and s1[0] in id_to_panel:
            valid_sewing_pairs.append(pair)
            print(f"Valid sewing pair {len(valid_sewing_pairs)-1}: panel {s0[0]} and panel {s1[0]}")
        else:
            print(f"Skipping invalid sewing pair {i}: panel {s0[0]} and panel {s1[0]} (panels not found)")
    
    print(f"Using {len(valid_sewing_pairs)} out of {len(sewing_pairs)} sewing pairs")
    
    for i, pair in enumerate(valid_sewing_pairs):
        s0, s1 = pair
        panel_to_segments[s0[0]].append((i, s0[1], s0[2]))
        panel_to_segments[s1[0]].append((i, s1[1], s1[2]))
    
    print(f"Panel IDs referenced in valid sewing pairs: {list(panel_to_segments.keys())}")
    
    for panel_id, segments in panel_to_segments.items():
        panel_size = len(id_to_panel[panel_id].boundary_vertices)
        segments = segments.copy()
        segments.sort(key=lambda x: x[1])
        # add in non-sewing segments
        i_curr = segments[0][1] - 1

        all_segments = []
        for i in range(len(segments)):
            sewing_id, e0, e1 = segments[i]
            if e0 == i_curr + 1:
                all_segments.append((sewing_id, e0, e1))
                i_curr = e1
            else:
                all_segments.append((None, i_curr + 1, e0 - 1))
                all_segments.append((sewing_id, e0, e1))
                i_curr = e1
        if i_curr % panel_size != (segments[0][1] - 1) % panel_size:
            all_segments.append((None, i_curr + 1, segments[0][1] - 1))
        panel_to_segments[panel_id] = all_segments

    # reverse index (moved outside the loop)
    segment_to_panels = defaultdict(list)
    for panel_id, segments in panel_to_segments.items():
        for j, (segment_id, e0, e1) in enumerate(segments):
            if segment_id is not None:
                segment_to_panels[segment_id].append((panel_id, j))

    return panel_to_segments, segment_to_panels


def make_spec(panels: List[ColorPanel], sewing_pairs: List[Tuple]):
    p2s, s2p = _extract_segments(panels, sewing_pairs)

    # Create mapping from color_id to panel
    id_to_panel = {panel.color_id: panel for panel in panels}

    # make a spec
    spec = []
    for i, panel in enumerate(panels):
        uv_scaled, rotation, translation = uv_unfold(panel)
        gc_panel = Panel(name=f"panel_{i}")
        spec.append(gc_panel)

        # Set translation and rotation properly
        gc_panel.translation = np.array(translation)
        gc_panel.rotation = R.from_matrix(rotation)
        
        print(f"Panel {i}: translation = {translation}")
        print(f"Panel {i}: rotation matrix = \n{rotation}")
        print(f"Panel {i}: rotation euler = {gc_panel.rotation.as_euler('XYZ', degrees=True)}")
        # Use UV coordinates for 2D pattern, not 3D mesh vertices
        edges = []
        
        # Check if panel has segments, if not create a full boundary edge
        if panel.color_id not in p2s or len(p2s[panel.color_id]) == 0:
            print(f"Panel {panel.color_id} has no segments, creating full boundary")
            # Create a single edge for the entire boundary
            vids = panel.boundary_vertices.tolist()
            segment_points = [uv_scaled[j].tolist() for j in vids]
            p0, p1, p2, p3 = fit_bezier_curve(segment_points)
            
            edges.append(
                CurveEdge(
                    start=uv_scaled[vids[0]].tolist(),
                    end=uv_scaled[vids[-1]].tolist(),
                    control_points=[p1.tolist(), p2.tolist()],
                    relative=False,
                )
            )
        else:
            for _, e0, e1 in p2s[panel.color_id]:
                if e1 < e0:
                    e1 += len(panel.boundary_vertices)
                if e1 >= len(panel.boundary_vertices) - 1:
                    vids = (
                        panel.boundary_vertices[e0:].tolist()
                        + panel.boundary_vertices[
                            : (e1 + 2) % len(panel.boundary_vertices)
                        ].tolist()
                    )
                else:
                    vids = panel.boundary_vertices[e0 : e1 + 2].tolist()

                segment_points = [uv_scaled[j].tolist() for j in vids]
                p0, p1, p2, p3 = fit_bezier_curve(segment_points)

                edges.append(
                    CurveEdge(
                        start=uv_scaled[vids[0]].tolist(),
                        end=uv_scaled[vids[-1]].tolist(),
                        control_points=[p1.tolist(), p2.tolist()],
                        relative=False,
                    )
                )
        edges = EdgeSequence(edges)
        gc_panel.edges = edges

    # put into a component
    component = Component(name="component")
    # assign each panel as a named attribute of the component
    for i, panel in enumerate(spec):
        setattr(component, f"panel_{panels[i].color_id}", panel)

    # add sewing pairs
    stitches = []
    for _, (s0, s1) in s2p.items():
        panel0 = getattr(component, f"panel_{s0[0]}")
        panel1 = getattr(component, f"panel_{s1[0]}")

        stitches.append(
            (
                pyg.Interface(panel0, panel0.edges[s0[1]]),
                pyg.Interface(panel1, panel1.edges[s1[1]]),
            )
        )

    print(stitches)

    component.stitching_rules = pyg.Stitches(*stitches)

    # assemble and to json
    pattern = component.assembly()

    # Save as json file
    sys_props = Properties("gc_files/system.json")
    folder = pattern.serialize(
        Path(sys_props["output"]),
        tag="",
        to_subfolder=True,
        with_3d=True,
        with_text=False,
        view_ids=False,
        with_printable=True, 
    )

    print(f"Success! pattern saved to {folder}")

    return folder


if __name__ == "__main__":
    panels, sewing_pairs = load_data()
    folder = make_spec(panels, sewing_pairs)
    print(f"pattern saved to {folder}")
