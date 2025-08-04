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


def check_self_intersection(points):
    """
    Check if a polygon defined by points has self-intersections.
    Returns True if there are self-intersections, False otherwise.
    """
    from shapely.geometry import Polygon
    from shapely.validation import explain_validity
    
    try:
        polygon = Polygon(points)
        if not polygon.is_valid:
            reason = explain_validity(polygon)
            if "Self-intersection" in reason:
                return True
    except:
        # If we can't create a polygon, assume there might be issues
        return True
    
    return False


def fix_self_intersection(points, max_iterations=5):
    """
    Try to fix self-intersections in a polygon by smoothing or reordering points.
    """
    import numpy as np
    from scipy.spatial.distance import cdist
    
    points = np.array(points)
    
    # Method 1: Try removing duplicate or very close points
    unique_points = []
    for i, point in enumerate(points):
        if i == 0:
            unique_points.append(point)
        else:
            # Check distance to previous points
            distances = [np.linalg.norm(point - p) for p in unique_points]
            if min(distances) > 1e-6:  # Threshold for considering points different
                unique_points.append(point)
    
    if len(unique_points) < 3:
        return points.tolist()  # Can't form a polygon
    
    points = np.array(unique_points)
    
    # Method 2: Check if reordering helps (sort by angle from centroid)
    centroid = np.mean(points, axis=0)
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
    sorted_indices = np.argsort(angles)
    reordered_points = points[sorted_indices]
    
    if not check_self_intersection(reordered_points):
        return reordered_points.tolist()
    
    # Method 3: Convex hull as fallback
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        hull_points = points[hull.vertices]
        return hull_points.tolist()
    except:
        pass
    
    # If all else fails, return original points
    return points.tolist()


def validate_and_fix_panel_boundaries(panels: List[ColorPanel]):
    """
    Validate panel boundaries and fix self-intersections if found.
    """
    for panel in panels:
        print(f"Validating panel {panel.color_id}...")
        
        # Get boundary points in 2D (project to XY plane)
        boundary_3d = panel.mesh.vertices[panel.boundary_vertices]
        boundary_2d = boundary_3d[:, :2]  # Project to XY plane
        
        if check_self_intersection(boundary_2d):
            print(f"Panel {panel.color_id}: Self-intersection detected, attempting to fix...")
            
            # Try to fix the intersection
            fixed_boundary_2d = fix_self_intersection(boundary_2d)
            
            # Update the boundary vertices if we have a valid fix
            if len(fixed_boundary_2d) >= 3 and not check_self_intersection(fixed_boundary_2d):
                print(f"Panel {panel.color_id}: Successfully fixed self-intersection")
                
                # Find the closest vertices in the original mesh for the fixed boundary
                from scipy.spatial.distance import cdist
                fixed_boundary_3d = np.column_stack([fixed_boundary_2d, boundary_3d[:len(fixed_boundary_2d), 2]])
                
                # Map back to vertex indices
                distances = cdist(fixed_boundary_3d, panel.mesh.vertices)
                new_boundary_indices = []
                for point in fixed_boundary_3d:
                    closest_idx = np.argmin(np.sum((panel.mesh.vertices - point) ** 2, axis=1))
                    new_boundary_indices.append(closest_idx)
                
                panel.boundary_vertices = np.array(new_boundary_indices)
            else:
                print(f"Panel {panel.color_id}: Could not fix self-intersection automatically")


def compute_boundary_orientation(vertices_2d, boundary_indices):
    """
    Compute the orientation of a boundary loop (clockwise or counter-clockwise).
    Returns True if counter-clockwise, False if clockwise.
    Uses the shoelace formula to compute the signed area.
    """
    area = 0.0
    n = len(boundary_indices)
    for i in range(n):
        j = (i + 1) % n
        v1 = vertices_2d[boundary_indices[i]]
        v2 = vertices_2d[boundary_indices[j]]
        area += (v2[0] - v1[0]) * (v2[1] + v1[1])
    return area > 0  # Positive area means counter-clockwise


def ensure_consistent_orientation(panels: List[ColorPanel], target_ccw=True):
    """
    Ensure all panels have consistent boundary orientation.
    
    Args:
        panels: List of ColorPanel objects
        target_ccw: If True, make all boundaries counter-clockwise; if False, clockwise
    """
    for panel in panels:
        # Get 2D projection of vertices for orientation check
        vertices = panel.mesh.vertices
        # Project to XY plane for orientation check
        vertices_2d = vertices[:, :2]
        
        # Check current orientation
        is_ccw = compute_boundary_orientation(vertices_2d, panel.boundary_vertices)
        
        # Reverse if needed
        if is_ccw != target_ccw:
            print(f"Panel {panel.color_id}: Reversing boundary orientation from {'CCW' if is_ccw else 'CW'} to {'CCW' if target_ccw else 'CW'}")
            panel.boundary_vertices = panel.boundary_vertices[::-1]
            # Also reverse boundary edges if they exist
            if hasattr(panel, 'boundary_edges') and panel.boundary_edges is not None:
                panel.boundary_edges = panel.boundary_edges[::-1]


def check_sewing_edge_direction(panel1: ColorPanel, edge1_start: int, edge1_end: int,
                               panel2: ColorPanel, edge2_start: int, edge2_end: int):
    """
    Check if two sewing edges have compatible directions.
    Returns True if they need to be reversed relative to each other.
    """
    # Get the vertices for each edge
    n1 = len(panel1.boundary_vertices)
    n2 = len(panel2.boundary_vertices)
    
    # Handle wrap-around for edge indices
    if edge1_end < edge1_start:
        edge1_indices = list(range(edge1_start, n1)) + list(range(0, edge1_end + 1))
    else:
        edge1_indices = list(range(edge1_start, edge1_end + 1))
    
    if edge2_end < edge2_start:
        edge2_indices = list(range(edge2_start, n2)) + list(range(0, edge2_end + 1))
    else:
        edge2_indices = list(range(edge2_start, edge2_end + 1))
    
    # Get actual vertex positions
    edge1_verts = [panel1.mesh.vertices[panel1.boundary_vertices[i]] for i in edge1_indices]
    edge2_verts = [panel2.mesh.vertices[panel2.boundary_vertices[i]] for i in edge2_indices]
    
    # Calculate edge directions
    if len(edge1_verts) >= 2 and len(edge2_verts) >= 2:
        dir1 = edge1_verts[-1] - edge1_verts[0]
        dir2 = edge2_verts[-1] - edge2_verts[0]
        
        # For sewing, edges should go in opposite directions when aligned
        # (because they face each other when sewn)
        dot_product = np.dot(dir1, dir2)
        
        # If dot product is positive, edges go in same direction and need reversal
        return dot_product > 0
    
    return False


def fix_sewing_pairs_direction(panels: List[ColorPanel], sewing_pairs: List[Tuple]) -> List[Tuple]:
    """
    Fix sewing pairs to ensure correct edge direction matching.
    Returns a new list of sewing pairs with corrected directions.
    Sewing pair format: ((panel_id, edge_start, edge_end, bool_flag), (panel_id, edge_start, edge_end, bool_flag))
    """
    id_to_panel = {panel.color_id: panel for panel in panels}
    fixed_pairs = []
    
    for i, pair in enumerate(sewing_pairs):
        s0, s1 = pair
        
        # Extract panel IDs and edge indices
        panel1_id, edge1_start, edge1_end = s0[0], s0[1], s0[2]
        panel2_id, edge2_start, edge2_end = s1[0], s1[1], s1[2]
        
        if panel1_id not in id_to_panel or panel2_id not in id_to_panel:
            fixed_pairs.append(pair)  # Keep as is if panels not found
            continue
            
        panel1 = id_to_panel[panel1_id]
        panel2 = id_to_panel[panel2_id]
        
        # Check if edge directions need adjustment
        need_reverse = check_sewing_edge_direction(
            panel1, edge1_start, edge1_end,
            panel2, edge2_start, edge2_end
        )
        
        if need_reverse:
            print(f"Sewing pair {i}: Reversing edge direction for panel {panel2_id}")
            # Reverse the second edge, keeping the bool flag
            s1_fixed = (panel2_id, edge2_end, edge2_start, s1[3])
            fixed_pairs.append((s0, s1_fixed))
        else:
            fixed_pairs.append(pair)
    
    return fixed_pairs


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
            mesh.vertices = mesh.vertices * 100 + [0, 100, 0]
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
            # print(f"Sewing pair {line_num}: panel IDs {pair[0][0]} and {pair[1][0]}")
            # Debug: print the full structure
            # print(f"  Full structure: {pair}")
            # print(f"  s0 length: {len(pair[0])}, s1 length: {len(pair[1])}")
        except Exception as e:
            print(f"Error parsing sewing pair line {line_num}: {e}")
    
    print(f"Total sewing pairs: {len(sewing_pairs)}")
    
    # Validate and fix panel boundaries first
    validate_and_fix_panel_boundaries(panels)
    
    # Ensure consistent boundary orientation for all panels
    ensure_consistent_orientation(panels, target_ccw=True)
    
    return panels, sewing_pairs
    

def uv_unfold(panel):
    import igl
    import numpy as np
    print("IGL path:", igl.__file__)

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
    import igl
    
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

    # Check if UV orientation matches 3D orientation
    # If UV is flipped, we need to flip it back
    boundary = igl.boundary_loop(faces)
    uv_is_ccw = compute_boundary_orientation(uv_scaled_2d, boundary)
    mesh_vertices_2d = vertices_3d[:, :2]  # Project to XY plane
    mesh_is_ccw = compute_boundary_orientation(mesh_vertices_2d, boundary)
    
    if uv_is_ccw != mesh_is_ccw:
        print(f"UV orientation mismatch detected. Flipping UV coordinates.")
        uv_scaled_2d[:, 0] = -uv_scaled_2d[:, 0]  # Flip X coordinate
        # Update rotation matrix accordingly
        R[:, 0] = -R[:, 0]  # Flip first column

    # Check for self-intersections in UV coordinates
    boundary = igl.boundary_loop(faces)
    uv_boundary_points = uv_scaled_2d[boundary]
    
    if check_self_intersection(uv_boundary_points):
        print(f"UV boundary has self-intersection, attempting to fix...")
        fixed_uv_boundary = fix_self_intersection(uv_boundary_points)
        
        if not check_self_intersection(fixed_uv_boundary):
            print(f"Successfully fixed UV self-intersection")
            # Update the UV coordinates for boundary vertices
            for i, fixed_point in enumerate(fixed_uv_boundary):
                if i < len(boundary):
                    uv_scaled_2d[boundary[i]] = fixed_point
        else:
            print(f"Warning: Could not fix UV self-intersection")

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


def make_spec_origin(panels: List[ColorPanel], sewing_pairs: List[Tuple]):
    # Fix sewing pairs direction before processing
    sewing_pairs = fix_sewing_pairs_direction(panels, sewing_pairs)
    
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
    sys_props = Properties("../gc_files/system.json")
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

def _to_rel(p0, p1, p):
    # compute relative coordinates
    vdir = p1 - p0
    vorth = np.array([-vdir[1], vdir[0]])

    # Avoid division by zero
    vdir_dot = np.dot(vdir, vdir)
    if vdir_dot < 1e-12:  # Very small vector, treat as zero
        return [0.0, 0.0]
    
    rel0 = np.dot(p - p0, vdir) / vdir_dot
    p_proj = p0 + vdir * rel0
    p_diff = p - p_proj
    
    # Avoid division by zero for orthogonal component
    vorth_norm = np.linalg.norm(vorth)
    if vorth_norm < 1e-12:
        rel1 = 0.0
    else:
        rel1 = np.dot(p_diff, vorth) / (vorth_norm * vorth_norm)
    
    return [rel0, rel1]


def resample_edge_points(points, target_count):
    """
    Resample a list of edge points to have exactly target_count points.
    Uses linear interpolation between existing points.
    """
    if len(points) == target_count:
        return points
    
    import numpy as np
    points = np.array(points)
    
    if len(points) < 2:
        return points.tolist()
    
    # Calculate cumulative distances along the edge
    distances = [0]
    for i in range(1, len(points)):
        dist = np.linalg.norm(points[i] - points[i-1])
        distances.append(distances[-1] + dist)
    
    total_length = distances[-1]
    if total_length < 1e-12:
        # Edge has zero length, just return repeated points
        return [points[0].tolist()] * target_count
    
    # Normalize distances
    distances = np.array(distances) / total_length
    
    # Create target parameter values
    target_params = np.linspace(0, 1, target_count)
    
    # Interpolate to get new points
    resampled_points = []
    for t in target_params:
        # Find the segment containing parameter t
        for i in range(len(distances) - 1):
            if distances[i] <= t <= distances[i + 1]:
                # Linear interpolation within this segment
                if distances[i + 1] - distances[i] < 1e-12:
                    # Avoid division by zero
                    new_point = points[i]
                else:
                    alpha = (t - distances[i]) / (distances[i + 1] - distances[i])
                    new_point = (1 - alpha) * points[i] + alpha * points[i + 1]
                resampled_points.append(new_point.tolist())
                break
        else:
            # t is exactly 1.0, use last point
            resampled_points.append(points[-1].tolist())
    
    return resampled_points


def get_edge_points_from_segment(panel, segment_start, segment_end, uv_coords):
    """
    Extract edge points from a panel segment.
    """
    if segment_end < segment_start:
        # Wrap-around case
        indices = list(range(segment_start, len(panel.boundary_vertices))) + list(range(0, segment_end + 1))
    else:
        indices = list(range(segment_start, segment_end + 1))
    
    return [uv_coords[panel.boundary_vertices[i]].tolist() for i in indices]


def make_spec(panels: List[ColorPanel], sewing_pairs: List[Tuple]):
    # Fix sewing pairs direction before processing
    sewing_pairs = fix_sewing_pairs_direction(panels, sewing_pairs)
    
    p2s, s2p = _extract_segments(panels, sewing_pairs)

    # Create mapping from color_id to panel and panel to index
    id_to_panel = {panel.color_id: panel for panel in panels}
    id_to_index = {panel.color_id: i for i, panel in enumerate(panels)}

    # make a spec
    spec = []
    panel_json = {}
    panel_uv_coords = {}  # Store UV coordinates for each panel
    
    for i, panel in enumerate(panels):
        uv_scaled, rotation, translation = uv_unfold(panel)
        panel_uv_coords[panel.color_id] = uv_scaled

        panel_json[f"panel_{i}"] = {
            "translation": translation.tolist(),
            "rotation": R.from_matrix(rotation).as_euler("xyz", degrees=True).tolist(),
            "vertices": [],
            "edges": [],
        }

        # Check if panel has segments, if not create a full boundary edge
        if panel.color_id not in p2s or len(p2s[panel.color_id]) == 0:
            print(f"Panel {panel.color_id} has no segments, creating full boundary")
            # Create a single edge for the entire boundary
            vids = panel.boundary_vertices.tolist()
            segment_points = [uv_scaled[j].tolist() for j in vids]
            
            # Add all boundary vertices
            panel_json[f"panel_{i}"]["vertices"] = segment_points
            
            p0, p1, p2, p3 = fit_bezier_curve(segment_points)
            
            panel_json[f"panel_{i}"]["edges"].append(
                {
                    "endpoints": [0, len(segment_points) - 1],
                    "curvature": {
                        "type": "cubic",
                        "params": [_to_rel(p0, p3, p1), _to_rel(p0, p3, p2)],
                    },
                }
            )
        else:
            # Collect all boundary vertices in order
            all_boundary_points = [uv_scaled[j].tolist() for j in panel.boundary_vertices]
            panel_json[f"panel_{i}"]["vertices"] = all_boundary_points
            
            vertex_idx = 0
            for j, (_, e0, e1) in enumerate(p2s[panel.color_id]):
                # Calculate the number of vertices in this segment
                if e1 < e0:
                    # Wrap-around case
                    segment_length = len(panel.boundary_vertices) - e0 + e1 + 1
                    vids = (
                        panel.boundary_vertices[e0:].tolist()
                        + panel.boundary_vertices[: e1 + 1].tolist()
                    )
                else:
                    segment_length = e1 - e0 + 1
                    vids = panel.boundary_vertices[e0 : e1 + 1].tolist()

                segment_points = [uv_scaled[k].tolist() for k in vids]
                p0, p1, p2, p3 = fit_bezier_curve(segment_points)

                # Endpoints refer to indices in the vertices array
                start_idx = e0
                end_idx = e1 if e1 >= e0 else e1 + len(panel.boundary_vertices)
                end_idx = end_idx % len(panel.boundary_vertices)
                
                panel_json[f"panel_{i}"]["edges"].append(
                    {
                        "endpoints": [start_idx, end_idx],
                        "curvature": {
                            "type": "cubic",
                            "params": [_to_rel(p0, p3, p1), _to_rel(p0, p3, p2)],
                        },
                    }
                )

            # No need to update last edge endpoints since we're using actual indices

    # Now process stitches and ensure edge point counts match
    stitches_json = []
    
    # First, determine target point counts for all sewing edges
    edge_target_counts = {}  # (panel_id, edge_idx) -> target_count
    
    for sewing_id, (s0, s1) in s2p.items():
        panel1_id, edge1_idx = s0[0], s0[1] 
        panel2_id, edge2_idx = s1[0], s1[1]
        
        # Get the corresponding sewing pair data
        corresponding_pair = sewing_pairs[sewing_id]
        
        # Get edge vertex counts from the sewing pair
        p1_data, p2_data = corresponding_pair
        p1_start, p1_end = p1_data[1], p1_data[2]
        p2_start, p2_end = p2_data[1], p2_data[2]
        
        # Calculate actual edge lengths
        panel1 = id_to_panel[panel1_id]
        panel2 = id_to_panel[panel2_id]
        
        if p1_end < p1_start:
            p1_edge_len = len(panel1.boundary_vertices) - p1_start + p1_end + 1
        else:
            p1_edge_len = p1_end - p1_start + 1
            
        if p2_end < p2_start:
            p2_edge_len = len(panel2.boundary_vertices) - p2_start + p2_end + 1
        else:
            p2_edge_len = p2_end - p2_start + 1
        
        # Use a reasonable target count that ensures both edges have the same number of points
        # Use the minimum to avoid over-sampling, but ensure it's at least 3 points
        target_count = max(min(p1_edge_len, p2_edge_len), 3)
        # Cap at a reasonable maximum to avoid excessive computation
        target_count = min(target_count, 100)
        
        edge_target_counts[(panel1_id, edge1_idx)] = target_count
        edge_target_counts[(panel2_id, edge2_idx)] = target_count
        
        print(f"Sewing edge {sewing_id}: Panel {panel1_id} edge {p1_edge_len} points, Panel {panel2_id} edge {p2_edge_len} points, target {target_count}")
    
    # Now regenerate vertices for panels that have sewing edges, using target counts
    for i, panel in enumerate(panels):
        panel_id = panel.color_id
        
        if panel_id in p2s and len(p2s[panel_id]) > 0:
            # Check if any edges of this panel need resampling
            panel_needs_resampling = any((panel_id, j) in edge_target_counts for j in range(len(p2s[panel_id])))
            
            if panel_needs_resampling:
                # Rebuild this panel's vertices and edges with proper point counts
                print(f"Resampling vertices for panel {panel_id} (index {i})")
                new_vertices = []
                
                for j, (_, e0, e1) in enumerate(p2s[panel_id]):
                    # Get target count for this edge
                    target_count = edge_target_counts.get((panel_id, j), None)
                    
                    # Get original edge points
                    original_points = get_edge_points_from_segment(panel, e0, e1, panel_uv_coords[panel_id])
                    
                    if target_count is not None:
                        print(f"  Edge {j}: resampling from {len(original_points)} to {target_count} points")
                        # Resample to target count
                        edge_points = resample_edge_points(original_points, target_count)
                    else:
                        edge_points = original_points
                    
                    # For the last edge, include all points to close the loop
                    if j == len(p2s[panel_id]) - 1:
                        new_vertices.extend(edge_points)
                    else:
                        # Exclude last point to avoid duplication with next edge's first point
                        new_vertices.extend(edge_points[:-1])
                
                # Update vertices
                panel_json[f"panel_{i}"]["vertices"] = new_vertices
                
                # Update edge endpoints
                vertex_offset = 0
                for j, (_, e0, e1) in enumerate(p2s[panel_id]):
                    target_count = edge_target_counts.get((panel_id, j), None)
                    if target_count is not None:
                        edge_length = target_count
                    else:
                        original_points = get_edge_points_from_segment(panel, e0, e1, panel_uv_coords[panel_id])
                        edge_length = len(original_points)
                    
                    start_idx = vertex_offset
                    if j == len(p2s[panel_id]) - 1:
                        # Last edge closes the loop
                        end_idx = 0  
                    else:
                        end_idx = vertex_offset + edge_length - 1
                        vertex_offset += edge_length - 1
                    
                    panel_json[f"panel_{i}"]["edges"][j]["endpoints"] = [start_idx, end_idx]
    
    # Finally, create stitches JSON
    for sewing_id, (s0, s1) in s2p.items():
        panel1_id, edge1_idx = s0[0], s0[1] 
        panel2_id, edge2_idx = s1[0], s1[1]
        
        # Use panel indices instead of color_ids
        panel1_idx = id_to_index[panel1_id]
        panel2_idx = id_to_index[panel2_id]
        
        stitches_json.append(
            [
                {"panel": f"panel_{panel1_idx}", "edge": edge1_idx},
                {"panel": f"panel_{panel2_idx}", "edge": edge2_idx},
            ]
        )

    spec_json = {
        "pattern": {
            "panels": panel_json,
            "stitches": stitches_json,
            "panel_order": [f"panel_{i}" for i in range(len(panels))],
            "parameters": {},
            "parameter_order": [],
            "properties": {
                "curvature_coords": "relative",
                "normalize_panel_translation": False,
                "normalized_edge_loops": True,
                "units_in_meter": 100,
            },
        }
    }
    with open("../outputs/sim_json/test_spec.json", "w") as f:
        json.dump(spec_json, f, indent=4)

    print(f"pattern saved to outputs/sim_json/test_spec.json")

if __name__ == "__main__":
    panels, sewing_pairs = load_data()
    # Use make_spec to generate JSON that test_sim.py uses
    make_spec(panels, sewing_pairs)
    print("Specification JSON saved to outputs/sim_json/test_spec.json")
    
    # Also test make_spec_origin
    # folder = make_spec_origin(panels, sewing_pairs)
    # print(f"pattern saved to {folder}")
