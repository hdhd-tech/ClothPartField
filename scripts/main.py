import trimesh
import numpy as np
from collections import Counter, defaultdict
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pairing import find_sewing_pairs, plot_all_segments_overview, plot_sewing_pairs

from typedef import ColorPanel


def extract_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Extract face colors using majority vote from vertex colors."""
    vertex_colors = mesh.visual.vertex_colors
    unique_colors = np.unique(vertex_colors, axis=0)
    color_to_id = {tuple(color): i for i, color in enumerate(unique_colors)}

    face_colors = np.zeros(mesh.faces.shape[0], dtype=int)
    for i, vertices in enumerate(mesh.faces):
        color_counts = Counter()
        for vertex in vertices:
            color = color_to_id[tuple(vertex_colors[vertex])]
            color_counts[color] += 1
        face_colors[i] = max(color_counts, key=color_counts.get)

    return face_colors

def handle_complex_boundary(edge_mapping, boundary_edges, bad_vertices):
    """Handle boundaries with complex topology."""
    print("Handling complex boundary topology...")
    print(f"  Total boundary edges to process: {len(boundary_edges)}")
    print(f"  Bad vertices: {len(bad_vertices)}")
    
    # Strategy 1: Extract multiple disconnected loops
    all_boundary_vertices = []
    all_boundary_edges = []
    unvisited_edges = set(boundary_edges)
    
    component_count = 0
    while unvisited_edges:
        # Start a new component
        start_edge = next(iter(unvisited_edges))
        component_vertices, component_edges = extract_boundary_component(
            edge_mapping, start_edge, unvisited_edges, bad_vertices
        )
        
        component_count += 1
        print(f"  Component {component_count}: {len(component_vertices)} vertices, {len(component_edges)} edges")
        
        if len(component_vertices) >= 3:  # Only keep meaningful components
            all_boundary_vertices.extend(component_vertices)
            all_boundary_edges.extend(component_edges)
        else:
            print(f"    Skipping component {component_count} (too small)")
    
    print(f"  Final result: {len(all_boundary_vertices)} vertices, {len(all_boundary_edges)} edges")
    return all_boundary_vertices, all_boundary_edges

def extract_boundary_component(edge_mapping, start_edge, unvisited_edges, bad_vertices):
    """Extract a single boundary component, handling problematic vertices."""
    component_vertices = []
    component_edges = []
    
    # Start from an edge
    current_v = start_edge[0]
    next_v = start_edge[1]
    
    # Try to follow the boundary as far as possible
    visited_vertices = set()
    
    while current_v not in visited_vertices:
        visited_vertices.add(current_v)
        component_vertices.append(current_v)
        
        # Add edge if it exists in unvisited
        edge = tuple(sorted([current_v, next_v]))
        if edge in unvisited_edges:
            component_edges.append((current_v, next_v))
            unvisited_edges.remove(edge)
        
        # Find next vertex
        neighbors = [v for v in edge_mapping[next_v] if v != current_v]
        
        if len(neighbors) == 0:
            # Dead end
            break
        elif len(neighbors) == 1:
            # Continue along boundary
            current_v, next_v = next_v, neighbors[0]
        else:
            # Handle T-junction or complex vertex
            print(f"T-junction at vertex {next_v} with neighbors {neighbors}")
            # Choose the unvisited neighbor, or break if all visited
            unvisited_neighbors = [v for v in neighbors if v not in visited_vertices]
            if unvisited_neighbors:
                current_v, next_v = next_v, unvisited_neighbors[0]
            else:
                break
    
    # Try to close the loop if possible
    if len(component_vertices) > 2:
        first_v = component_vertices[0]
        last_v = component_vertices[-1]
        if last_v in edge_mapping and first_v in edge_mapping[last_v]:
            edge = tuple(sorted([last_v, first_v]))
            if edge in unvisited_edges:
                component_edges.append((last_v, first_v))
                unvisited_edges.remove(edge)
    
    return component_vertices, component_edges

def extract_boundary(mesh: trimesh.Trimesh) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Extract boundary vertices and edges from a mesh with robust handling."""
    edge_counts = Counter()
    for face in mesh.faces:
        v0, v1, v2 = sorted(face.tolist())
        edge_counts[(v0, v1)] += 1
        edge_counts[(v1, v2)] += 1
        edge_counts[(v0, v2)] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    # Debug mesh statistics
    total_edges = len(edge_counts)
    print(f"  Mesh stats: {len(mesh.faces)} faces, {len(mesh.vertices)} vertices, {total_edges} total edges, {len(boundary_edges)} boundary edges")
    
    edge_mapping = defaultdict(list)
    for edge in boundary_edges:
        edge_mapping[edge[0]].append(edge[1])
        edge_mapping[edge[1]].append(edge[0])

    # Find vertices with bad degree (≠ 2)
    bad_vertices = {v: nbrs for v, nbrs in edge_mapping.items() if len(nbrs) != 2}
    
    if bad_vertices:
        print(f"Found {len(bad_vertices)} bad boundary vertices (degree≠2): {bad_vertices}")
        return handle_complex_boundary(edge_mapping, boundary_edges, bad_vertices)
    
    # Check if we have multiple disconnected boundary loops (all vertices have degree 2)
    all_boundary_vertices = set()
    for edge in boundary_edges:
        all_boundary_vertices.update(edge)
        
    if len(boundary_edges) > 20 and len(all_boundary_vertices) == len(boundary_edges):
        print(f"  Detected multiple disconnected boundary loops: {len(boundary_edges)} edges, {len(all_boundary_vertices)} vertices")
        # Use handle_complex_boundary to extract all loops
        return handle_complex_boundary(edge_mapping, boundary_edges, {})
    
    # Original simple loop extraction for clean boundaries
    v = boundary_edges[0][0]
    boundary_vertices = [v]
    visited = {v}

    while True:
        candidates = edge_mapping[v]
        if len(candidates) != 2:
            raise ValueError("Boundary is not a loop")
        v = candidates[0] if candidates[0] not in visited else candidates[1]
        if v in visited:
            break
        boundary_vertices.append(v)
        visited.add(v)

    boundary_edges = [
        (boundary_vertices[i], boundary_vertices[(i + 1) % len(boundary_vertices)])
        for i in range(len(boundary_vertices))
    ]

    return boundary_vertices, boundary_edges

def simplify_boundary_topology(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Attempt to simplify mesh topology before boundary extraction."""
    # Remove duplicate vertices
    #mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    
    # Try to fix winding
    mesh.fix_normals()
    
    # Remove small disconnected components
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Keep the largest component
        largest = max(components, key=lambda x: len(x.faces))
        print(f"Keeping largest component with {len(largest.faces)} faces out of {len(components)} components")
        return largest
    
    return mesh

def extract_largest_component(
    mesh: trimesh.Trimesh, face_colors: np.ndarray, color_id: int
) -> Optional[ColorPanel]:
    """Extract the largest connected component for a given color."""
    cc_face_indices = np.where(face_colors == color_id)[0]
    if len(cc_face_indices) == 0:
        return None

    cc_mesh = mesh.submesh([cc_face_indices])[0]
    face_adjacency = cc_mesh.face_adjacency
    components = trimesh.graph.connected_components(face_adjacency)

    if len(components) == 0:
        return None

    largest_component = max(components, key=len)
    print(
        f"Color {color_id}: Keeping {len(np.unique(largest_component))} out of {len(cc_mesh.faces)} faces."
    )

    largest_cc_mesh = cc_mesh.submesh([largest_component])[0]
    

    # Try to simplify topology before boundary extraction
    try:
        #largest_cc_mesh = simplify_boundary_topology(largest_cc_mesh)
        boundary_vertices, boundary_edges = extract_boundary(largest_cc_mesh)

            
    except Exception as e:
        print(f"Warning: Boundary extraction failed for color {color_id}: {e}")
        print("Attempting fallback boundary extraction...")
        boundary_vertices, boundary_edges = extract_boundary_fallback(largest_cc_mesh)

    return ColorPanel(
        color_id=color_id,
        boundary_vertices=boundary_vertices,
        boundary_edges=boundary_edges,
        mesh=largest_cc_mesh,
    )


def build_boundary_color_mapping(
    panels: List[ColorPanel], kdtree: KDTree
) -> defaultdict:
    """Build mapping from boundary edges to adjacent colors."""
    boundary_to_colors = defaultdict(list)

    for panel in panels:
        print(f"Processing panel {panel.color_id} with {len(panel.boundary_edges)} boundary edges")
        
        for edge in panel.boundary_edges:
            v0 = kdtree.query(panel.mesh.vertices[edge[0]])[1]
            v1 = kdtree.query(panel.mesh.vertices[edge[1]])[1]
            boundary_to_colors[(v0, v1)].append(panel.color_id)
            boundary_to_colors[(v1, v0)].append(panel.color_id)
    
    return boundary_to_colors

def compute_edge_labels(
    panels: List[ColorPanel], boundary_to_colors: defaultdict, kdtree: KDTree
) -> List[List[int]]:
    """Compute edge labels for each panel based on adjacent colors."""
    edge_labels = []

    for panel in panels:
        color_adj = []
            
        for edge in panel.boundary_edges:
            v0 = kdtree.query(panel.mesh.vertices[edge[0]])[1]
            v1 = kdtree.query(panel.mesh.vertices[edge[1]])[1]
            cand = set(boundary_to_colors[(v0, v1)]) - {panel.color_id}
            
            if len(cand) == 1:
                color_adj.append(cand.pop())
            elif len(cand) == 0:
                color_adj.append(-1)
            else:
                raise ValueError("Multiple colors found for edge")

        edge_labels.append(color_adj)
        print(f"Panel {panel.color_id} edge labels: {color_adj}")

    return edge_labels


def process_mesh(mesh_path: str, min_faces_threshold: int = 15) -> Tuple[List[ColorPanel], List[List[int]], KDTree]:
    """Main processing pipeline for mesh segmentation."""
    mesh = trimesh.load(mesh_path, process=True)
    face_colors = extract_colors(mesh)
    kdtree = KDTree(mesh.vertices)

    # Extract panels for each color
    panels = []
    for color_id in range(max(face_colors) + 1):
        color_face_count = np.sum(face_colors == color_id)
        print(f"Processing color {color_id}")
        if color_face_count < min_faces_threshold:
            print(f"  Skipping color {color_id}: only {color_face_count} faces (< {min_faces_threshold} threshold)")
            continue
        panel = extract_largest_component(mesh, face_colors, color_id)
        if panel:
            panels.append(panel)

    # Build boundary mappings and compute edge labels
    boundary_to_colors = build_boundary_color_mapping(panels, kdtree)
    edge_labels = compute_edge_labels(panels, boundary_to_colors, kdtree)

    return panels, edge_labels, kdtree


def save_results(
    panels: List[ColorPanel],
    pairs: List,
    output_dir: str = "../outputs",
):
    """Save results to files."""
    with open(f"{output_dir}/sewing_pairs/dress_colored_vert_b_sewing_pairs.txt", "w") as f:
        for pair in pairs:
            f.write(f"{pair}\n")
    for i, panel in enumerate(panels):
        panel.mesh.export(f"{output_dir}/mesh_panel/dress_colored_vert_b_panel_{i}.ply")
        with open(f"{output_dir}/sewing_pairs/boundary_vertices_{i}.txt", "w") as f:
            for v in panel.boundary_vertices:
                f.write(f"{v}\n")
        with open(f"{output_dir}/sewing_pairs/boundary_edges_{i}.txt", "w") as f:
            for edge in panel.boundary_edges:
                f.write(f"{edge[0]} {edge[1]}\n")


def extract_boundary_fallback(mesh: trimesh.Trimesh) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Fallback boundary extraction for very problematic meshes."""
    print("Using fallback boundary extraction...")
    
    # Get all boundary edges
    edge_counts = Counter()
    for face in mesh.faces:
        v0, v1, v2 = sorted(face.tolist())
        edge_counts[(v0, v1)] += 1
        edge_counts[(v1, v2)] += 1
        edge_counts[(v0, v2)] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    
    if not boundary_edges:
        print("Warning: No boundary edges found, creating artificial boundary")
        # Create a simple boundary from the convex hull
        hull = mesh.convex_hull
        return list(range(len(hull.vertices))), [(i, (i+1) % len(hull.vertices)) for i in range(len(hull.vertices))]
    
    # Just return all boundary vertices and edges without trying to form loops
    boundary_vertices = list(set([v for edge in boundary_edges for v in edge]))
    
    print(f"Fallback: Found {len(boundary_vertices)} boundary vertices and {len(boundary_edges)} boundary edges")
    
    return boundary_vertices, boundary_edges


def main():
    """Main function with clear pipeline."""
    # Process mesh
    panels, edge_labels, kdtree = process_mesh("../outputs/colored_meshes_outputs/dress_colored_vert.ply")


    pairs = find_sewing_pairs(edge_labels, panels, kdtree, min_len=10)
    print(f"Found {len(pairs)} sewing pairs: {pairs}")

    # Generate visualizations
    plot_sewing_pairs(pairs, panels)
    plot_all_segments_overview(pairs, panels)

    # Save results
    save_results(panels, pairs)


if __name__ == "__main__":
    main()
