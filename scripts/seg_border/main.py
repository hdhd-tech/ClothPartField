import trimesh
import numpy as np
from collections import Counter, defaultdict
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Tuple, Optional

from pairing import find_sewing_pairs, plot_all_segments_overview, plot_sewing_pairs


@dataclass
class ColorPanel:
    """Represents a single colored panel with its boundary information."""

    color_id: int
    boundary_vertices: List[int]
    boundary_edges: List[Tuple[int, int]]
    mesh: trimesh.Trimesh


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


def extract_boundary(mesh: trimesh.Trimesh) -> Tuple[List[int], List[Tuple[int, int]]]:
    """Extract boundary vertices and edges from a mesh."""
    edge_counts = Counter()
    for face in mesh.faces:
        v0, v1, v2 = sorted(face.tolist())
        edge_counts[(v0, v1)] += 1
        edge_counts[(v1, v2)] += 1
        edge_counts[(v0, v2)] += 1

    boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]
    edge_mapping = defaultdict(list)
    for edge in boundary_edges:
        edge_mapping[edge[0]].append(edge[1])
        edge_mapping[edge[1]].append(edge[0])

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
    boundary_vertices, boundary_edges = extract_boundary(largest_cc_mesh)

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


def process_mesh(mesh_path: str) -> Tuple[List[ColorPanel], List[List[int]], KDTree]:
    """Main processing pipeline for mesh segmentation."""
    mesh = trimesh.load(mesh_path, process=True)
    face_colors = extract_colors(mesh)
    kdtree = KDTree(mesh.vertices)

    # Extract panels for each color
    panels = []
    for color_id in range(max(face_colors) + 1):
        print(f"Processing color {color_id}")
        panel = extract_largest_component(mesh, face_colors, color_id)
        if panel:
            panels.append(panel)

    # Build boundary mappings and compute edge labels
    boundary_to_colors = build_boundary_color_mapping(panels, kdtree)
    edge_labels = compute_edge_labels(panels, boundary_to_colors, kdtree)

    return panels, edge_labels, kdtree


def save_results(panels: List[ColorPanel], pairs: List, output_dir: str = "test_data"):
    """Save results to files."""
    with open(f"{output_dir}/dress_colored_vert_b_sewing_pairs.txt", "w") as f:
        for pair in pairs:
            f.write(f"{pair}\n")


def main():
    """Main function with clear pipeline."""
    # Process mesh
    panels, edge_labels, kdtree = process_mesh("test_data/dress_colored_vert_b.ply")

    # Find sewing pairs
    boundaries = [
        (panel.boundary_vertices, panel.boundary_edges, panel.mesh) for panel in panels
    ]
    pairs = find_sewing_pairs(edge_labels, boundaries, kdtree, min_len=6)
    print(f"Found {len(pairs)} sewing pairs: {pairs}")

    # Generate visualizations
    plot_sewing_pairs(pairs, boundaries)
    plot_all_segments_overview(pairs, boundaries)

    # Save results
    save_results(panels, pairs)


if __name__ == "__main__":
    main()
