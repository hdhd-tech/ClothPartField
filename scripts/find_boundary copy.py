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



LABEL_FILE = "outputs/labels/vertex_labels_full.npy"
MESH_FILE = "outputs/colored_meshes/dress_colored_vert.ply"
SAVE_DIR = "outputs/tmp"
os.makedirs(SAVE_DIR, exist_ok=True)

vertex_labels = np.load(LABEL_FILE)
mesh = trimesh.load(MESH_FILE, process=True)
vertices, faces = mesh.vertices, mesh.faces

G = mesh_to_graph(vertices, faces)

adj_pairs = find_adjacent_label_pairs(vertex_labels, faces)

all_paths = []

for label_a, label_b in adj_pairs:
    seam_A, seam_B = extract_seam_pairs(vertex_labels, faces, (label_a, label_b))
    if len(seam_A) == 0 or len(seam_B) == 0:
        continue  # Skip if invalid seam

    # Compute endpoints explicitly spanning the two seams
    v_start, v_end = get_furthest_cross_pair(vertices, seam_A, seam_B)

    # Compute shortest path on mesh edges
    path_vertices = shortest_path_on_mesh(G, v_start, v_end)
    path_pts = vertices[path_vertices]

    all_paths.append({
        "labels": (label_a, label_b),
        "start_vertex": v_start,
        "end_vertex": v_end,
        "path_pts": path_pts
    })

    np.save(os.path.join(SAVE_DIR, f"seam_{label_a}_{label_b}.npy"), path_pts)
    print(f"Processed seam {label_a}-{label_b}, path length: {len(path_pts)}")

def visualize_paths(mesh, paths):
    scene = trimesh.Scene(mesh)
    for path in paths:
        path_curve = trimesh.load_path(path["path_pts"])
        scene.add_geometry(path_curve)
    scene.show()

visualize_paths(mesh, all_paths)
