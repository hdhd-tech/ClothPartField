#!/usr/bin/env python3
"""
Simple script to check for non-manifold triangles in shirt.obj
"""

import trimesh
import numpy as np

def check_nonmanifold_triangles(obj_path):
    """Check for non-manifold triangles in the mesh"""
    print(f"Loading mesh from: {obj_path}")
    
    try:
        # Load the mesh
        mesh = trimesh.load(obj_path)
        print(f"Loaded mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        
        # Check basic properties
        print(f"\nMesh Properties:")
        print(f"- Watertight: {mesh.is_watertight}")
        print(f"- Winding consistent: {mesh.is_winding_consistent}")
        print(f"- Has volume: {mesh.is_volume}")
        
        # Check for non-manifold issues using correct API
        print(f"\nNon-manifold Analysis:")
        
        # Get unique edges
        edges = mesh.edges_unique
        print(f"- Total unique edges: {len(edges)}")
        
        # Check for non-manifold edges (edges shared by more than 2 faces)
        edge_face_count = mesh.face_adjacency_edges
        nonmanifold_edge_count = 0
        
        # Count how many faces share each edge
        edge_to_faces = {}
        for face_idx, face in enumerate(mesh.faces):
            for i in range(3):
                edge = tuple(sorted([face[i], face[(i+1)%3]]))
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Find edges shared by more than 2 faces
        nonmanifold_edges = []
        for edge, face_list in edge_to_faces.items():
            if len(face_list) > 2:
                nonmanifold_edges.append(edge)
                nonmanifold_edge_count += 1
        
        print(f"- Non-manifold edges: {nonmanifold_edge_count}")
        
        # Check for non-manifold vertices
        vertex_face_count = {}
        for face in mesh.faces:
            for vertex in face:
                if vertex not in vertex_face_count:
                    vertex_face_count[vertex] = 0
                vertex_face_count[vertex] += 1
        
        nonmanifold_vertices = []
        for vertex, count in vertex_face_count.items():
            if count > 2:  # Vertex shared by more than 2 faces
                nonmanifold_vertices.append(vertex)
        
        print(f"- Non-manifold vertices: {len(nonmanifold_vertices)}")
        
        # Check for degenerate faces
        degenerate_faces = []
        for face_idx, face in enumerate(mesh.faces):
            # Check if face has duplicate vertices
            if len(set(face)) < 3:
                degenerate_faces.append(face_idx)
        
        print(f"- Degenerate faces: {len(degenerate_faces)}")
        
        # Check face areas
        face_areas = mesh.area_faces
        zero_area_faces = np.sum(face_areas < 1e-10)
        print(f"- Zero area faces: {zero_area_faces}")
        
        # Check for boundary edges
        boundary_edges = []
        for edge, face_list in edge_to_faces.items():
            if len(face_list) == 1:  # Edge only belongs to one face
                boundary_edges.append(edge)
        
        print(f"- Boundary edges: {len(boundary_edges)}")
        
        # Summary
        print(f"\nSummary:")
        if nonmanifold_edge_count > 0:
            print(f"❌ Found {nonmanifold_edge_count} non-manifold edges")
        else:
            print("✅ No non-manifold edges found")
            
        if len(nonmanifold_vertices) > 0:
            print(f"❌ Found {len(nonmanifold_vertices)} non-manifold vertices")
        else:
            print("✅ No non-manifold vertices found")
            
        if not mesh.is_watertight:
            print("❌ Mesh is not watertight")
        else:
            print("✅ Mesh is watertight")
            
        if len(degenerate_faces) > 0:
            print(f"❌ Found {len(degenerate_faces)} degenerate faces")
        else:
            print("✅ No degenerate faces found")
            
        if zero_area_faces > 0:
            print(f"❌ Found {zero_area_faces} zero area faces")
        else:
            print("✅ No zero area faces found")
        
        # Return results
        return {
            'nonmanifold_edges': nonmanifold_edge_count,
            'nonmanifold_vertices': len(nonmanifold_vertices),
            'degenerate_faces': len(degenerate_faces),
            'zero_area_faces': zero_area_faces,
            'boundary_edges': len(boundary_edges),
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'nonmanifold_edge_list': nonmanifold_edges,
            'nonmanifold_vertex_list': nonmanifold_vertices
        }
        
    except Exception as e:
        print(f"Error loading mesh: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Check shirt.obj
    obj_path = "shirt.obj"
    results = check_nonmanifold_triangles(obj_path)
    
    if results:
        print(f"\nResults saved for further processing")
        # You can use these results to decide if the mesh needs cleaning 