from dataclasses import dataclass
from typing import List, Tuple
import trimesh


@dataclass
class Segment:
    """Represents a boundary segment with its properties."""

    label: int
    start: int
    end: int
    length: int


@dataclass
class ColorPanel:
    """Represents a single colored panel with its boundary information."""

    color_id: int
    boundary_vertices: List[int]
    boundary_edges: List[Tuple[int, int]]
    mesh: trimesh.Trimesh
