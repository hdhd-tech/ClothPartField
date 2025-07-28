import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import KDTree
from typing import List, Tuple, Set
from dataclasses import dataclass


@dataclass
class Segment:
    """Represents a boundary segment with its properties."""

    label: int
    start: int
    end: int
    length: int


def split_fuzzy(
    labels: List[int],
    min_len: int = 3,
    fuzzy_labels: Tuple[int, ...] = (-1,),
    circular: bool = True,
) -> List[Tuple[int, int, int]]:
    """Split labels into segments, cleaning up fuzzy/short segments."""
    labels = np.asarray(labels).copy()
    n = len(labels)

    # 1. RLE with circular wrap
    change = labels != np.roll(labels, 1)
    starts = np.flatnonzero(change)
    if starts.size == 0:
        return [(labels[0], 0, n - 1)]

    starts = np.sort(starts)
    ends = np.roll(starts - 1, -1)
    ends[-1] = (starts[0] - 1) % n
    seg_labels = labels[starts]

    # 2. Fix fuzzy runs
    for i, (lab, st, en) in enumerate(zip(seg_labels, starts, ends)):
        length = (en - st + 1) % n or n
        if length >= min_len or lab not in fuzzy_labels:
            continue

        left_i = (i - 1) % len(starts)
        right_i = (i + 1) % len(starts)
        left_lab = seg_labels[left_i]
        right_lab = seg_labels[right_i]

        idxs = np.arange(st, st + length) % n
        mid = len(idxs) // 2
        labels[idxs[:mid]] = left_lab
        labels[idxs[mid:]] = right_lab

    # 3. Re-RLE to output segments
    change = labels != np.roll(labels, 1)
    starts = np.flatnonzero(change)
    starts = np.sort(starts)
    ends = np.roll(starts - 1, -1)
    ends[-1] = (starts[0] - 1) % n
    seg_labels = labels[starts]

    # merge first/last if same label (circular)
    segs = [
        (lab, st, en)
        for lab, st, en in zip(seg_labels.tolist(), starts.tolist(), ends.tolist())
    ]
    if circular and len(segs) > 1 and segs[0][0] == segs[-1][0]:
        lab, st1, en1 = segs[-1]
        _, st0, en0 = segs[0]
        segs[0] = (lab, st1, en0)
        segs.pop()

    return segs


def get_segment_edges(
    panel_id: int, segment: Segment, boundaries: List, kdtree: KDTree
) -> Set[Tuple[int, int]]:
    """Get all global edges for a segment."""
    bv, be, cc = boundaries[panel_id]
    segment_edges = set()

    for i in range(segment.length):
        edge_idx = (segment.start + i) % len(be)
        edge = be[edge_idx]
        v0_local, v1_local = edge
        v0_global = kdtree.query(cc.vertices[v0_local])[1]
        v1_global = kdtree.query(cc.vertices[v1_local])[1]
        edge_key = (min(v0_global, v1_global), max(v0_global, v1_global))
        segment_edges.add(edge_key)

    return segment_edges


def find_best_match(
    panel_id: int,
    segment: Segment,
    all_segments: List[List[Segment]],
    boundaries: List,
    kdtree: KDTree,
    used_segments: Set,
) -> Tuple[int, int, int]:
    """Find the best matching segment for a given segment."""
    segment_edges = get_segment_edges(panel_id, segment, boundaries, kdtree)
    best_match = None
    best_overlap = 0

    for other_panel_id, other_segments in enumerate(all_segments):
        if other_panel_id <= panel_id:  # Avoid duplicates and self
            continue

        for other_seg_idx, other_segment in enumerate(other_segments):
            if (other_panel_id, other_seg_idx) in used_segments:
                continue

            other_segment_edges = get_segment_edges(
                other_panel_id, other_segment, boundaries, kdtree
            )
            shared_edges = segment_edges.intersection(other_segment_edges)
            overlap = len(shared_edges)

            # Require high overlap (80% of shorter segment)
            min_required = min(segment.length, other_segment.length) * 0.8
            if overlap >= min_required:
                length_ratio = max(segment.length, other_segment.length) / min(
                    segment.length, other_segment.length
                )
                if length_ratio <= 1.5:  # Similar lengths
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = (other_panel_id, other_seg_idx, overlap)

    return best_match


def find_sewing_pairs(
    edge_labels: List[List[int]], boundaries: List, kdtree: KDTree, min_len: int = 3
) -> List[Tuple]:
    """Find sewing pairs based on shared edges."""
    # Split and filter segments
    all_segments = []
    for panel_id, labels in enumerate(edge_labels):
        segments = split_fuzzy(
            labels, min_len=min_len, fuzzy_labels=(-1,), circular=True
        )
        filtered_segments = []

        for label, start, end in segments:
            bv, be, cc = boundaries[panel_id]
            seg_length = (end - start + 1) % len(be) or len(be)
            if seg_length >= min_len and label != -1:
                filtered_segments.append(Segment(label, start, end, seg_length))

        all_segments.append(filtered_segments)

    print("Finding sewing pairs based on shared edges...")
    sewing_pairs = []
    used_segments = set()

    for panel_id, segments in enumerate(all_segments):
        for seg_idx, segment in enumerate(segments):
            if (panel_id, seg_idx) in used_segments:
                continue

            best_match = find_best_match(
                panel_id, segment, all_segments, boundaries, kdtree, used_segments
            )

            if best_match:
                other_panel_id, other_seg_idx, overlap = best_match
                other_segment = all_segments[other_panel_id][other_seg_idx]

                print(
                    f"PAIR: Panel {panel_id} seg(label={segment.label}, len={segment.length}) <-> "
                    f"Panel {other_panel_id} seg(label={other_segment.label}, len={other_segment.length}), "
                    f"{overlap} shared edges"
                )

                sewing_pair = (
                    (panel_id, segment.start, segment.end, False),
                    (other_panel_id, other_segment.start, other_segment.end, False),
                )
                sewing_pairs.append(sewing_pair)
                used_segments.add((panel_id, seg_idx))
                used_segments.add((other_panel_id, other_seg_idx))

    print(f"Found {len(sewing_pairs)} sewing pairs")
    return sewing_pairs


def plot_panel_with_segment(
    ax, panel_data: Tuple, segment_data: Tuple, pair_idx: int, is_reversed: bool = False
):
    """Plot a single panel with highlighted segment."""
    bv, be, cc = panel_data
    panel_id, start, end, reverse = segment_data
    bv = np.array(bv)

    # Plot full mesh in light gray
    ax.scatter(
        cc.vertices[:, 0],
        cc.vertices[:, 1],
        cc.vertices[:, 2],
        alpha=0.1,
        color="lightgray",
        s=1,
    )

    # Plot full boundary in blue
    boundary_coords = cc.vertices[bv]
    ax.plot(
        boundary_coords[:, 0],
        boundary_coords[:, 1],
        boundary_coords[:, 2],
        "b-",
        alpha=0.3,
        linewidth=1,
        label="Full boundary",
    )

    # Highlight specific segment in red
    seg_length = (end - start + 1) % len(bv) or len(bv)
    seg_indices = [(start + i) % len(bv) for i in range(seg_length)]
    if is_reversed:
        seg_indices = seg_indices[::-1]

    seg_coords = cc.vertices[bv[seg_indices]]
    ax.plot(
        seg_coords[:, 0],
        seg_coords[:, 1],
        seg_coords[:, 2],
        "r-",
        linewidth=3,
        label=f"Segment {start}-{end}" + (" (reversed)" if is_reversed else ""),
    )
    ax.scatter(seg_coords[:, 0], seg_coords[:, 1], seg_coords[:, 2], color="red", s=20)

    ax.set_title(f"Panel {panel_id} - Segment ({start}, {end})")
    ax.legend()
    ax.set_aspect("equal")


def plot_sewing_pairs(
    sewing_pairs: List, boundaries: List, save_dir: str = "test_data"
):
    """Plot each sewing pair as a side-by-side comparison."""
    for pair_idx, (seg1_data, seg2_data) in enumerate(sewing_pairs):
        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(16, 8), subplot_kw={"projection": "3d"}
        )

        panel1_id, start1, end1, reverse1 = seg1_data
        panel2_id, start2, end2, reverse2 = seg2_data

        plot_panel_with_segment(
            ax1, boundaries[panel1_id], seg1_data, pair_idx, reverse1
        )
        plot_panel_with_segment(
            ax2, boundaries[panel2_id], seg2_data, pair_idx, reverse2
        )

        plt.tight_layout()
        plt.savefig(
            f"{save_dir}/sewing_pair_{pair_idx}.png", dpi=150, bbox_inches="tight"
        )
        plt.close()


def plot_all_segments_overview(
    sewing_pairs: List, boundaries: List, save_dir: str = "test_data"
):
    """Plot all sewing pairs in a single overview."""
    fig = plt.figure(figsize=(20, 15))
    num_panels = len(boundaries)
    cols = min(3, num_panels)
    rows = (num_panels + cols - 1) // cols
    colors = plt.cm.tab10(range(len(sewing_pairs)))

    for panel_id, (bv, be, cc) in enumerate(boundaries):
        bv = np.array(bv)
        ax = fig.add_subplot(rows, cols, panel_id + 1, projection="3d")

        # Plot full mesh and boundary
        ax.scatter(
            cc.vertices[:, 0],
            cc.vertices[:, 1],
            cc.vertices[:, 2],
            alpha=0.05,
            color="lightgray",
            s=0.5,
        )
        boundary_coords = cc.vertices[bv]
        ax.plot(
            boundary_coords[:, 0],
            boundary_coords[:, 1],
            boundary_coords[:, 2],
            "k-",
            alpha=0.3,
            linewidth=1,
        )

        # Plot segments that are part of sewing pairs
        for pair_idx, (
            (p1_id, start1, end1, rev1),
            (p2_id, start2, end2, rev2),
        ) in enumerate(sewing_pairs):
            if p1_id == panel_id:
                seg_length = (end1 - start1 + 1) % len(bv) or len(bv)
                seg_indices = [(start1 + i) % len(bv) for i in range(seg_length)]
                seg_coords = cc.vertices[bv[seg_indices]]
                ax.plot(
                    seg_coords[:, 0],
                    seg_coords[:, 1],
                    seg_coords[:, 2],
                    color=colors[pair_idx],
                    linewidth=3,
                    label=f"Pair {pair_idx} -> Panel {p2_id}",
                )
            elif p2_id == panel_id:
                seg_length = (end2 - start2 + 1) % len(bv) or len(bv)
                seg_indices = [(start2 + i) % len(bv) for i in range(seg_length)]
                if rev2:
                    seg_indices = seg_indices[::-1]
                seg_coords = cc.vertices[bv[seg_indices]]
                ax.plot(
                    seg_coords[:, 0],
                    seg_coords[:, 1],
                    seg_coords[:, 2],
                    color=colors[pair_idx],
                    linewidth=3,
                    label=f"Pair {pair_idx} -> Panel {p1_id}",
                )

        ax.set_title(f"Panel {panel_id}")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(f"{save_dir}/sewing_pairs_overview.png", dpi=150, bbox_inches="tight")
    plt.close()
