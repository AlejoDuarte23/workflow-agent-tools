import math
from typing import Annotated
from collections import defaultdict

from app.types import NodeDict, LineDict


class Component:
    def __init__(self):
        self.nodes: dict[int, NodeDict] = {}
        self.lines: dict[int, LineDict] = {}
        self.tag: int = 0
        self.line_tag: int = 0

    def create_node_tag(self) -> int:
        self.tag += 1
        return self.tag

    def create_line_tag(self) -> int:
        self.line_tag += 1
        return self.line_tag


class RectangularTrussBeam(Component):
    def __init__(
        self,
        length: Annotated[float, "Length of the beam along the X-axis (m)"],
        width: Annotated[float, "Width of the beam along the Y-axis (m)"],
        height: Annotated[float, "Height of the beam along the Z-axis (m)"],
        n_diagonals: Annotated[int, "Number of diagonal sections along the beam"],
    ):
        """Create a rectangular truss beam starting at origin (0, 0, 0)."""
        super().__init__()
        self.length = length
        self.width = width
        self.height = height
        self.n_diagonals = n_diagonals

    def create_chord_nodes(self, yo: float, zo: float) -> dict[int, NodeDict]:
        """Create nodes along a chord (longitudinal edge) of the beam."""
        n_nodes = self.n_diagonals * 2 + 1
        dx = self.length / (n_nodes - 1)
        chord_nodes = {}
        for i in range(n_nodes):
            node = {
                "x": i * dx,
                "y": yo,
                "z": zo,
            }
            tag = self.create_node_tag()
            chord_nodes[tag] = node
        self.nodes.update(chord_nodes)
        return chord_nodes

    def create_chord_lines(self, node_dict: dict[int, NodeDict]) -> dict[int, LineDict]:
        """Create lines connecting consecutive nodes in a chord."""
        node_ids = list(node_dict.keys())
        chord_lines = {}
        for i in range(len(node_ids) - 1):
            chord_lines[self.create_line_tag()] = {
                "NodeI": node_ids[i],
                "NodeJ": node_ids[i + 1],
            }
        self.lines.update(chord_lines)
        return chord_lines

    def create_diagonals(self, chord_a_ids: list[int], chord_b_ids: list[int]) -> None:
        """Create diagonal bracing between two parallel chords."""
        if self.n_diagonals % 2 == 0:
            # Even number of diagonals
            top_indices = chord_a_ids[::2]
            bottom_indices = chord_b_ids[1:-1:2]
            tags_a = sorted([top_indices[0]] + [top_indices[-1]] + top_indices[1:-1] * 2)
            tags_b = sorted(bottom_indices[:] * 2)
            for tag_a, tag_b in zip(tags_a, tags_b):
                self.lines[self.create_line_tag()] = {
                    "NodeI": tag_a,
                    "NodeJ": tag_b,
                }
        else:
            # Odd number of diagonals
            top_indices = chord_a_ids[:-1:2]
            bottom_indices = chord_b_ids[1::2]
            tags_a = sorted([top_indices[0]] + top_indices[1:] * 2)
            tags_b = sorted([bottom_indices[-1]] + bottom_indices[:-1] * 2)
            for tag_a, tag_b in zip(tags_a, tags_b):
                self.lines[self.create_line_tag()] = {
                    "NodeI": tag_a,
                    "NodeJ": tag_b,
                }

    def create_vertical_bracing(self, chord_a_ids: list[int], chord_b_ids: list[int]) -> None:
        """Create vertical bracing (struts) between two chords at even indices."""
        for i, (tag_a, tag_b) in enumerate(zip(chord_a_ids, chord_b_ids)):
            #if i % 2 == 0:
            self.lines[self.create_line_tag()] = {
                "NodeI": tag_a,
                "NodeJ": tag_b,
            }

    def build(self) -> tuple[
        Annotated[dict[int, NodeDict], "Dictionary of node IDs to node coordinates"],
        Annotated[dict[int, LineDict], "Dictionary of line IDs to line connectivity"],
        Annotated[list[int], "List of top-left chord node IDs"],
        Annotated[list[int], "List of top-right chord node IDs"],
    ]:
        """Build the rectangular truss beam geometry."""
        # Create four chords at corners of the rectangular cross-section
        # Bottom-left chord (y=0, z=0)
        chord_bl = self.create_chord_nodes(yo=0, zo=0)
        # Bottom-right chord (y=width, z=0)
        chord_br = self.create_chord_nodes(yo=self.width, zo=0)
        # Top-left chord (y=0, z=height)
        chord_tl = self.create_chord_nodes(yo=0, zo=self.height)
        # Top-right chord (y=width, z=height)
        chord_tr = self.create_chord_nodes(yo=self.width, zo=self.height)

        # Get node IDs for each chord
        chord_bl_ids = list(chord_bl.keys())
        chord_br_ids = list(chord_br.keys())
        chord_tl_ids = list(chord_tl.keys())
        chord_tr_ids = list(chord_tr.keys())

        # Create longitudinal lines along each chord
        self.create_chord_lines(chord_bl)
        self.create_chord_lines(chord_br)
        self.create_chord_lines(chord_tl)
        self.create_chord_lines(chord_tr)

        # Create diagonal bracing on each face
        # Left face (z=0): between bottom-left and top-left
        self.create_diagonals(chord_bl_ids, chord_tl_ids)
        # Right face (z=width): between bottom-right and top-right
        self.create_diagonals(chord_br_ids, chord_tr_ids)
        # Bottom face (y=0): between bottom-left and bottom-right
        self.create_diagonals(chord_bl_ids, chord_br_ids)
        # Top face (y=height): between top-left and top-right
        self.create_diagonals(chord_tr_ids, chord_tl_ids)

        # Create vertical bracing (struts) connecting the faces
        # Left face verticals
        self.create_vertical_bracing(chord_bl_ids, chord_tl_ids)
        # Right face verticals
        self.create_vertical_bracing(chord_br_ids, chord_tr_ids)
        # Bottom face horizontals
        self.create_vertical_bracing(chord_bl_ids, chord_br_ids)
        # Top face horizontals
        self.create_vertical_bracing(chord_tl_ids, chord_tr_ids)

        return self.nodes, self.lines, chord_tl_ids, chord_tr_ids

    def clean_model(self) -> tuple[
        Annotated[dict[int, NodeDict], "Dictionary of deduplicated node IDs to coordinates"],
        Annotated[dict[int, LineDict], "Dictionary of valid line IDs to connectivity"],
    ]:
        """Remove duplicate nodes and zero-length lines."""
        # Create mapping from coordinates to node IDs
        coord_to_nodes = defaultdict(list)
        for node_id, attrs in self.nodes.items():
            coord = (attrs["x"], attrs["y"], attrs["z"])
            coord_to_nodes[coord].append(node_id)

        # Find duplicates
        node_replacements = {}
        for coord, ids in coord_to_nodes.items():
            if len(ids) > 1:
                kept_node = min(ids)
                for dup_node in ids:
                    if dup_node != kept_node:
                        node_replacements[dup_node] = kept_node

        # Update lines to use kept nodes
        for line in self.lines.values():
            if line["NodeI"] in node_replacements:
                line["NodeI"] = node_replacements[line["NodeI"]]
            if line["NodeJ"] in node_replacements:
                line["NodeJ"] = node_replacements[line["NodeJ"]]

        # Remove duplicate nodes
        for dup_node in node_replacements.keys():
            if dup_node in self.nodes:
                del self.nodes[dup_node]

        # Remove duplicate and zero-length lines
        unique_lines = {}
        seen_lines = set()
        for line_tag, line in self.lines.items():
            canonical = tuple(sorted([line["NodeI"], line["NodeJ"]]))
            if canonical in seen_lines:
                continue
            if line["NodeI"] not in self.nodes or line["NodeJ"] not in self.nodes:
                continue
            node1 = self.nodes[line["NodeI"]]
            node2 = self.nodes[line["NodeJ"]]
            dx = node1["x"] - node2["x"]
            dy = node1["y"] - node2["y"]
            dz = node1["z"] - node2["z"]
            length = math.sqrt(dx * dx + dy * dy + dz * dz)
            if length > 0:
                seen_lines.add(canonical)
                unique_lines[line_tag] = line

        self.lines = unique_lines
        return self.nodes, self.lines

    def remove_top_edge_nodes(
        self,
        chord_tl_ids: list[int],
        chord_tr_ids: list[int],
    ) -> tuple[dict[int, NodeDict], dict[int, LineDict]]:
        """
        Remove the first and last nodes of the top chords and their connected lines.
        Then add end "stitch" members so the remaining top-end nodes stay connected
        for both odd and even n_diagonals.
        """
        if len(chord_tl_ids) < 3 or len(chord_tr_ids) < 3:
            raise ValueError("Need at least 3 nodes per top chord to remove end nodes safely.")

        # 4 corner top nodes to remove
        nodes_to_remove = {
            chord_tl_ids[0],
            chord_tl_ids[-1],
            chord_tr_ids[0],
            chord_tr_ids[-1],
        }

        # Remove lines connected to removed nodes
        lines_to_remove: list[int] = []
        for line_tag, line in self.lines.items():
            if line["NodeI"] in nodes_to_remove or line["NodeJ"] in nodes_to_remove:
                lines_to_remove.append(line_tag)
        for lt in lines_to_remove:
            del self.lines[lt]

        # Remove the nodes
        for nid in nodes_to_remove:
            self.nodes.pop(nid, None)

        # Remaining end nodes on top chords (new ends after deletion)
        tl_left_keep = chord_tl_ids[1]
        tl_right_keep = chord_tl_ids[-2]
        tr_left_keep = chord_tr_ids[1]
        tr_right_keep = chord_tr_ids[-2]

        eps = 1e-9

        def find_node_id(x: float, y: float, z: float) -> int | None:
            for node_id, n in self.nodes.items():
                if (
                    abs(n["x"] - x) <= eps
                    and abs(n["y"] - y) <= eps
                    and abs(n["z"] - z) <= eps
                ):
                    return node_id
            return None

        def has_member(a: int, b: int) -> bool:
            aa, bb = (a, b) if a < b else (b, a)
            for ln in self.lines.values():
                i, j = ln["NodeI"], ln["NodeJ"]
                ii, jj = (i, j) if i < j else (j, i)
                if ii == aa and jj == bb:
                    return True
            return False

        def add_member(a: int | None, b: int | None) -> None:
            if a is None or b is None or a == b:
                return
            if not has_member(a, b):
                self.lines[self.create_line_tag()] = {"NodeI": a, "NodeJ": b}

        # Bottom end nodes by geometry
        bl_left = find_node_id(0.0, 0.0, 0.0)
        br_left = find_node_id(0.0, self.width, 0.0)
        bl_right = find_node_id(self.length, 0.0, 0.0)
        br_right = find_node_id(self.length, self.width, 0.0)

        # Stitch diagonals on the side faces (y=0 and y=width)
        add_member(bl_left, tl_left_keep)
        add_member(bl_right, tl_right_keep)
        add_member(br_left, tr_left_keep)
        add_member(br_right, tr_right_keep)

        return self.nodes, self.lines


if __name__ == "__main__":
    # Example usage
    beam = RectangularTrussBeam(
        length=10.0,
        width=1.0,
        height=1.5,
        n_diagonals=6,
    )
    nodes, lines, chord_tl_ids, chord_tr_ids = beam.build()
    nodes, lines = beam.clean_model()
    nodes, lines = beam.remove_top_edge_nodes(chord_tl_ids, chord_tr_ids)

    print(f"Number of nodes: {len(nodes)}")
    print(f"Number of lines: {len(lines)}")
    print("\nNodes:")
    for tag, node in nodes.items():
        print(f"  {tag}: x={node['x']:.2f}, y={node['y']:.2f}, z={node['z']:.2f}")
    print("\nLines:")
    for tag, line in lines.items():
        print(f"  {tag}: NodeI={line['NodeI']}, NodeJ={line['NodeJ']}")
