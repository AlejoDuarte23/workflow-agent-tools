import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from app.types import MemberType
Vec3 = np.ndarray

def compute_beam_vertices_rect(A: Vec3, B: Vec3, width: float, height: float) -> np.ndarray:
    """Compute the 8 vertices of a rectangular beam between points A and B."""
    v = B - A
    length = np.linalg.norm(v)
    if length == 0:
        return np.array([A] * 8)
    v_hat = v / length

    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    helper = min(axes, key=lambda ax: abs(np.dot(v_hat, ax)))

    local_y = np.cross(v_hat, helper)
    local_y /= np.linalg.norm(local_y)
    local_z = np.cross(v_hat, local_y)
    local_z /= np.linalg.norm(local_z)

    local_y *= width / 2.0
    local_z *= height / 2.0

    v0, v1, v2, v3 = A + local_y + local_z, A + local_y - local_z, A - local_y - local_z, A - local_y + local_z
    v4, v5, v6, v7 = B + local_y + local_z, B + local_y - local_z, B - local_y - local_z, B - local_y + local_z
    return np.stack([v0, v1, v2, v3, v4, v5, v6, v7])


def add_beam_mesh(fig: go.Figure, verts: np.ndarray, color: str) -> None:
    """Add a beam mesh to a Plotly figure."""
    quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
    i_list, j_list, k_list = [], [], []
    for a, b, c, d in quads:
        i_list.extend([a, a])
        j_list.extend([b, c])
        k_list.extend([c, d])  # Draw one side
        i_list.extend([a, a])
        j_list.extend([c, d])
        k_list.extend([b, c])  # Draw the other side for visibility

    fig.add_trace(
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=i_list,
            j=j_list,
            k=k_list,
            color=color,
            flatshading=False,
            opacity=1.0,
            hoverinfo="skip",
            lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3, roughness=0.9),
            showscale=False,
        )
    )


def normalise(val: float, vmin: float, vmax: float) -> float:
    """Normalize a value between vmin and vmax to [0, 1]."""
    if vmax == vmin:
        return 0.5
    return (val - vmin) / (vmax - vmin)


def jet_colour(val: float, vmin: float, vmax: float, colorscale: list) -> str:
    """Gets a color from a provided Plotly sequential colorscale list."""
    idx = int(normalise(val, vmin, vmax) * (len(colorscale) - 1))
    return colorscale[idx]


def plot_deformed_mesh(
    nodes: dict[int, dict],
    lines: dict[int, dict],
    members: dict[int, dict],
    cross_sections: dict[int, dict],
    disp_dict: dict[int, dict],
    scale: float = 25,
    critical_combination_name: str | None = None,
) -> go.Figure:
    """Return a Plotly figure of the scaled deformed shape with meshed elements.
    
    Args:
        nodes: Dictionary of node data with coordinates.
        lines: Dictionary of line data with node connectivity.
        members: Dictionary of member data with cross-section assignments.
        cross_sections: Dictionary of cross-section properties.
        disp_dict: Dictionary mapping node IDs to displacement dict with dx, dy, dz.
        scale: Scale factor for deformation visualization.
        critical_combination_name: Name of the critical load combination (optional).
    """
    
    # Helper to get displacement component with default
    def get_disp(nid: int, component: str) -> float:
        node_disp = disp_dict.get(nid)
        if node_disp is None:
            return 0.0
        return node_disp.get(component, 0.0)

    # ------------------------------------------------------------------ #
    # 1. Deformed node coordinates (apply all 3 displacement components)
    # ------------------------------------------------------------------ #
    def_nodes: dict[int, dict] = {}
    for nid, data in nodes.items():
        def_nodes[nid] = {
            "x": data["x"] + get_disp(nid, "dx") * scale,
            "y": data["y"] + get_disp(nid, "dy") * scale,
            "z": data["z"] + get_disp(nid, "dz") * scale,
        }

    # mean z-displacement per line (for coloring)
    line_disp = {
        lid: (get_disp(ln["Ni"], "dz") + get_disp(ln["Nj"], "dz")) / 2.0
        for lid, ln in lines.items()
    }
    dmin, dmax = (min(line_disp.values()), max(line_disp.values())) if line_disp else (0.0, 0.0)

    # ------------------------------------------------------------------ #
    # 2. Figure with bounding cube
    # ------------------------------------------------------------------ #
    fig = go.Figure()

    all_x, all_y, all_z = (
        [n["x"] for n in def_nodes.values()],
        [n["y"] for n in def_nodes.values()],
        [n["z"] for n in def_nodes.values()],
    )
    if all_x:  # keep aspect 1:1:1
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)
        z_min, z_max = min(all_z), max(all_z)
        cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2
        half = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2 or 1.0
        fig.add_trace(
            go.Scatter3d(
                x=[cx - half, cx + half] * 4,
                y=[cy - half, cy - half, cy + half, cy + half] * 2,
                z=[cz - half] * 4 + [cz + half] * 4,
                mode="markers",
                marker=dict(size=0, color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # colour helper
    scale_name = "Jet_r"
    color_scale = getattr(px.colors.sequential, scale_name)

    def map_colour(val: float) -> str:
        if dmax == dmin:
            return color_scale[0]
        ratio = (val - dmin) / (dmax - dmin)
        return color_scale[int(ratio * (len(color_scale) - 1))]

    # draw members (assumes helper funcs exist)
    for lid, ln in lines.items():
        cs_id = members[lid]["cross_section_id"]
        if cs_id not in cross_sections:
            continue
        n1, n2 = def_nodes[ln["Ni"]], def_nodes[ln["Nj"]]
        A = np.array([n1["x"], n1["y"], n1["z"]])
        B = np.array([n2["x"], n2["y"], n2["z"]])
        cs = cross_sections[cs_id]
        verts = compute_beam_vertices_rect(A, B, width=float(cs["h"]), height=float(cs["b"]))
        add_beam_mesh(fig, verts, map_colour(line_disp[lid]))

    # nodes
    fig.add_trace(
        go.Scatter3d(
            x=[n["x"] for n in def_nodes.values()],
            y=[n["y"] for n in def_nodes.values()],
            z=[n["z"] for n in def_nodes.values()],
            mode="markers",
            marker=dict(size=3, color="black"),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # colour-bar
    fig.add_trace(
        go.Scatter3d(
            x=[None],
            y=[None],
            z=[None],
            mode="markers",
            marker=dict(
                colorscale=scale_name,
                cmin=dmin,
                cmax=dmax,
                color=[dmin, dmax],
                showscale=True,
                colorbar=dict(
                    title="ΔZ [mm]",
                    # -- size ----------------------------------------------------------------
                    len=0.45,  # 45 % of plot height  (default: 1.0 = full height)
                    lenmode="fraction",  # "fraction" = percentage, "pixels" = absolute px
                    thickness=25,  # 25 px wide           (default: 30 px)
                    thicknessmode="pixels",
                    # -- position ------------------------------------------------------------
                    y=0.5,
                    yanchor="middle",
                    x=1.02,
                    xanchor="left",
                    # -- appearance (optional) ----------------------------------------------
                    outlinewidth=1,
                    ticks="outside",
                    tickfont=dict(size=12),
                ),
            ),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # ------------------------------------------------------------------ #
    # 3. Max-displacement box with critical combination info
    # ------------------------------------------------------------------ #
    # Calculate total resultant deformation for each node
    max_resultant = 0.0
    for nid in nodes.keys():
        dx = get_disp(nid, "dx")
        dy = get_disp(nid, "dy")
        dz = get_disp(nid, "dz")
        resultant = np.sqrt(dx**2 + dy**2 + dz**2)
        if resultant > max_resultant:
            max_resultant = resultant
    
    # Build annotation text with critical combination if provided
    if critical_combination_name:
        annotation_text = (
            f"<b>Critical Combination: {critical_combination_name}</b><br>"
            f"Max Resultant Deformation: {max_resultant:.3f} mm"
        )
    else:
        annotation_text = f"<b>Max Resultant Deformation</b><br>{max_resultant:.3f} mm"
    
    fig.add_annotation(
        text=annotation_text,
        xref="paper",
        yref="paper",
        x=0.5,
        y=1,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.85)",
        borderwidth=1,
        font=dict(size=16, color="black"),
    )

    # ------------------------------------------------------------------ #
    # 4. Layout
    # ------------------------------------------------------------------ #
    fig.update_layout(
        scene=dict(
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            aspectmode="data",
            bgcolor="white",
            camera=dict(eye=dict(x=1.3, y=1.3, z=1.3)),
        ),
        showlegend=False,
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig
