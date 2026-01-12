import plotly.graph_objects as go
import numpy as np

from app.types import NodesInfoDict, LinesInfoDict, MembersDict, CrossSectionsDict


def plot_model_3d(
    nodes: NodesInfoDict,
    lines: LinesInfoDict,
    members: MembersDict,
    cross_sections: CrossSectionsDict,
) -> go.Figure:
    """Plot 3D model of truss beam with blue lines and green spheres for nodes."""
    x_nodes = [n["x"] for n in nodes.values()]
    y_nodes = [n["y"] for n in nodes.values()]
    z_nodes = [n["z"] for n in nodes.values()]

    fig = go.Figure()

    # This is a trick to stabilize the scene by adding an INVISIBLE bounding box
    x0, x1 = min(x_nodes), max(x_nodes)
    y0, y1 = min(y_nodes), max(y_nodes)
    z0, z1 = min(z_nodes), max(z_nodes)

    # Find the center point of the model
    x_center, y_center, z_center = (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2
    # Find the largest dimension of the model
    max_range = max(x1 - x0, y1 - y0, z1 - z0)
    half_range = max_range / 2.0

    # Define the 8 corners of a perfect cube centered around the model
    xb = [x_center - half_range, x_center + half_range]
    yb = [y_center - half_range, y_center + half_range]
    zb = [z_center - half_range, z_center + half_range]

    fig.add_trace(go.Scatter3d(
        x=[xb[0], xb[1], xb[0], xb[1], xb[0], xb[1], xb[0], xb[1]],
        y=[yb[0], yb[0], yb[1], yb[1], yb[0], yb[0], yb[1], yb[1]],
        z=[zb[0], zb[0], zb[0], zb[0], zb[1], zb[1], zb[1], zb[1]],
        mode='markers',
        marker=dict(size=0, color='rgba(0,0,0,0)'),  # Make markers invisible
        showlegend=False,
        hoverinfo='none'
    ))

    # Draw green spheres for nodes
    fig.add_trace(
        go.Scatter3d(
            x=x_nodes, y=y_nodes, z=z_nodes, mode="markers",
            marker=dict(size=6, color="green"), hoverinfo="text", showlegend=False,
        )
    )

    # Draw blue lines for members
    for member in members.values():
        line = lines[member["line_id"]]
        ni, nj = nodes[line["Ni"]], nodes[line["Nj"]]
        fig.add_trace(
            go.Scatter3d(
                x=[ni["x"], nj["x"]],
                y=[ni["y"], nj["y"]],
                z=[ni["z"], nj["z"]],
                mode="lines",
                line=dict(color="blue", width=4),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_visible=False,
            yaxis_visible=False,
            zaxis_visible=False,
            bgcolor="white",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig
