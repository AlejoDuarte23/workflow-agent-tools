import viktor as vkt
import pandas as pd
import plotly.graph_objects as go


class Parametrization(vkt.Parametrization):
    intro = vkt.Text("""
# ETABS reaction heatmap
Enter your node coordinates and reactions manually in the tables below, then visualize the results.
""")

    # Table 1: Node Coordinates
    node_coords = vkt.Table(
        "Node Coordinates",
        default=[
            {"node_name": "N1", "x": 0.0, "y": 0.0, "z": 0.0},
            {"node_name": "N2", "x": 5000.0, "y": 0.0, "z": 0.0},
            {"node_name": "N3", "x": 10000.0, "y": 0.0, "z": 0.0},
            {"node_name": "N4", "x": 0.0, "y": 5000.0, "z": 0.0},
        ],
    )
    node_coords.node_name = vkt.TextField("Node Name")
    node_coords.x = vkt.NumberField("X (mm)", num_decimals=2)
    node_coords.y = vkt.NumberField("Y (mm)", num_decimals=2)
    node_coords.z = vkt.NumberField("Z (mm)", num_decimals=2)

    lb1 = vkt.LineBreak()

    # Table 2: Node Reactions & Loads
    node_reactions = vkt.Table(
        "Node Reactions & Loads",
        default=[
            {
                "node_name": "N1",
                "fx": 0.0,
                "fy": 0.0,
                "fz": -150.0,
                "mx": 0.0,
                "my": 0.0,
                "mz": 0.0,
            },
            {
                "node_name": "N2",
                "fx": 0.0,
                "fy": 0.0,
                "fz": -200.0,
                "mx": 0.0,
                "my": 0.0,
                "mz": 0.0,
            },
            {
                "node_name": "N3",
                "fx": 0.0,
                "fy": 0.0,
                "fz": -175.0,
                "mx": 0.0,
                "my": 0.0,
                "mz": 0.0,
            },
            {
                "node_name": "N4",
                "fx": 0.0,
                "fy": 0.0,
                "fz": -180.0,
                "mx": 0.0,
                "my": 0.0,
                "mz": 0.0,
            },
        ],
    )
    node_reactions.node_name = vkt.TextField("Node Name")
    node_reactions.fx = vkt.NumberField("FX (kN)", num_decimals=2)
    node_reactions.fy = vkt.NumberField("FY (kN)", num_decimals=2)
    node_reactions.fz = vkt.NumberField("FZ (kN)", num_decimals=2)
    node_reactions.mx = vkt.NumberField("MX (kN·m)", num_decimals=2)
    node_reactions.my = vkt.NumberField("MY (kN·m)", num_decimals=2)
    node_reactions.mz = vkt.NumberField("MZ (kN·m)", num_decimals=2)


class Controller(vkt.Controller):
    parametrization = Parametrization

    @vkt.PlotlyView("Heatmap")
    def plot_heat_map(self, params, **kwargs):
        """Create heatmap visualization from input tables"""
        # Convert input tables to dataframes
        coords_df = pd.DataFrame(params.node_coords)
        reactions_df = pd.DataFrame(params.node_reactions)

        # Check if we have data
        if coords_df.empty or reactions_df.empty:
            return vkt.PlotlyResult({})

        # Merge coordinates with reactions based on node name
        merged_df = pd.merge(reactions_df, coords_df, on="node_name", how="inner")

        if merged_df.empty:
            return vkt.PlotlyResult({})

        # Get min/max for color scale
        FZ_min, FZ_max = merged_df["fz"].min(), merged_df["fz"].max()

        # Create plotly scatter plot
        fig = go.Figure(
            data=go.Scatter(
                x=merged_df["x"],
                y=merged_df["y"],
                mode="markers+text",
                marker=dict(
                    size=16,
                    color=merged_df["fz"],
                    colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]],
                    colorbar=dict(title=dict(text="FZ (kN)")),
                    cmin=FZ_min,
                    cmax=FZ_max,
                ),
                text=[
                    f"{node}<br>{fz:.1f}"
                    for node, fz in zip(merged_df["node_name"], merged_df["fz"])
                ],
                textposition="top center",
            )
        )

        # Style the plot
        fig.update_layout(
            title="Node Reaction Heatmap",
            xaxis=dict(title=dict(text="X (mm)")),
            yaxis=dict(title=dict(text="Y (mm)")),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        fig.update_xaxes(linecolor="LightGrey")
        fig.update_yaxes(linecolor="LightGrey")

        return vkt.PlotlyResult(fig)
