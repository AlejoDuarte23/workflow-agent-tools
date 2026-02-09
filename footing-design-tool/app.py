import json
import plotly.graph_objects as go
import viktor as vkt
from footing_calculations import (
    create_bearing_capacity_interpolator,
    find_optimal_footing_design,
    get_top_n_designs,
)


class Parametrization(vkt.Parametrization):
    # Introduction Section
    section_intro = vkt.Section("Introduction")
    section_intro.intro = vkt.Text("""
## Footing Design Calculator
This app performs structural checks for a concrete footing design according to ACI 318.
### Two-Way Shear (Punching) - C.11.11.2.1 NSR10

**Critical perimeter:**
$$b_o = 2(b_1 + d) + 2(b_2 + d)$$

**Critical area:**
$$A_o = (b_1 + d)(b_2 + d)$$

**Shear capacity (three conditions):**
$$V_{c1} = 0.75 \\times 0.33 \\times \\lambda \\times \\sqrt{f'_c} \\times b_o \\times d \\times 1000 \\quad \\text{(C.11-33)}$$

$$V_{c2} = 0.75 \\times 0.17 \\times \\left(1 + \\frac{2}{\\beta}\\right) \\times \\lambda \\times \\sqrt{f'_c} \\times b_o \\times d \\times 1000 \\quad \\text{(C.11.11.2.1(a))}$$

where $\\beta = \\frac{\\text{long side}}{\\text{short side}}$

$$V_{c3} = 0.75 \\times 0.083 \\times \\left(\\alpha_s \\frac{d}{b_o} + 2\\right) \\times \\lambda \\times \\sqrt{f'_c} \\times b_o \\times d \\times 1000 \\quad \\text{(C.11-32)}$$

where $\\alpha_s = 40$ for interior columns

### One-Way Shear (Beam Action) - C.11.11.1.1 NSR10

**Shear stress:**
$$\\sigma_V = \\frac{V_u}{B \\times d}$$

**Shear capacity:**
$$V_c = 0.75 \\times \\frac{\\sqrt{f'_c}}{6} \\times 1000 \\text{ kPa}$$

### Bearing Capacity Check - C.7.12.2.1 NSR10

**Bearing strength:**
$$P_{resist} = 1000 \\times 0.65 \\times 0.85 \\times f'_c \\times \\sqrt{\\frac{A_2}{A_1}} \\times A_1 \\times \\sqrt{\\sqrt{\\frac{A_2}{A_1}}}$$

where:
- $A_1 = b_1 \\times b_2$ (pedestal area)
- $A_2 = (b_1 + d)(b_2 + d) - b_1 \\times b_2$ (effective bearing area)
""")

    # Node Coordinates Section
    section_node_coords = vkt.Section("Node Coordinates")
    section_node_coords.intro = vkt.Text("""
### Enter node coordinates from ETABS or other structural software
    """)
    section_node_coords.node_coords = vkt.Table(
        "Node Coordinates",
        default=[
            {"node_name": "N1", "x": 0.0, "y": 0.0, "z": 0.0},
            {"node_name": "N2", "x": 5.0, "y": 0.0, "z": 0.0},
            {"node_name": "N3", "x": 5.0, "y": 5.0, "z": 0.0},
            {"node_name": "N4", "x": 0.0, "y": 5.0, "z": 0.0},
        ],
    )
    section_node_coords.node_coords.node_name = vkt.TextField("Node Name")
    section_node_coords.node_coords.x = vkt.NumberField("X (m)", num_decimals=2)
    section_node_coords.node_coords.y = vkt.NumberField("Y (m)", num_decimals=2)
    section_node_coords.node_coords.z = vkt.NumberField("Z (m)", num_decimals=2)

    # Node Reactions Section
    section_node_reactions = vkt.Section("Node Reactions & Loads")
    section_node_reactions.intro = vkt.Text("""
### Enter reaction forces and moments for each node and load combination
    """)
    section_node_reactions.node_reactions = vkt.Table(
        "Node Reactions & Loads",
        default=[
            {
                "node_name": "N1",
                "load_combo": "LC1",
                "F1": 0.0,
                "F2": 0.0,
                "F3": -15.0,
                "M1": 10.0,
                "M2": 8.0,
                "M3": 0.0,
            },
            {
                "node_name": "N1",
                "load_combo": "LC2",
                "F1": 5.0,
                "F2": 3.0,
                "F3": -18.0,
                "M1": 15.0,
                "M2": 12.0,
                "M3": 0.0,
            },
            {
                "node_name": "N2",
                "load_combo": "LC1",
                "F1": 0.0,
                "F2": 0.0,
                "F3": -20.0,
                "M1": 15.0,
                "M2": 12.0,
                "M3": 0.0,
            },
            {
                "node_name": "N2",
                "load_combo": "LC2",
                "F1": 8.0,
                "F2": 4.0,
                "F3": -23.0,
                "M1": 20.0,
                "M2": 18.0,
                "M3": 0.0,
            },
            {
                "node_name": "N3",
                "load_combo": "LC1",
                "F1": 0.0,
                "F2": 0.0,
                "F3": -20.0,
                "M1": 15.0,
                "M2": 12.0,
                "M3": 0.0,
            },
            {
                "node_name": "N3",
                "load_combo": "LC2",
                "F1": 8.0,
                "F2": 4.0,
                "F3": -23.0,
                "M1": 20.0,
                "M2": 18.0,
                "M3": 0.0,
            },
            {
                "node_name": "N4",
                "load_combo": "LC1",
                "F1": 0.0,
                "F2": 0.0,
                "F3": -15.0,
                "M1": 10.0,
                "M2": 8.0,
                "M3": 0.0,
            },
            {
                "node_name": "N4",
                "load_combo": "LC2",
                "F1": 5.0,
                "F2": 3.0,
                "F3": -18.0,
                "M1": 15.0,
                "M2": 12.0,
                "M3": 0.0,
            },
        ],
    )
    section_node_reactions.node_reactions.node_name = vkt.TextField("Node Name")
    section_node_reactions.node_reactions.load_combo = vkt.TextField("Load Combo")
    section_node_reactions.node_reactions.F1 = vkt.NumberField(
        "F1 (kN)", num_decimals=2
    )
    section_node_reactions.node_reactions.F2 = vkt.NumberField(
        "F2 (kN)", num_decimals=2
    )
    section_node_reactions.node_reactions.F3 = vkt.NumberField(
        "F3 (kN)", num_decimals=2
    )
    section_node_reactions.node_reactions.M1 = vkt.NumberField(
        "M1 (kN·m)", num_decimals=2
    )
    section_node_reactions.node_reactions.M2 = vkt.NumberField(
        "M2 (kN·m)", num_decimals=2
    )
    section_node_reactions.node_reactions.M3 = vkt.NumberField(
        "M3 (kN·m)", num_decimals=2
    )

    # Material Properties
    section_materials = vkt.Section("Material Properties")
    section_materials.fc = vkt.NumberField(
        "Fc Concrete",
        default=28,
        suffix="MPa",
        description="Concrete compressive strength",
    )
    section_materials.fy = vkt.NumberField(
        "Fy Steel", default=420, suffix="MPa", description="Steel yield strength"
    )
    section_materials.gamma_fill = vkt.NumberField(
        "γ Fill Material",
        default=19.5,
        suffix="kN/m³",
        description="Unit weight of fill material",
    )

    # Soil Properties
    section_soil = vkt.Section("Soil Properties")
    section_soil.gamma_soil = vkt.NumberField(
        "γ Soil", default=20, suffix="kN/m³", description="Unit weight of soil"
    )
    section_soil.phi = vkt.NumberField(
        "φ", default=25, suffix="°", description="Soil friction angle"
    )

    # Depth-Dependent Bearing Capacity Table
    section_bearing = vkt.Section("Depth vs Bearing Capacity")
    section_bearing.intro = vkt.Text("""
### Allowable Bearing Capacity at Different Depths
Enter bearing capacity values for different foundation depths. The design will interpolate based on total foundation depth (pedestal height + slab thickness).
    """)
    section_bearing.bearing_table = vkt.Table(
        "Bearing Capacity Table",
        default=[
            {"depth": 1.0, "bearing_capacity": 100.0},
            {"depth": 1.5, "bearing_capacity": 150.0},
            {"depth": 2.0, "bearing_capacity": 250.0},
        ],
    )
    section_bearing.bearing_table.depth = vkt.NumberField("Depth (m)", num_decimals=2)
    section_bearing.bearing_table.bearing_capacity = vkt.NumberField(
        "Bearing Capacity (kPa)", num_decimals=1
    )

    # Footing Dimensions (Initial values for iteration)
    section_footing = vkt.Section("Footing Dimensions")
    section_footing.b = vkt.NumberField(
        "B: Initial Width",
        default=1.0,
        suffix="m",
        min=0.5,
        description="Initial footing width (starts at 1m)",
    )
    section_footing.l = vkt.NumberField(
        "L: Initial Length",
        default=1.0,
        suffix="m",
        min=0.5,
        description="Initial footing length (starts at 1m)",
    )
    section_footing.h = vkt.NumberField(
        "H: Initial Thickness",
        default=0.3,
        suffix="m",
        min=0.15,
        description="Initial slab thickness (starts at 300mm)",
    )
    section_footing.d = vkt.NumberField(
        "d: Effective Depth",
        default=0.210,
        suffix="m",
        description="Effective depth (h - 90mm cover)",
    )

    # Pedestal Dimensions (Initial values for iteration)
    section_pedestal = vkt.Section("Pedestal Dimensions")
    section_pedestal.h_ped = vkt.NumberField(
        "h: Initial Pedestal Size",
        default=0.300,
        suffix="m",
        description="Initial pedestal dimension (square, starts at 300mm)",
    )
    section_pedestal.b_ped = vkt.NumberField(
        "b: Initial Pedestal Size",
        default=0.300,
        suffix="m",
        description="Initial pedestal dimension (square, starts at 300mm)",
    )
    section_pedestal.ped_height = vkt.NumberField(
        "Pedestal Height",
        default=0.600,
        suffix="m",
        description="Pedestal height above footing (600mm)",
    )

    # Equations Display
    section_equations = vkt.Section(
        "Design Equations (NSR-10)", initially_expanded=False
    )

    # Results Export Section
    section_export = vkt.Section("Export Results")
    section_export.download = vkt.DownloadButton(
        "Download Design Results (JSON)", "download_design_results"
    )


class Controller(vkt.Controller):
    parametrization = Parametrization

    @staticmethod
    def get_iteration_ranges() -> dict[str, list[float]]:
        """Get standard iteration ranges for footing design optimization.

        Returns:
            Dictionary with pedestal_sizes, pedestal_heights, thickness_options, footing_dims
        """
        return {
            "pedestal_sizes": [
                round(0.30 + i * 0.05, 2) for i in range(7)
            ],  # 0.30 to 0.60, step 50mm
            "pedestal_heights": [
                round(0.60 + i * 0.10, 2) for i in range(10)
            ],  # 0.60 to 1.50, step 100mm
            "thickness_options": [
                round(0.30 + i * 0.10, 2) for i in range(4)
            ],  # 0.30 to 0.60, step 100mm
            "footing_dims": [
                round(1.0 + i * 0.2, 1) for i in range(16)
            ],  # 1.0 to 4.0, step 0.2m
        }

    @staticmethod
    def create_reactions_lookup(node_reactions_list: list) -> dict[str, list[dict]]:
        """Create lookup dictionary for reactions by node name.

        Args:
            node_reactions_list: List of reaction dictionaries from parametrization

        Returns:
            Dictionary mapping node names to lists of load combination dictionaries
        """
        reactions_by_node = {}
        for reaction in node_reactions_list:
            node_name = reaction.get("node_name", "")
            if node_name:
                if node_name not in reactions_by_node:
                    reactions_by_node[node_name] = []
                reactions_by_node[node_name].append(
                    {
                        "load_combo": reaction.get("load_combo", "LC1"),
                        "F1": float(reaction.get("F1", 0) or 0),
                        "F2": float(reaction.get("F2", 0) or 0),
                        "F3": float(reaction.get("F3", 0) or 0),
                        "M1": float(reaction.get("M1", 0) or 0),
                        "M2": float(reaction.get("M2", 0) or 0),
                        "M3": float(reaction.get("M3", 0) or 0),
                    }
                )
        return reactions_by_node

    def get_optimal_designs_for_all_nodes(
        self, params, **kwargs
    ) -> tuple[dict, dict, dict]:
        """Calculate optimal footing designs for all nodes.

        Args:
            params: Parametrization instance with all input parameters

        Returns:
            Tuple of (optimal_designs, coords_by_node, reactions_by_node)
            - optimal_designs: dict mapping node names to optimal design dicts
            - coords_by_node: dict mapping node names to coordinate dicts
            - reactions_by_node: dict mapping node names to load combo lists
        """
        # Material and design parameters
        fc = params.section_materials.fc
        gamma_concrete = 24.0
        gamma_fill = params.section_materials.gamma_fill

        # Build depth vs bearing capacity lookup from table
        bearing_table = params.section_bearing.bearing_table or []
        get_bearing_capacity = create_bearing_capacity_interpolator(bearing_table)

        # Get iteration ranges
        ranges = self.get_iteration_ranges()
        pedestal_sizes = ranges["pedestal_sizes"]
        pedestal_heights = ranges["pedestal_heights"]
        thickness_options = ranges["thickness_options"]
        footing_dims = ranges["footing_dims"]

        # Get node data
        node_coords_list = params.section_node_coords.node_coords or []
        node_reactions_list = params.section_node_reactions.node_reactions or []

        # Create lookup dictionaries
        coords_by_node = {}
        for coord in node_coords_list:
            node_name = coord.get("node_name", "")
            if node_name:
                coords_by_node[node_name] = {
                    "x": float(coord.get("x", 0) or 0),  # Already in meters
                    "y": float(coord.get("y", 0) or 0),
                }

        reactions_by_node = self.create_reactions_lookup(node_reactions_list)

        # Calculate optimal designs for each node
        optimal_designs = {}

        for node_name in coords_by_node:
            if node_name not in reactions_by_node:
                continue

            load_combos = reactions_by_node[node_name]

            # Check if any load combo has non-zero axial load
            has_axial_load = any(abs(lc["F3"]) > 0.001 for lc in load_combos)
            if not has_axial_load:
                continue

            # Find optimal design using helper function
            optimal = find_optimal_footing_design(
                load_combos=load_combos,
                fc=fc,
                gamma_concrete=gamma_concrete,
                gamma_fill=gamma_fill,
                bearing_capacity_func=get_bearing_capacity,
                pedestal_sizes=pedestal_sizes,
                pedestal_heights=pedestal_heights,
                thickness_options=thickness_options,
                footing_dims=footing_dims,
            )

            if optimal:
                optimal_designs[node_name] = optimal

        return optimal_designs, coords_by_node, reactions_by_node

    @vkt.PlotlyView("Footing Layout (2D)", duration_guess=3)
    def view_footing_layout(self, params, **kwargs):
        """2D plan view showing all footings and pedestals with optimal designs."""

        # Get optimal designs for all nodes
        optimal_designs, coords_by_node, reactions_by_node = (
            self.get_optimal_designs_for_all_nodes(params)
        )

        # Create Plotly figure
        fig = go.Figure()

        # Colors
        footing_color = "rgba(180, 180, 180, 0.6)"
        pedestal_color = "rgba(100, 100, 100, 0.8)"

        # Track bounds for layout
        all_x = []
        all_y = []

        for node_name, coords in coords_by_node.items():
            cx = coords["x"]
            cy = coords["y"]

            if node_name in optimal_designs:
                opt = optimal_designs[node_name]
                B = opt["B"]
                L = opt["L"]
                h = opt["h"]
                ped = opt["pedestal_size"]
                ped_h = opt["pedestal_height"]

                # Draw footing rectangle
                x0, x1 = cx - B / 2, cx + B / 2
                y0, y1 = cy - L / 2, cy + L / 2
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1, x1, x0, x0],
                        y=[y0, y0, y1, y1, y0],
                        mode="lines",
                        fill="toself",
                        fillcolor=footing_color,
                        line=dict(color="rgba(100,100,100,1)", width=2),
                        name=f"{node_name} Footing",
                        hoverinfo="text",
                        text=f"{node_name}<br>Footing: {B}m × {L}m<br>Thickness: {h * 1000:.0f}mm<br>Depth: {(ped_h + h) * 1000:.0f}mm<br>Weight: {opt['total_weight']:.1f}kN",
                        showlegend=False,
                    )
                )

                # Draw pedestal rectangle
                px0, px1 = cx - ped / 2, cx + ped / 2
                py0, py1 = cy - ped / 2, cy + ped / 2
                fig.add_trace(
                    go.Scatter(
                        x=[px0, px1, px1, px0, px0],
                        y=[py0, py0, py1, py1, py0],
                        mode="lines",
                        fill="toself",
                        fillcolor=pedestal_color,
                        line=dict(color="rgba(50,50,50,1)", width=2),
                        name=f"{node_name} Pedestal",
                        hoverinfo="text",
                        text=f"{node_name}<br>Pedestal: {ped * 1000:.0f}mm × {ped * 1000:.0f}mm<br>Height: {ped_h * 1000:.0f}mm",
                        showlegend=False,
                    )
                )

                # Add node label
                fig.add_annotation(
                    x=cx,
                    y=cy,
                    text=f"<b>{node_name}</b>",
                    showarrow=False,
                    font=dict(size=11, color="white"),
                    bgcolor="rgba(50,50,50,0.7)",
                    borderpad=4,
                )

                all_x.extend([x0, x1])
                all_y.extend([y0, y1])
            else:
                # Node without optimal design - just mark position
                fig.add_trace(
                    go.Scatter(
                        x=[cx],
                        y=[cy],
                        mode="markers+text",
                        marker=dict(size=12, color="red", symbol="x"),
                        text=[node_name],
                        textposition="top center",
                        name=f"{node_name} (No design)",
                        showlegend=False,
                    )
                )
                all_x.append(cx)
                all_y.append(cy)

        # Calculate plot bounds
        if all_x and all_y:
            margin = 2.0
            x_range = [min(all_x) - margin, max(all_x) + margin]
            y_range = [min(all_y) - margin, max(all_y) + margin]
        else:
            x_range = [-5, 20]
            y_range = [-5, 20]

        # Layout
        fig.update_layout(
            title="Footing Layout Plan - Optimal Designs",
            xaxis=dict(
                title="X (m)",
                scaleanchor="y",
                scaleratio=1,
                range=x_range,
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
                griddash="dash",
            ),
            yaxis=dict(
                title="Y (m)",
                range=y_range,
                showgrid=True,
                gridcolor="rgba(200, 200, 200, 0.3)",
                griddash="dash",
            ),
            plot_bgcolor="white",
            margin=dict(l=60, r=60, t=60, b=60),
        )

        return vkt.PlotlyResult(fig.to_json())

    @vkt.DataView("Footing Iteration Results", duration_guess=3)
    def view_footing_iterations(self, params, **kwargs):
        """Display detailed iteration results and optimal designs for each node."""
        # Get iteration ranges for display
        ranges = self.get_iteration_ranges()

        # Calculate optimal designs for all nodes
        optimal_designs, coords_by_node, reactions_by_node = (
            self.get_optimal_designs_for_all_nodes(params)
        )

        # Build DataView from results
        main_group = vkt.DataGroup()
        node_groups = []

        for node_name in coords_by_node:
            if node_name not in reactions_by_node:
                continue

            load_combos = reactions_by_node[node_name]

            # Check if any load combo has non-zero axial load
            has_axial_load = any(abs(lc["F3"]) > 0.001 for lc in load_combos)
            if not has_axial_load:
                node_groups.append(
                    vkt.DataItem(
                        f"Node {node_name}",
                        "Skipped - No axial load",
                        status=vkt.DataStatus.WARNING,
                    )
                )
                continue

            # Get optimal design for this node
            optimal = optimal_designs.get(node_name)

            # For nodes with optimal design, also get top 5 alternatives for display
            if optimal:
                # Re-calculate to get top 5 (we could optimize this further by caching)
                fc = params.section_materials.fc
                gamma_concrete = 24.0
                gamma_fill = params.section_materials.gamma_fill
                bearing_table = params.section_bearing.bearing_table or []
                get_bearing_capacity = create_bearing_capacity_interpolator(
                    bearing_table
                )

                sorted_designs = get_top_n_designs(
                    load_combos=load_combos,
                    fc=fc,
                    gamma_concrete=gamma_concrete,
                    gamma_fill=gamma_fill,
                    bearing_capacity_func=get_bearing_capacity,
                    pedestal_sizes=ranges["pedestal_sizes"],
                    pedestal_heights=ranges["pedestal_heights"],
                    thickness_options=ranges["thickness_options"],
                    footing_dims=ranges["footing_dims"],
                    n=5,
                )
            else:
                sorted_designs = []

            # Create subgroup for this node
            node_subgroup = vkt.DataGroup()

            # Add load combo summary
            num_combos = len(load_combos)
            total_combinations = (
                len(ranges["pedestal_sizes"])
                * len(ranges["pedestal_heights"])
                * len(ranges["thickness_options"])
                * len(ranges["footing_dims"]) ** 2
            )
            node_subgroup.add(
                vkt.DataItem("Load Combinations", num_combos),
                vkt.DataItem("Combinations Evaluated", total_combinations),
                vkt.DataItem(
                    "Compliant Designs Found",
                    len(sorted_designs),
                    status=vkt.DataStatus.SUCCESS
                    if len(sorted_designs) > 0
                    else vkt.DataStatus.ERROR,
                ),
            )

            # Add optimal design as flat items (no nested subgroups)
            if optimal is not None:
                design_subgroup = vkt.DataGroup()
                design_subgroup.add(
                    vkt.DataItem(
                        "Pedestal Size",
                        optimal["pedestal_size"] * 1000,
                        suffix="mm",
                        number_of_decimals=0,
                    ),
                    vkt.DataItem(
                        "Pedestal Height",
                        optimal["pedestal_height"] * 1000,
                        suffix="mm",
                        number_of_decimals=0,
                    ),
                    vkt.DataItem(
                        "Footing B × L", f"{optimal['B']:.2f}m × {optimal['L']:.2f}m"
                    ),
                    vkt.DataItem(
                        "Slab Thickness (h)",
                        optimal["h"] * 1000,
                        suffix="mm",
                        number_of_decimals=0,
                    ),
                    vkt.DataItem(
                        "Foundation Depth",
                        optimal["foundation_depth"] * 1000,
                        suffix="mm",
                        number_of_decimals=0,
                    ),
                    vkt.DataItem(
                        "Allowable Bearing",
                        optimal["bearing_capacity"],
                        suffix="kPa",
                        number_of_decimals=1,
                    ),
                    vkt.DataItem(
                        "Footing Area",
                        optimal["B"] * optimal["L"],
                        suffix="m²",
                        number_of_decimals=2,
                    ),
                    vkt.DataItem(
                        "Footing Concrete Weight",
                        optimal["footing_weight"],
                        suffix="kN",
                        number_of_decimals=1,
                    ),
                    vkt.DataItem(
                        "Fill Material Weight",
                        optimal["fill_weight"],
                        suffix="kN",
                        number_of_decimals=1,
                    ),
                    vkt.DataItem(
                        "Total Weight",
                        optimal["total_weight"],
                        suffix="kN",
                        number_of_decimals=1,
                    ),
                    vkt.DataItem(
                        "Governing Combo", optimal["governing_combo"] or "N/A"
                    ),
                    vkt.DataItem(
                        "Max Bearing Pressure",
                        optimal["sigma_max"],
                        suffix="kPa",
                        number_of_decimals=1,
                        status=vkt.DataStatus.SUCCESS,
                    ),
                )

                node_subgroup.add(
                    vkt.DataItem(
                        "✓ OPTIMAL DESIGN",
                        subgroup=design_subgroup,
                        status=vkt.DataStatus.SUCCESS,
                        status_message=f"Min Weight: {optimal['total_weight']:.1f} kN",
                    )
                )

                # Add top alternatives as flat list
                if len(sorted_designs) > 1:
                    alt_subgroup = vkt.DataGroup()
                    for i, alt in enumerate(sorted_designs[1:], start=2):
                        alt_subgroup.add(
                            vkt.DataItem(
                                f"#{i}: Ped={alt['pedestal_size'] * 1000:.0f}mm, H={alt['pedestal_height'] * 1000:.0f}mm",
                                f"B={alt['B']:.1f}m × L={alt['L']:.1f}m × h={alt['h'] * 1000:.0f}mm, Depth={alt['foundation_depth'] * 1000:.0f}mm, W={alt['total_weight']:.1f}kN",
                            )
                        )
                    node_subgroup.add(
                        vkt.DataItem("Alternative Designs", subgroup=alt_subgroup)
                    )
            else:
                node_subgroup.add(
                    vkt.DataItem(
                        "Design Status",
                        "No compliant design found - increase footing range",
                        status=vkt.DataStatus.ERROR,
                    )
                )

            node_groups.append(
                vkt.DataItem(
                    f"Node {node_name}",
                    subgroup=node_subgroup,
                    status=vkt.DataStatus.SUCCESS if optimal else vkt.DataStatus.ERROR,
                )
            )

        # Add all nodes to main group
        if node_groups:
            main_group.add(*node_groups)
        else:
            main_group.add(
                vkt.DataItem(
                    "No Data",
                    "Please add nodes to both tables with matching names",
                    status=vkt.DataStatus.WARNING,
                )
            )

        return vkt.DataResult(main_group)

    def download_design_results(self, params, **kwargs):
        """Export optimal footing designs to JSON file."""

        # Get optimal designs for all nodes
        optimal_designs, coords_by_node, reactions_by_node = (
            self.get_optimal_designs_for_all_nodes(params)
        )

        # Build simplified export data structure
        export_data = {"project": "Footing Design Results", "nodes": []}

        # Add each node's design data
        for node_name in sorted(coords_by_node.keys()):
            coords = coords_by_node[node_name]

            node_data = {
                "node_name": node_name,
                "coordinates_m": {"x": coords["x"], "y": coords["y"]},
            }

            # Check if node has an optimal design
            if node_name in optimal_designs:
                opt = optimal_designs[node_name]
                node_data["governing_load_combo"] = opt.get("governing_combo", "N/A")
                node_data["pedestal"] = {
                    "size_mm": round(opt["pedestal_size"] * 1000),
                    "height_mm": round(opt["pedestal_height"] * 1000),
                }
                node_data["footing"] = {
                    "width_B_mm": round(opt["B"] * 1000),
                    "length_L_mm": round(opt["L"] * 1000),
                    "thickness_h_mm": round(opt["h"] * 1000),
                }
            else:
                node_data["design_status"] = "NO_DESIGN_FOUND"

            export_data["nodes"].append(node_data)

        # Convert to JSON string with nice formatting
        json_content = json.dumps(export_data, indent=2, ensure_ascii=False)

        # Return as downloadable file
        return vkt.DownloadResult(json_content, file_name="footing_design_results.json")
