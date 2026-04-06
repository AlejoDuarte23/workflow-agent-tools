import viktor as vkt
import numpy as np
from dataclasses import dataclass, asdict


# ==============================================
# DATA CONTAINERS
# ==============================================

@dataclass
class SoilParameters:
    name: str
    unit_weight_kN_m3: float
    friction_angle_deg: float
    allowable_bearing_kPa: float
    notes: str


@dataclass
class AxialLoads:
    dead_load_kN: float
    live_load_kN: float

    @property
    def service_total_kN(self) -> float:
        return self.dead_load_kN + self.live_load_kN


# ==============================================
# HELPER FUNCTIONS - PILE CHECKS
# ==============================================

def check_single_pile(
    pile_diameter_mm: float,
    pile_length_mm: float,
    axial_service_load_kN: float | None = None,
):
    """
    Simple single pile geometry check.

    Checks:
    - positive diameter and length
    - diameter range
    - length range
    - slenderness ratio L/D

    Returns a dictionary with:
    - ok
    - messages
    - computed values
    """
    messages = []
    ok = True

    if pile_diameter_mm <= 0:
        ok = False
        messages.append("Pile diameter must be greater than 0 mm.")
    if pile_length_mm <= 0:
        ok = False
        messages.append("Pile length must be greater than 0 mm.")

    if not ok:
        return {
            "ok": ok,
            "messages": messages,
            "diameter_mm": pile_diameter_mm,
            "length_mm": pile_length_mm,
        }

    # Basic practical ranges for a simple script
    if pile_diameter_mm < 250:
        ok = False
        messages.append("Pile diameter is very small. Check units or input value.")
    elif pile_diameter_mm > 2000:
        ok = False
        messages.append("Pile diameter is very large. Check units or input value.")
    else:
        messages.append("Pile diameter is within a reasonable input range.")

    if pile_length_mm < 2000:
        ok = False
        messages.append("Pile length is very short. Check units or input value.")
    elif pile_length_mm > 60000:
        ok = False
        messages.append("Pile length is very large. Check units or input value.")
    else:
        messages.append("Pile length is within a reasonable input range.")

    slenderness_ratio = pile_length_mm / pile_diameter_mm

    if slenderness_ratio < 5:
        ok = False
        messages.append(f"L/D = {slenderness_ratio:.2f}, which is too low for a typical pile.")
    elif slenderness_ratio < 10:
        messages.append(f"L/D = {slenderness_ratio:.2f}. This is low and should be reviewed.")
    elif slenderness_ratio <= 60:
        messages.append(f"L/D = {slenderness_ratio:.2f}. This looks reasonable for a simple check.")
    else:
        messages.append(f"L/D = {slenderness_ratio:.2f}. This is slender and should be reviewed.")

    if axial_service_load_kN is not None:
        if axial_service_load_kN <= 0:
            ok = False
            messages.append("Axial service load must be greater than 0 kN.")
        else:
            messages.append(f"Axial service load = {axial_service_load_kN:.2f} kN.")

    return {
        "ok": ok,
        "messages": messages,
        "diameter_mm": pile_diameter_mm,
        "length_mm": pile_length_mm,
        "slenderness_ratio_L_over_D": slenderness_ratio,
        "axial_service_load_kN": axial_service_load_kN,
    }


def check_pile_group(
    pile_diameter_mm: float,
    pile_length_mm: float,
    pile_centres_horizontal_mm: float,
    pile_centres_vertical_mm: float,
    n_piles: int = 3,
    total_service_load_kN: float | None = None,
):
    """
    Simple pile group configuration check.

    Checks:
    - pile count
    - positive spacing
    - spacing / diameter ratio
    - single pile geometry again
    - equal diameter and equal length are assumed because one diameter and one length are used

    Returns a dictionary.
    """
    messages = []
    ok = True

    if n_piles < 1:
        ok = False
        messages.append("Number of piles must be at least 1.")

    if pile_centres_horizontal_mm <= 0:
        ok = False
        messages.append("Horizontal pile centre spacing must be greater than 0 mm.")

    if pile_centres_vertical_mm <= 0:
        ok = False
        messages.append("Vertical pile centre spacing must be greater than 0 mm.")

    single_result = check_single_pile(
        pile_diameter_mm=pile_diameter_mm,
        pile_length_mm=pile_length_mm,
        axial_service_load_kN=None,
    )

    if not single_result["ok"]:
        ok = False
        messages.append("Single pile geometry check failed.")
        messages.extend(single_result["messages"])
        return {
            "ok": ok,
            "messages": messages,
        }

    s_h_over_d = pile_centres_horizontal_mm / pile_diameter_mm
    s_v_over_d = pile_centres_vertical_mm / pile_diameter_mm

    if s_h_over_d < 2.5:
        ok = False
        messages.append(
            f"Horizontal spacing ratio s/D = {s_h_over_d:.2f}, which is too small for a simple layout check."
        )
    else:
        messages.append(
            f"Horizontal spacing ratio s/D = {s_h_over_d:.2f}, acceptable for this simple check."
        )

    if s_v_over_d < 2.5:
        ok = False
        messages.append(
            f"Vertical spacing ratio s/D = {s_v_over_d:.2f}, which is too small for a simple layout check."
        )
    else:
        messages.append(
            f"Vertical spacing ratio s/D = {s_v_over_d:.2f}, acceptable for this simple check."
        )

    load_per_pile_kN = None
    if total_service_load_kN is not None:
        if total_service_load_kN <= 0:
            ok = False
            messages.append("Total service load must be greater than 0 kN.")
        else:
            load_per_pile_kN = total_service_load_kN / n_piles
            messages.append(
                f"Estimated average service load per pile = {load_per_pile_kN:.2f} kN "
                f"(total load divided equally by {n_piles} piles)."
            )

    messages.append("This helper assumes all piles in the group use the same diameter and the same length.")

    return {
        "ok": ok,
        "messages": messages,
        "n_piles": n_piles,
        "pile_diameter_mm": pile_diameter_mm,
        "pile_length_mm": pile_length_mm,
        "pile_centres_horizontal_mm": pile_centres_horizontal_mm,
        "pile_centres_vertical_mm": pile_centres_vertical_mm,
        "spacing_ratio_horizontal_s_over_D": s_h_over_d,
        "spacing_ratio_vertical_s_over_D": s_v_over_d,
        "average_service_load_per_pile_kN": load_per_pile_kN,
    }


def check_pile_cap_thickness(
    pile_cap_thickness_mm: float,
    pile_diameter_mm: float,
):
    """
    Simple pile cap thickness sanity check.

    Checks:
    - positive thickness
    - thickness / pile diameter ratio

    This is only a quick geometric check.
    """
    messages = []
    ok = True

    if pile_cap_thickness_mm <= 0:
        ok = False
        messages.append("Pile cap thickness must be greater than 0 mm.")

    if pile_diameter_mm <= 0:
        ok = False
        messages.append("Pile diameter must be greater than 0 mm.")

    if not ok:
        return {
            "ok": ok,
            "messages": messages,
        }

    ratio_t_over_d = pile_cap_thickness_mm / pile_diameter_mm

    if ratio_t_over_d < 0.75:
        ok = False
        messages.append(
            f"Thickness ratio t/D = {ratio_t_over_d:.2f}, which is low for a simple pile cap check."
        )
    elif ratio_t_over_d < 1.0:
        messages.append(
            f"Thickness ratio t/D = {ratio_t_over_d:.2f}. This may work, but it should be reviewed."
        )
    else:
        messages.append(
            f"Thickness ratio t/D = {ratio_t_over_d:.2f}. This looks reasonable for a simple sanity check."
        )

    return {
        "ok": ok,
        "messages": messages,
        "pile_cap_thickness_mm": pile_cap_thickness_mm,
        "pile_diameter_mm": pile_diameter_mm,
        "thickness_ratio_t_over_D": ratio_t_over_d,
    }


# ==============================================
# PLOTLY HELPER - 3 PILE CAP DRAWING
# ==============================================

def create_pile_cap_plotly(
    pile_centres_vertical=1350.0,
    pile_centres_horizontal=1350.0,
    pile_diameter=450.0,
    clearance=375.0,
    width_indent=500.0,
    length2=750.0,
    length1=None,
    pile_cap_length=None,
):
    """
    Create Plotly figure for 3-pile cap plan view.

    Parameters
    ----------
    pile_centres_vertical : float
        Vertical distance between the top pile centre and the bottom pile centres (mm).
    pile_centres_horizontal : float
        Horizontal distance between the two bottom pile centres (mm).
    pile_diameter : float
        Pile diameter (mm).
    clearance : float
        Offset from pile centre to the cap edge (mm).
    width_indent : float
        Horizontal indent on each side at the top (mm).
    length2 : float
        Vertical height of the straight side before the slope starts (mm).
    length1 : float or None
        Bottom width of the pile cap (mm).
        If None, it is computed as pile_centres_horizontal + 2 * clearance.
    pile_cap_length : float or None
        Total height of the pile cap (mm).
        If None, it is computed as pile_centres_vertical + 2 * clearance.

    Returns
    -------
    plotly figure
    """
    import plotly.graph_objects as go

    if length1 is None:
        length1 = pile_centres_horizontal + 2.0 * clearance
    if pile_cap_length is None:
        pile_cap_length = pile_centres_vertical + 2.0 * clearance

    if length1 <= 0 or pile_cap_length <= 0:
        raise ValueError("length1 and pile_cap_length must be greater than zero.")
    if width_indent < 0:
        raise ValueError("width_indent must be zero or greater.")
    if width_indent * 2 >= length1:
        raise ValueError("width_indent is too large for the given bottom width.")
    if length2 < 0 or length2 > pile_cap_length:
        raise ValueError("length2 must be between 0 and pile_cap_length.")

    # Cap geometry
    bottom_y = -pile_cap_length / 2.0
    top_y = pile_cap_length / 2.0
    shoulder_y = bottom_y + length2
    top_width = length1 - 2.0 * width_indent

    # Pile centres
    bottom_left_pile = (-pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
    bottom_right_pile = (pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
    top_pile = (0.0, pile_centres_vertical / 2.0)

    # Cap outline: symmetric polygon
    cap_points = [
        (-length1 / 2.0, bottom_y),
        (length1 / 2.0, bottom_y),
        (length1 / 2.0, shoulder_y),
        (top_width / 2.0, top_y),
        (-top_width / 2.0, top_y),
        (-length1 / 2.0, shoulder_y),
        (-length1 / 2.0, bottom_y),  # Close polygon
    ]

    cap_x = [p[0] for p in cap_points]
    cap_y = [p[1] for p in cap_points]

    # Create figure
    fig = go.Figure()

    # Colors matching footing sizing tool
    pile_cap_color = "rgba(180, 180, 180, 0.6)"
    pile_color = "rgba(100, 100, 100, 0.8)"

    # Draw pile cap outline
    fig.add_trace(go.Scatter(
        x=cap_x,
        y=cap_y,
        mode='lines',
        line=dict(color='rgba(100,100,100,1)', width=2),
        name='Pile Cap',
        fill='toself',
        fillcolor=pile_cap_color,
        showlegend=False,
    ))

    # Draw piles
    pile_radius = pile_diameter / 2.0
    theta = np.linspace(0, 2*np.pi, 100)

    for i, (cx, cy) in enumerate([bottom_left_pile, bottom_right_pile, top_pile]):
        pile_x = cx + pile_radius * np.cos(theta)
        pile_y = cy + pile_radius * np.sin(theta)

        fig.add_trace(go.Scatter(
            x=pile_x,
            y=pile_y,
            mode='lines',
            line=dict(color='rgba(50,50,50,1)', width=2),
            name=f'Pile {i+1}',
            fill='toself',
            fillcolor=pile_color,
            showlegend=False,
        ))

    # Add centre lines
    margin = max(length1, pile_cap_length) * 0.15
    fig.add_shape(
        type='line',
        x0=-length1/2 - margin, y0=0, x1=length1/2 + margin, y1=0,
        line=dict(color='gray', dash='dash', width=1)
    )
    fig.add_shape(
        type='line',
        x0=0, y0=bottom_y - margin, x1=0, y1=top_y + margin,
        line=dict(color='gray', dash='dash', width=1)
    )

    # Layout
    fig.update_layout(
        title='3-Pile Cap Plan View',
        xaxis=dict(
            title='X (mm)',
            scaleanchor='y',
            scaleratio=1,
            range=[-length1/2 - margin, length1/2 + margin],
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            griddash="dash",
        ),
        yaxis=dict(
            title='Y (mm)',
            range=[bottom_y - margin, top_y + margin],
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            griddash="dash",
        ),
        plot_bgcolor="white",
        margin=dict(l=60, r=60, t=60, b=60),
        showlegend=True,
        width=800,
        height=800,
    )

    return fig


# ==============================================
# PARAMETRIZATION
# ==============================================

class Parametrization(vkt.Parametrization):
    """Define all inputs for the pile cap design tool"""

    # Introduction
    intro_section = vkt.Section("Description")

    intro_section.text = vkt.Text("""
## Pile Cap Design Tool

This tool checks pile cap configurations for 3-pile groups.

**Features:**
- Input node coordinates and load cases per node
- Configure pile geometry (diameter, length, spacing)
- Configure pile cap geometry (thickness, clearance, taper)
- Input soil parameters
- Check single pile, pile group, and pile cap thickness
- Visualize pile cap and piles in 2D

**Note:** This tool performs sanity checks only. It does NOT optimize the design.
All checks are simple geometric validations. For detailed geotechnical and structural design,
consult with a qualified engineer.
    """)

    # Nodes definition
    nodes_section = vkt.Section("Nodes")
    nodes_section.nodes = vkt.Table(
        "Node Coordinates",
        default=[
            {"node_name": "N1", "x": 0.0, "y": 0.0, "z": 0.0},
            {"node_name": "N2", "x": 5.0, "y": 0.0, "z": 0.0},
            {"node_name": "N3", "x": 5.0, "y": 5.0, "z": 0.0},
            {"node_name": "N4", "x": 0.0, "y": 5.0, "z": 0.0},
        ],
    )
    nodes_section.nodes.node_name = vkt.TextField("Node Name")
    nodes_section.nodes.x = vkt.NumberField("X", num_decimals=2, suffix="m")
    nodes_section.nodes.y = vkt.NumberField("Y", num_decimals=2, suffix="m")
    nodes_section.nodes.z = vkt.NumberField("Z", num_decimals=2, suffix="m")

    # Load cases for all nodes
    load_cases_section = vkt.Section("Load Cases")
    load_cases_section.load_cases = vkt.Table("Load Cases", default=[
        {"case_name": "LC1", "node": "N1", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N1", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
        {"case_name": "LC1", "node": "N2", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N2", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
        {"case_name": "LC1", "node": "N3", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N3", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
        {"case_name": "LC1", "node": "N4", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N4", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
    ])
    load_cases_section.load_cases.case_name = vkt.TextField("Load Case Name")
    load_cases_section.load_cases.node = vkt.TextField("Node")
    load_cases_section.load_cases.F1 = vkt.NumberField("F1", suffix="kN")
    load_cases_section.load_cases.F2 = vkt.NumberField("F2", suffix="kN")
    load_cases_section.load_cases.F3 = vkt.NumberField("F3", suffix="kN")
    load_cases_section.load_cases.M1 = vkt.NumberField("M1", suffix="kN-m")
    load_cases_section.load_cases.M2 = vkt.NumberField("M2", suffix="kN-m")
    load_cases_section.load_cases.M3 = vkt.NumberField("M3", suffix="kN-m")

    # Pile configuration
    pile_section = vkt.Section("Pile Configuration")
    pile_section.pile_diameter = vkt.NumberField(
        "Pile Diameter",
        default=450.0,
        min=250,
        max=2000,
        num_decimals=0,
        suffix="mm",
        description="Diameter of each pile"
    )
    pile_section.pile_length = vkt.NumberField(
        "Pile Length",
        default=8000.0,
        min=2000,
        max=60000,
        num_decimals=0,
        suffix="mm",
        description="Length of each pile"
    )
    pile_section.pile_centres_horizontal = vkt.NumberField(
        "Horizontal Pile Spacing",
        default=1350.0,
        min=500,
        num_decimals=0,
        suffix="mm",
        description="Horizontal distance between the two bottom pile centres"
    )
    pile_section.pile_centres_vertical = vkt.NumberField(
        "Vertical Pile Spacing",
        default=1350.0,
        min=500,
        num_decimals=0,
        suffix="mm",
        description="Vertical distance between top pile and bottom pile centres"
    )

    # Pile cap configuration
    cap_section = vkt.Section("Pile Cap Configuration")
    cap_section.pile_cap_thickness = vkt.NumberField(
        "Pile Cap Thickness",
        default=750.0,
        min=300,
        num_decimals=0,
        suffix="mm",
        description="Thickness of the pile cap"
    )
    cap_section.clearance = vkt.NumberField(
        "Clearance",
        default=375.0,
        min=100,
        num_decimals=0,
        suffix="mm",
        description="Offset from pile centre to cap edge"
    )
    cap_section.width_indent = vkt.NumberField(
        "Width Indent",
        default=500.0,
        min=0,
        num_decimals=0,
        suffix="mm",
        description="Horizontal indent on each side at the top"
    )
    cap_section.length2 = vkt.NumberField(
        "Straight Section Length",
        default=750.0,
        min=0,
        num_decimals=0,
        suffix="mm",
        description="Vertical height of straight side before slope starts"
    )

    # Soil parameters
    soil_section = vkt.Section("Soil Parameters")
    soil_section.soil_name = vkt.TextField(
        "Soil Type",
        default="Medium Dense Sand",
        description="Descriptive name for the soil"
    )
    soil_section.unit_weight = vkt.NumberField(
        "Unit Weight",
        default=18.0,
        min=10,
        max=30,
        num_decimals=1,
        suffix="kN/m³",
        description="Unit weight of soil"
    )
    soil_section.friction_angle = vkt.NumberField(
        "Friction Angle",
        default=32.0,
        min=0,
        max=50,
        num_decimals=1,
        suffix="°",
        description="Internal friction angle of soil"
    )
    soil_section.allowable_bearing = vkt.NumberField(
        "Allowable Bearing Pressure",
        default=250.0,
        min=0,
        num_decimals=1,
        suffix="kPa",
        description="Allowable bearing pressure from geotechnical report"
    )
    soil_section.soil_notes = vkt.TextField(
        "Notes",
        default="Example values only. Use site investigation data for actual design.",
        description="Additional notes about soil conditions"
    )


# ==============================================
# CONTROLLER
# ==============================================

class Controller(vkt.Controller):
    """Main controller for pile cap design tool"""

    parametrization = Parametrization

    @vkt.PlotlyView("Pile Cap Layout (2D)", duration_guess=3)
    def view_pile_cap_layout(self, params, **kwargs):
        """2D plan view showing all pile caps at node locations"""
        import plotly.graph_objects as go

        try:
            # Get pile cap parameters (in mm)
            pile_centres_vertical_mm = params.pile_section.pile_centres_vertical
            pile_centres_horizontal_mm = params.pile_section.pile_centres_horizontal
            pile_diameter_mm = params.pile_section.pile_diameter
            clearance_mm = params.cap_section.clearance
            width_indent_mm = params.cap_section.width_indent
            length2_mm = params.cap_section.length2

            # Auto-calculate dimensions (in mm)
            length1_mm = pile_centres_horizontal_mm + 2.0 * clearance_mm
            pile_cap_length_mm = pile_centres_vertical_mm + 2.0 * clearance_mm

            # Convert to meters for plotting
            pile_centres_vertical = pile_centres_vertical_mm / 1000.0
            pile_centres_horizontal = pile_centres_horizontal_mm / 1000.0
            pile_diameter = pile_diameter_mm / 1000.0
            clearance = clearance_mm / 1000.0
            width_indent = width_indent_mm / 1000.0
            length2 = length2_mm / 1000.0
            length1 = length1_mm / 1000.0
            pile_cap_length = pile_cap_length_mm / 1000.0

            # Calculate pile cap geometry (in meters)
            bottom_y = -pile_cap_length / 2.0
            top_y = pile_cap_length / 2.0
            shoulder_y = bottom_y + length2
            top_width = length1 - 2.0 * width_indent

            # Pile centres relative to cap center
            bottom_left_pile = (-pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
            bottom_right_pile = (pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
            top_pile = (0.0, pile_centres_vertical / 2.0)

            # Cap outline points (relative to center)
            cap_points_rel = [
                (-length1 / 2.0, bottom_y),
                (length1 / 2.0, bottom_y),
                (length1 / 2.0, shoulder_y),
                (top_width / 2.0, top_y),
                (-top_width / 2.0, top_y),
                (-length1 / 2.0, shoulder_y),
                (-length1 / 2.0, bottom_y),  # Close polygon
            ]

            # Colors matching footing sizing tool
            pile_cap_color = "rgba(180, 180, 180, 0.6)"
            pile_color = "rgba(100, 100, 100, 0.8)"

            # Create figure
            fig = go.Figure()

            # Track bounds for layout
            all_x = []
            all_y = []

            # Draw pile cap for each node
            for node_row in params.nodes_section.nodes:
                node_name = node_row['node_name']
                cx = node_row['x']  # Node center x (in meters)
                cy = node_row['y']  # Node center y (in meters)

                # Transform cap points to node location
                cap_x = [cx + p[0] for p in cap_points_rel]
                cap_y = [cy + p[1] for p in cap_points_rel]

                # Draw pile cap outline
                fig.add_trace(go.Scatter(
                    x=cap_x,
                    y=cap_y,
                    mode='lines',
                    line=dict(color='rgba(100,100,100,1)', width=2),
                    fill='toself',
                    fillcolor=pile_cap_color,
                    name=f'{node_name} Cap',
                    hoverinfo='text',
                    text=f'{node_name}<br>Pile Cap<br>Location: ({cx:.2f}, {cy:.2f}) m<br>Size: {length1:.3f}m × {pile_cap_length:.3f}m',
                    showlegend=False,
                ))

                # Draw piles at this node
                pile_radius = pile_diameter / 2.0
                theta = np.linspace(0, 2*np.pi, 100)

                for i, (px_rel, py_rel) in enumerate([bottom_left_pile, bottom_right_pile, top_pile]):
                    px = cx + px_rel
                    py = cy + py_rel
                    pile_x = px + pile_radius * np.cos(theta)
                    pile_y = py + pile_radius * np.sin(theta)

                    fig.add_trace(go.Scatter(
                        x=pile_x,
                        y=pile_y,
                        mode='lines',
                        line=dict(color='rgba(50,50,50,1)', width=2),
                        fill='toself',
                        fillcolor=pile_color,
                        name=f'{node_name} Pile {i+1}',
                        hoverinfo='text',
                        text=f'{node_name} Pile {i+1}<br>Ø {pile_diameter_mm:.0f}mm',
                        showlegend=False,
                    ))

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

                # Track bounds
                all_x.extend([cx - length1/2, cx + length1/2])
                all_y.extend([cy + bottom_y, cy + top_y])

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
                title='Pile Cap Layout Plan - All Nodes',
                xaxis=dict(
                    title='X (m)',
                    scaleanchor='y',
                    scaleratio=1,
                    range=x_range,
                    showgrid=True,
                    gridcolor="rgba(200, 200, 200, 0.3)",
                    griddash="dash",
                ),
                yaxis=dict(
                    title='Y (m)',
                    range=y_range,
                    showgrid=True,
                    gridcolor="rgba(200, 200, 200, 0.3)",
                    griddash="dash",
                ),
                plot_bgcolor="white",
                margin=dict(l=60, r=60, t=60, b=60),
                showlegend=False,
            )

            return vkt.PlotlyResult(fig.to_json())

        except Exception as e:
            # Create error figure
            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16, color="red")
            )
            fig.update_layout(
                title="Error in Pile Cap Visualization",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False)
            )
            return vkt.PlotlyResult(fig.to_json())

    @vkt.WebView("Design Checks", duration_guess=3)
    def view_design_checks(self, params, **kwargs):
        """Display all design checks"""

        # Extract parameters
        pile_diameter_mm = params.pile_section.pile_diameter
        pile_length_mm = params.pile_section.pile_length
        pile_cap_thickness_mm = params.cap_section.pile_cap_thickness
        pile_centres_horizontal_mm = params.pile_section.pile_centres_horizontal
        pile_centres_vertical_mm = params.pile_section.pile_centres_vertical

        # Calculate total load from all load cases
        load_cases = params.load_cases_section.load_cases
        total_F3 = sum(row.F3 for row in load_cases)

        # Soil parameters
        soil = SoilParameters(
            name=params.soil_section.soil_name,
            unit_weight_kN_m3=params.soil_section.unit_weight,
            friction_angle_deg=params.soil_section.friction_angle,
            allowable_bearing_kPa=params.soil_section.allowable_bearing,
            notes=params.soil_section.soil_notes,
        )

        # Run checks
        n_piles = 3

        single_check = check_single_pile(
            pile_diameter_mm=pile_diameter_mm,
            pile_length_mm=pile_length_mm,
            axial_service_load_kN=total_F3 / n_piles,
        )

        group_check = check_pile_group(
            pile_diameter_mm=pile_diameter_mm,
            pile_length_mm=pile_length_mm,
            pile_centres_horizontal_mm=pile_centres_horizontal_mm,
            pile_centres_vertical_mm=pile_centres_vertical_mm,
            n_piles=n_piles,
            total_service_load_kN=total_F3,
        )

        cap_check = check_pile_cap_thickness(
            pile_cap_thickness_mm=pile_cap_thickness_mm,
            pile_diameter_mm=pile_diameter_mm,
        )

        # Build HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Pile Cap Design Check Report</title>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <style>
                body {{
                    font-family: 'Arial', 'Helvetica', sans-serif;
                    margin: 20px;
                    background-color: #ffffff;
                    color: #000000;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                }}
                h1 {{
                    color: #000000;
                    border-bottom: 2px solid #000000;
                    padding-bottom: 10px;
                    font-size: 24px;
                    font-weight: bold;
                }}
                h2 {{
                    color: #000000;
                    margin-top: 30px;
                    border-bottom: 1px solid #666666;
                    padding-bottom: 5px;
                    font-size: 18px;
                    font-weight: bold;
                }}
                h3 {{
                    color: #333333;
                    margin-top: 20px;
                    font-size: 16px;
                    font-weight: bold;
                }}
                .equation-block {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    margin: 15px 0;
                    border-left: 3px solid #666666;
                }}
                .equation {{
                    margin: 10px 0;
                    font-size: 16px;
                }}
                .description {{
                    color: #666666;
                    font-style: italic;
                    margin: 10px 0;
                    font-size: 14px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 15px 0;
                    border: 1px solid #cccccc;
                }}
                th {{
                    background-color: #e0e0e0;
                    color: #000000;
                    padding: 10px;
                    text-align: left;
                    font-weight: bold;
                    border: 1px solid #cccccc;
                }}
                td {{
                    padding: 8px;
                    border: 1px solid #cccccc;
                }}
                tr:nth-child(even) {{
                    background-color: #f9f9f9;
                }}
                .result-box {{
                    background-color: #ffffff;
                    border: 1px solid #cccccc;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .result-value {{
                    font-weight: bold;
                }}
                .section {{
                    margin: 30px 0;
                }}
                .input-params {{
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-left: 3px solid #666666;
                    margin: 10px 0;
                }}
                .check-pass {{
                    background-color: #d4edda;
                    border-left: 3px solid #28a745;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .check-fail {{
                    background-color: #f8d7da;
                    border-left: 3px solid #dc3545;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .check-warning {{
                    background-color: #fff3cd;
                    border-left: 3px solid #ffc107;
                    padding: 15px;
                    margin: 15px 0;
                }}
                .status-badge {{
                    display: inline-block;
                    padding: 4px 12px;
                    border-radius: 4px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                .status-pass {{
                    background-color: #28a745;
                    color: white;
                }}
                .status-fail {{
                    background-color: #dc3545;
                    color: white;
                }}
            </style>
        </head>
        <body>
            <div class="container">
            <h1>Pile Cap Design Check Report</h1>

            <div class="section">
                <h2>Design Parameters</h2>
                <div class="input-params">
                    <h3>Soil Parameters</h3>
                    <p><strong>Soil Type:</strong> {soil.name}</p>
                    <p>\\(\\gamma_{{\\text{{soil}}}} = {soil.unit_weight_kN_m3:.1f}\\) kN/m³</p>
                    <p>\\(\\phi = {soil.friction_angle_deg:.1f}\\)°</p>
                    <p>\\(q_{{\\text{{allow}}}} = {soil.allowable_bearing_kPa:.1f}\\) kPa</p>
                    <p class="description">{soil.notes}</p>
                </div>
            </div>

            <div class="section">
                <h2>Load Summary</h2>
                <p><strong>Total Vertical Load:</strong> \\(\\sum F_3 = {total_F3:.2f}\\) kN</p>
                <p><strong>Number of Piles:</strong> \\(n = {n_piles}\\)</p>
                <p><strong>Average Load per Pile:</strong> \\(P_{{\\text{{avg}}}} = \\frac{{\\sum F_3}}{{n}} = {total_F3/n_piles:.2f}\\) kN</p>
            </div>

            <div class="section">
                <h2>Check Equations</h2>

                <h3>Pile Slenderness Ratio</h3>
                <div class="equation-block">
                    <div class="equation">$$\\frac{{L}}{{D}} = \\frac{{{single_check['length_mm']:.0f}\\,\\text{{mm}}}}{{{single_check['diameter_mm']:.0f}\\,\\text{{mm}}}} = {single_check['slenderness_ratio_L_over_D']:.2f}$$</div>
                    <div class="description">Typical range: 10 ≤ L/D ≤ 60</div>
                </div>

                <h3>Pile Spacing Ratio</h3>
                <div class="equation-block">
                    <div class="equation">$$\\frac{{s_h}}{{D}} = \\frac{{{pile_centres_horizontal_mm:.0f}\\,\\text{{mm}}}}{{{pile_diameter_mm:.0f}\\,\\text{{mm}}}} = {group_check['spacing_ratio_horizontal_s_over_D']:.2f}$$</div>
                    <div class="equation">$$\\frac{{s_v}}{{D}} = \\frac{{{pile_centres_vertical_mm:.0f}\\,\\text{{mm}}}}{{{pile_diameter_mm:.0f}\\,\\text{{mm}}}} = {group_check['spacing_ratio_vertical_s_over_D']:.2f}$$</div>
                    <div class="description">Minimum recommended: s/D ≥ 2.5</div>
                </div>

                <h3>Pile Cap Thickness Ratio</h3>
                <div class="equation-block">
                    <div class="equation">$$\\frac{{t}}{{D}} = \\frac{{{pile_cap_thickness_mm:.0f}\\,\\text{{mm}}}}{{{pile_diameter_mm:.0f}\\,\\text{{mm}}}} = {cap_check['thickness_ratio_t_over_D']:.2f}$$</div>
                    <div class="description">Minimum recommended: t/D ≥ 0.75</div>
                </div>
            </div>

            <div class="{'check-pass' if single_check['ok'] else 'check-fail'}">
                <h2>Single Pile Check <span class="status-badge {'status-pass' if single_check['ok'] else 'status-fail'}">{'PASS' if single_check['ok'] else 'FAIL'}</span></h2>
                <ul>
        """

        for msg in single_check['messages']:
            html += f"<li>{msg}</li>"

        html += f"""
                </ul>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Pile Diameter (\\(D\\))</td><td>{single_check['diameter_mm']:.0f} mm</td></tr>
                    <tr><td>Pile Length (\\(L\\))</td><td>{single_check['length_mm']:.0f} mm</td></tr>
                    <tr><td>Slenderness Ratio (\\(L/D\\))</td><td>{single_check['slenderness_ratio_L_over_D']:.2f}</td></tr>
                </table>
            </div>

            <div class="{'check-pass' if group_check['ok'] else 'check-fail'}">
                <h2>Pile Group Check <span class="status-badge {'status-pass' if group_check['ok'] else 'status-fail'}">{'PASS' if group_check['ok'] else 'FAIL'}</span></h2>
                <ul>
        """

        for msg in group_check['messages']:
            html += f"<li>{msg}</li>"

        html += f"""
                </ul>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Number of Piles</td><td>{group_check['n_piles']}</td></tr>
                    <tr><td>Horizontal Spacing Ratio (\\(s_h/D\\))</td><td>{group_check['spacing_ratio_horizontal_s_over_D']:.2f}</td></tr>
                    <tr><td>Vertical Spacing Ratio (\\(s_v/D\\))</td><td>{group_check['spacing_ratio_vertical_s_over_D']:.2f}</td></tr>
        """

        if group_check.get('average_service_load_per_pile_kN'):
            html += f"<tr><td>Avg Load per Pile</td><td>{group_check['average_service_load_per_pile_kN']:.2f} kN</td></tr>"

        html += """
                </table>
            </div>
        """

        html += f"""
            <div class="{'check-pass' if cap_check['ok'] else 'check-fail'}">
                <h2>Pile Cap Thickness Check <span class="status-badge {'status-pass' if cap_check['ok'] else 'status-fail'}">{'PASS' if cap_check['ok'] else 'FAIL'}</span></h2>
                <ul>
        """

        for msg in cap_check['messages']:
            html += f"<li>{msg}</li>"

        html += f"""
                </ul>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Pile Cap Thickness (\\(t\\))</td><td>{cap_check['pile_cap_thickness_mm']:.0f} mm</td></tr>
                    <tr><td>Pile Diameter (\\(D\\))</td><td>{cap_check['pile_diameter_mm']:.0f} mm</td></tr>
                    <tr><td>Thickness Ratio (\\(t/D\\))</td><td>{cap_check['thickness_ratio_t_over_D']:.2f}</td></tr>
                </table>
            </div>

            <div class="check-warning">
                <h2>⚠️ Important Notice</h2>
                <p><strong>These checks are for preliminary design validation only.</strong></p>
                <ul>
                    <li>Geometric and basic ratio checks performed</li>
                    <li>Does NOT include detailed geotechnical analysis</li>
                    <li>Does NOT include structural capacity calculations</li>
                    <li>Does NOT include punching shear checks</li>
                    <li>Does NOT include moment and bending checks</li>
                    <li>Always consult with qualified geotechnical and structural engineers</li>
                    <li>Use site-specific investigation data for final design</li>
                </ul>
            </div>

            </div>
        </body>
        </html>
        """

        return vkt.WebResult(html=html)
