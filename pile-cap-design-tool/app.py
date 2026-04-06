import math
from dataclasses import dataclass
from html import escape

import numpy as np
import viktor as vkt


N_PILES_IN_GROUP = 3
MIN_GROUP_SPACING_RATIO = 2.5


@dataclass
class SoilParameters:
    name: str
    unit_weight_kN_m3: float
    friction_angle_deg: float
    notes: str


def get_row_value(row, key, default=None):
    """Read VIKTOR table rows whether they arrive as objects or mappings."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def validate_design_inputs(
    pile_diameter_mm: float,
    pile_length_mm: float,
    pile_cap_thickness_mm: float,
    pile_centres_horizontal_mm: float,
    pile_centres_vertical_mm: float,
    factor_of_safety: float,
    soil: SoilParameters,
    column_size_mm: float,
    clear_cover_mm: float,
    bar_diameter_mm: float,
    concrete_strength_mpa: float,
    phi_shear: float,
):
    errors = []
    warnings = []

    if pile_diameter_mm <= 0:
        errors.append("Pile diameter must be greater than 0 mm.")
    if pile_length_mm <= 0:
        errors.append("Pile length must be greater than 0 mm.")
    if pile_cap_thickness_mm <= 0:
        errors.append("Pile cap thickness must be greater than 0 mm.")
    if pile_centres_horizontal_mm <= 0:
        errors.append("Horizontal pile spacing must be greater than 0 mm.")
    if pile_centres_vertical_mm <= 0:
        errors.append("Vertical pile spacing must be greater than 0 mm.")
    if factor_of_safety <= 0:
        errors.append("Factor of safety must be greater than 0.")
    if column_size_mm <= 0:
        errors.append("Column or pedestal size must be greater than 0 mm.")
    if clear_cover_mm < 0:
        errors.append("Clear cover must be zero or greater.")
    if bar_diameter_mm <= 0:
        errors.append("Main bar diameter must be greater than 0 mm.")
    if concrete_strength_mpa <= 0:
        errors.append("Concrete strength must be greater than 0 MPa.")
    if phi_shear <= 0 or phi_shear > 1:
        errors.append("Shear reduction factor phi must be between 0 and 1.")
    if soil.unit_weight_kN_m3 <= 0:
        errors.append("Soil unit weight must be greater than 0 kN/m^3.")
    if soil.friction_angle_deg <= 0 or soil.friction_angle_deg >= 45:
        errors.append("Friction angle must be between 0 and 45 degrees for this model.")

    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    slenderness_ratio = pile_length_mm / pile_diameter_mm
    thickness_ratio = pile_cap_thickness_mm / pile_diameter_mm
    effective_depth_mm = pile_cap_thickness_mm - clear_cover_mm - 0.5 * bar_diameter_mm
    spacing_ratio_h = pile_centres_horizontal_mm / pile_diameter_mm
    spacing_ratio_v = pile_centres_vertical_mm / pile_diameter_mm
    min_spacing_ratio = min(spacing_ratio_h, spacing_ratio_v)

    if effective_depth_mm <= 0:
        errors.append("Effective depth must be greater than 0 mm. Increase cap thickness or reduce cover/bar diameter.")

    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}

    if slenderness_ratio < 10:
        warnings.append(f"Single pile slenderness L/D = {slenderness_ratio:.2f} is below 10.")
    elif slenderness_ratio > 60:
        warnings.append(f"Single pile slenderness L/D = {slenderness_ratio:.2f} exceeds 60.")

    if thickness_ratio < 0.75:
        warnings.append(f"Pile cap thickness ratio t/D = {thickness_ratio:.2f} is below 0.75.")

    if min_spacing_ratio < MIN_GROUP_SPACING_RATIO:
        warnings.append(
            "Minimum pile spacing is below 2.5D, so group interaction is significant and the efficiency reduction becomes controlling."
        )

    return {
        "ok": True,
        "errors": errors,
        "warnings": warnings,
        "slenderness_ratio": slenderness_ratio,
        "thickness_ratio": thickness_ratio,
        "spacing_ratio_h": spacing_ratio_h,
        "spacing_ratio_v": spacing_ratio_v,
        "min_spacing_ratio": min_spacing_ratio,
        "effective_depth_mm": effective_depth_mm,
    }


def calculate_axial_capacity(
    pile_diameter_mm: float,
    pile_length_mm: float,
    factor_of_safety: float,
    soil: SoilParameters,
):
    diameter_m = pile_diameter_mm / 1000.0
    length_m = pile_length_mm / 1000.0

    phi_rad = math.radians(soil.friction_angle_deg)
    delta_deg = 0.67 * soil.friction_angle_deg
    delta_rad = math.radians(delta_deg)

    sigma_v_tip_kPa = soil.unit_weight_kN_m3 * length_m
    sigma_v_avg_kPa = sigma_v_tip_kPa / 2.0
    base_area_m2 = math.pi * diameter_m**2 / 4.0
    shaft_area_m2 = math.pi * diameter_m * length_m

    earth_pressure_coefficient = 1.0 - math.sin(phi_rad)
    beta = earth_pressure_coefficient * math.tan(delta_rad)
    nq = math.exp(math.pi * math.tan(phi_rad)) * math.tan(math.radians(45.0) + phi_rad / 2.0) ** 2

    allowable_bearing_kPa = (nq * sigma_v_tip_kPa) / factor_of_safety
    ultimate_shaft_kN = beta * sigma_v_avg_kPa * shaft_area_m2
    ultimate_tip_kN = nq * sigma_v_tip_kPa * base_area_m2
    ultimate_single_kN = ultimate_shaft_kN + ultimate_tip_kN
    allowable_single_kN = ultimate_single_kN / factor_of_safety

    return {
        "diameter_m": diameter_m,
        "length_m": length_m,
        "phi_rad": phi_rad,
        "delta_deg": delta_deg,
        "delta_rad": delta_rad,
        "sigma_v_tip_kPa": sigma_v_tip_kPa,
        "sigma_v_avg_kPa": sigma_v_avg_kPa,
        "base_area_m2": base_area_m2,
        "shaft_area_m2": shaft_area_m2,
        "earth_pressure_coefficient": earth_pressure_coefficient,
        "beta": beta,
        "nq": nq,
        "allowable_bearing_kPa": allowable_bearing_kPa,
        "ultimate_shaft_kN": ultimate_shaft_kN,
        "ultimate_tip_kN": ultimate_tip_kN,
        "ultimate_single_kN": ultimate_single_kN,
        "allowable_single_kN": allowable_single_kN,
    }


def calculate_group_efficiency(spacing_ratio_h: float, spacing_ratio_v: float):
    theta_h_deg = math.degrees(math.atan(1.0 / spacing_ratio_h))
    theta_v_deg = math.degrees(math.atan(1.0 / spacing_ratio_v))
    theta_avg_deg = (theta_h_deg + theta_v_deg) / 2.0
    efficiency = 1.0 - theta_avg_deg / 90.0
    efficiency = max(0.0, min(1.0, efficiency))
    return {
        "theta_h_deg": theta_h_deg,
        "theta_v_deg": theta_v_deg,
        "theta_avg_deg": theta_avg_deg,
        "efficiency": efficiency,
    }


def calculate_punching_shear(
    pile_cap_thickness_mm: float,
    governing_axial_demand_kN: float,
    column_size_mm: float,
    concrete_strength_mpa: float,
    clear_cover_mm: float,
    bar_diameter_mm: float,
    phi_shear: float,
):
    effective_depth_mm = pile_cap_thickness_mm - clear_cover_mm - 0.5 * bar_diameter_mm
    if effective_depth_mm <= 0:
        raise ValueError("Effective depth must be greater than 0 mm for punching shear.")

    critical_perimeter_mm = 4.0 * (column_size_mm + effective_depth_mm)
    vu_n_per_mm2 = (governing_axial_demand_kN * 1000.0) / (
        critical_perimeter_mm * effective_depth_mm
    )
    vc_mpa = 0.17 * math.sqrt(concrete_strength_mpa)
    phi_vc_mpa = phi_shear * vc_mpa

    return {
        "effective_depth_mm": effective_depth_mm,
        "critical_perimeter_mm": critical_perimeter_mm,
        "vu_mpa": vu_n_per_mm2,
        "vc_mpa": vc_mpa,
        "phi_vc_mpa": phi_vc_mpa,
        "utilization": vu_n_per_mm2 / phi_vc_mpa if phi_vc_mpa > 0 else math.inf,
        "status": "PASS" if vu_n_per_mm2 <= phi_vc_mpa else "FAIL",
    }


def build_reaction_rows(reaction_rows, allowable_single_kN: float, allowable_group_kN: float | None):
    comparison_rows = []
    for row in reaction_rows:
        case_name = str(get_row_value(row, "case_name", "") or "")
        node = str(get_row_value(row, "node", "") or "")
        axial_demand_kN = abs(float(get_row_value(row, "F3", 0.0) or 0.0))

        single_utilization = axial_demand_kN / allowable_single_kN if allowable_single_kN > 0 else math.inf
        if allowable_group_kN is None or allowable_group_kN <= 0:
            group_utilization = None
            status = "GROUP SPACING NOT ACCEPTED"
        else:
            group_utilization = axial_demand_kN / allowable_group_kN
            status = "PASS" if group_utilization <= 1.0 else "FAIL"

        comparison_rows.append(
            {
                "case_name": case_name,
                "node": node,
                "axial_demand_kN": axial_demand_kN,
                "allowable_single_kN": allowable_single_kN,
                "single_utilization": single_utilization,
                "allowable_group_kN": allowable_group_kN,
                "group_utilization": group_utilization,
                "status": status,
            }
        )
    return comparison_rows


def find_governing_row(rows, utilization_key: str):
    filtered_rows = [row for row in rows if row.get(utilization_key) is not None]
    if not filtered_rows:
        return None
    return max(filtered_rows, key=lambda row: row[utilization_key])


def create_validation_error_html(errors, warnings):
    error_items = "".join(f"<li>{escape(message)}</li>" for message in errors)

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Pile Axial Capacity Report</title>
        <style>
            body {{ font-family: Arial, Helvetica, sans-serif; margin: 0; background: #f3f5f7; color: #111827; }}
            .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
            .card {{ background: #ffffff; border: 1px solid #d1d5db; border-radius: 10px; padding: 20px; margin-bottom: 20px; }}
            .error {{ border-left: 6px solid #b91c1c; }}
            .warning {{ border-left: 6px solid #d97706; }}
            h1, h2 {{ margin-top: 0; }}
            ul {{ margin-bottom: 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="card error">
                <h1>Pile Axial Capacity Report</h1>
                <h2>Input Validation Failed</h2>
                <ul>{error_items}</ul>
            </div>
        </div>
    </body>
    </html>
    """


def format_utilization(value):
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def format_status_class(status: str):
    if status == "PASS":
        return "status-pass"
    if status == "FAIL":
        return "status-fail"
    return "status-warn"


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

    bottom_y = -pile_cap_length / 2.0
    top_y = pile_cap_length / 2.0
    shoulder_y = bottom_y + length2
    top_width = length1 - 2.0 * width_indent

    bottom_left_pile = (-pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
    bottom_right_pile = (pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
    top_pile = (0.0, pile_centres_vertical / 2.0)

    cap_points = [
        (-length1 / 2.0, bottom_y),
        (length1 / 2.0, bottom_y),
        (length1 / 2.0, shoulder_y),
        (top_width / 2.0, top_y),
        (-top_width / 2.0, top_y),
        (-length1 / 2.0, shoulder_y),
        (-length1 / 2.0, bottom_y),
    ]

    cap_x = [point[0] for point in cap_points]
    cap_y = [point[1] for point in cap_points]

    fig = go.Figure()
    pile_cap_color = "rgba(180, 180, 180, 0.6)"
    pile_color = "rgba(100, 100, 100, 0.8)"

    fig.add_trace(
        go.Scatter(
            x=cap_x,
            y=cap_y,
            mode="lines",
            line=dict(color="rgba(100,100,100,1)", width=2),
            name="Pile Cap",
            fill="toself",
            fillcolor=pile_cap_color,
            showlegend=False,
        )
    )

    pile_radius = pile_diameter / 2.0
    theta = np.linspace(0, 2 * np.pi, 100)

    for index, (cx, cy) in enumerate([bottom_left_pile, bottom_right_pile, top_pile]):
        pile_x = cx + pile_radius * np.cos(theta)
        pile_y = cy + pile_radius * np.sin(theta)

        fig.add_trace(
            go.Scatter(
                x=pile_x,
                y=pile_y,
                mode="lines",
                line=dict(color="rgba(50,50,50,1)", width=2),
                name=f"Pile {index + 1}",
                fill="toself",
                fillcolor=pile_color,
                showlegend=False,
            )
        )

    margin = max(length1, pile_cap_length) * 0.15
    fig.add_shape(
        type="line",
        x0=-length1 / 2 - margin,
        y0=0,
        x1=length1 / 2 + margin,
        y1=0,
        line=dict(color="gray", dash="dash", width=1),
    )
    fig.add_shape(
        type="line",
        x0=0,
        y0=bottom_y - margin,
        x1=0,
        y1=top_y + margin,
        line=dict(color="gray", dash="dash", width=1),
    )

    fig.update_layout(
        title="3-Pile Cap Plan View",
        xaxis=dict(
            title="X (mm)",
            scaleanchor="y",
            scaleratio=1,
            range=[-length1 / 2 - margin, length1 / 2 + margin],
            showgrid=True,
            gridcolor="rgba(200, 200, 200, 0.3)",
            griddash="dash",
        ),
        yaxis=dict(
            title="Y (mm)",
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


class Parametrization(vkt.Parametrization):
    intro_section = vkt.Section("Description")
    intro_section.text = vkt.Text(
        """
## Pile Axial Capacity Tool

This tool evaluates axial pile capacity for the existing 3-pile cap layout and compares it against the reaction loads table.

**Features**
- Manual reaction load table per node and load case
- Single pile allowable axial capacity from soil and geometry inputs
- 3-pile group allowable axial capacity with Converse-Labarre group efficiency
- Node-by-node and case-by-case utilization checks
- MathJax report with equations, substituted values, and governing cases
- 2D pile cap layout view
        """
    )

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

    reaction_loads_section = vkt.Section("Reaction Loads")
    reaction_loads_section.load_cases = vkt.Table(
        "Reaction Loads",
        default=[
            {"case_name": "LC1", "node": "N1", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
            {"case_name": "LC2", "node": "N1", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
            {"case_name": "LC1", "node": "N2", "F1": 0.0, "F2": 0.0, "F3": -2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
            {"case_name": "LC2", "node": "N2", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
            {"case_name": "LC1", "node": "N3", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
            {"case_name": "LC2", "node": "N3", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
            {"case_name": "LC1", "node": "N4", "F1": 0.0, "F2": 0.0, "F3": 2400.0, "M1": 0.0, "M2": 0.0, "M3": 0.0},
            {"case_name": "LC2", "node": "N4", "F1": 50.0, "F2": 30.0, "F3": 2600.0, "M1": 100.0, "M2": 80.0, "M3": 0.0},
        ],
    )
    reaction_loads_section.load_cases.case_name = vkt.TextField("Load Case Name")
    reaction_loads_section.load_cases.node = vkt.TextField("Node")
    reaction_loads_section.load_cases.F1 = vkt.NumberField("F1", suffix="kN")
    reaction_loads_section.load_cases.F2 = vkt.NumberField("F2", suffix="kN")
    reaction_loads_section.load_cases.F3 = vkt.NumberField("F3", suffix="kN")
    reaction_loads_section.load_cases.M1 = vkt.NumberField("M1", suffix="kN-m")
    reaction_loads_section.load_cases.M2 = vkt.NumberField("M2", suffix="kN-m")
    reaction_loads_section.load_cases.M3 = vkt.NumberField("M3", suffix="kN-m")

    pile_section = vkt.Section("Pile Configuration")
    pile_section.pile_diameter = vkt.NumberField(
        "Pile Diameter",
        default=450.0,
        min=250,
        max=2000,
        num_decimals=0,
        suffix="mm",
        description="Diameter of each pile",
    )
    pile_section.pile_length = vkt.NumberField(
        "Pile Length",
        default=8000.0,
        min=2000,
        max=60000,
        num_decimals=0,
        suffix="mm",
        description="Length of each pile",
    )
    pile_section.pile_centres_horizontal = vkt.NumberField(
        "Horizontal Pile Spacing",
        default=1350.0,
        min=500,
        num_decimals=0,
        suffix="mm",
        description="Horizontal distance between the two bottom pile centres",
    )
    pile_section.pile_centres_vertical = vkt.NumberField(
        "Vertical Pile Spacing",
        default=1350.0,
        min=500,
        num_decimals=0,
        suffix="mm",
        description="Vertical distance between top pile and bottom pile centres",
    )

    cap_section = vkt.Section("Pile Cap Configuration")
    cap_section.pile_cap_thickness = vkt.NumberField(
        "Pile Cap Thickness",
        default=750.0,
        min=300,
        num_decimals=0,
        suffix="mm",
        description="Thickness of the pile cap",
    )
    cap_section.clearance = vkt.NumberField(
        "Clearance",
        default=375.0,
        min=100,
        num_decimals=0,
        suffix="mm",
        description="Offset from pile centre to cap edge",
    )
    cap_section.width_indent = vkt.NumberField(
        "Width Indent",
        default=500.0,
        min=0,
        num_decimals=0,
        suffix="mm",
        description="Horizontal indent on each side at the top",
    )
    cap_section.length2 = vkt.NumberField(
        "Straight Section Length",
        default=750.0,
        min=0,
        num_decimals=0,
        suffix="mm",
        description="Vertical height of straight side before slope starts",
    )
    cap_section.column_size = vkt.NumberField(
        "Column or Pedestal Size",
        default=500.0,
        min=200,
        num_decimals=0,
        suffix="mm",
        description="Equivalent square loaded area used for punching shear",
    )
    cap_section.clear_cover = vkt.NumberField(
        "Bottom Clear Cover",
        default=75.0,
        min=25,
        num_decimals=0,
        suffix="mm",
        description="Concrete cover used to estimate effective depth",
    )
    cap_section.bar_diameter = vkt.NumberField(
        "Main Bar Diameter",
        default=25.0,
        min=10,
        num_decimals=0,
        suffix="mm",
        description="Bar diameter used to estimate effective depth",
    )

    soil_section = vkt.Section("Soil Parameters")
    soil_section.soil_name = vkt.TextField(
        "Soil Type",
        default="Medium Dense Sand",
        description="Descriptive name for the soil",
    )
    soil_section.unit_weight = vkt.NumberField(
        "Unit Weight",
        default=18.0,
        min=10,
        max=30,
        num_decimals=1,
        suffix="kN/m^3",
        description="Unit weight of soil",
    )
    soil_section.friction_angle = vkt.NumberField(
        "Friction Angle",
        default=32.0,
        min=1,
        max=44,
        num_decimals=1,
        suffix="deg",
        description="Internal friction angle of soil",
    )
    soil_section.factor_of_safety = vkt.NumberField(
        "Factor of Safety",
        default=2.5,
        min=1.1,
        num_decimals=2,
        description="Factor of safety used to convert ultimate capacity to allowable capacity",
    )
    soil_section.soil_notes = vkt.TextField(
        "Notes",
        default="Uniform frictional soil profile assumed over the full pile length for this model.",
        description="Additional notes about soil conditions",
    )

    concrete_section = vkt.Section("Concrete Parameters")
    concrete_section.concrete_strength = vkt.NumberField(
        "Concrete Strength f'c",
        default=30.0,
        min=17,
        num_decimals=1,
        suffix="MPa",
        description="Concrete compressive strength used for the punching shear check",
    )
    concrete_section.phi_shear = vkt.NumberField(
        "Shear Reduction Factor phi",
        default=0.75,
        min=0.5,
        max=1.0,
        num_decimals=2,
        description="Strength reduction factor used for punching shear",
    )


class Controller(vkt.Controller):
    parametrization = Parametrization

    @vkt.PlotlyView("Pile Cap Layout (2D)", duration_guess=3)
    def view_pile_cap_layout(self, params, **kwargs):
        import plotly.graph_objects as go

        try:
            pile_centres_vertical_mm = params.pile_section.pile_centres_vertical
            pile_centres_horizontal_mm = params.pile_section.pile_centres_horizontal
            pile_diameter_mm = params.pile_section.pile_diameter
            clearance_mm = params.cap_section.clearance
            width_indent_mm = params.cap_section.width_indent
            length2_mm = params.cap_section.length2

            length1_mm = pile_centres_horizontal_mm + 2.0 * clearance_mm
            pile_cap_length_mm = pile_centres_vertical_mm + 2.0 * clearance_mm

            pile_centres_vertical = pile_centres_vertical_mm / 1000.0
            pile_centres_horizontal = pile_centres_horizontal_mm / 1000.0
            pile_diameter = pile_diameter_mm / 1000.0
            clearance = clearance_mm / 1000.0
            width_indent = width_indent_mm / 1000.0
            length2 = length2_mm / 1000.0
            length1 = length1_mm / 1000.0
            pile_cap_length = pile_cap_length_mm / 1000.0

            bottom_y = -pile_cap_length / 2.0
            top_y = pile_cap_length / 2.0
            shoulder_y = bottom_y + length2
            top_width = length1 - 2.0 * width_indent

            bottom_left_pile = (-pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
            bottom_right_pile = (pile_centres_horizontal / 2.0, -pile_centres_vertical / 2.0)
            top_pile = (0.0, pile_centres_vertical / 2.0)

            cap_points_rel = [
                (-length1 / 2.0, bottom_y),
                (length1 / 2.0, bottom_y),
                (length1 / 2.0, shoulder_y),
                (top_width / 2.0, top_y),
                (-top_width / 2.0, top_y),
                (-length1 / 2.0, shoulder_y),
                (-length1 / 2.0, bottom_y),
            ]

            pile_cap_color = "rgba(180, 180, 180, 0.6)"
            pile_color = "rgba(100, 100, 100, 0.8)"

            fig = go.Figure()
            all_x = []
            all_y = []

            for node_row in params.nodes_section.nodes:
                node_name = get_row_value(node_row, "node_name", "")
                cx = float(get_row_value(node_row, "x", 0.0) or 0.0)
                cy = float(get_row_value(node_row, "y", 0.0) or 0.0)

                cap_x = [cx + point[0] for point in cap_points_rel]
                cap_y = [cy + point[1] for point in cap_points_rel]

                fig.add_trace(
                    go.Scatter(
                        x=cap_x,
                        y=cap_y,
                        mode="lines",
                        line=dict(color="rgba(100,100,100,1)", width=2),
                        fill="toself",
                        fillcolor=pile_cap_color,
                        name=f"{node_name} Cap",
                        hoverinfo="text",
                        text=(
                            f"{node_name}<br>Pile Cap<br>Location: ({cx:.2f}, {cy:.2f}) m"
                            f"<br>Size: {length1:.3f}m x {pile_cap_length:.3f}m"
                        ),
                        showlegend=False,
                    )
                )

                pile_radius = pile_diameter / 2.0
                theta = np.linspace(0, 2 * np.pi, 100)

                for index, (px_rel, py_rel) in enumerate([bottom_left_pile, bottom_right_pile, top_pile]):
                    px = cx + px_rel
                    py = cy + py_rel
                    pile_x = px + pile_radius * np.cos(theta)
                    pile_y = py + pile_radius * np.sin(theta)

                    fig.add_trace(
                        go.Scatter(
                            x=pile_x,
                            y=pile_y,
                            mode="lines",
                            line=dict(color="rgba(50,50,50,1)", width=2),
                            fill="toself",
                            fillcolor=pile_color,
                            name=f"{node_name} Pile {index + 1}",
                            hoverinfo="text",
                            text=f"{node_name} Pile {index + 1}<br>Diameter {pile_diameter_mm:.0f} mm",
                            showlegend=False,
                        )
                    )

                fig.add_annotation(
                    x=cx,
                    y=cy,
                    text=f"<b>{escape(str(node_name))}</b>",
                    showarrow=False,
                    font=dict(size=11, color="white"),
                    bgcolor="rgba(50,50,50,0.7)",
                    borderpad=4,
                )

                all_x.extend([cx - length1 / 2, cx + length1 / 2])
                all_y.extend([cy + bottom_y, cy + top_y])

            if all_x and all_y:
                margin = 2.0
                x_range = [min(all_x) - margin, max(all_x) + margin]
                y_range = [min(all_y) - margin, max(all_y) + margin]
            else:
                x_range = [-5, 20]
                y_range = [-5, 20]

            fig.update_layout(
                title="Pile Cap Layout Plan - All Nodes",
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
                showlegend=False,
            )

            return vkt.PlotlyResult(fig.to_json())

        except Exception as error:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error creating plot: {str(error)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=16, color="red"),
            )
            fig.update_layout(
                title="Error in Pile Cap Visualization",
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
            )
            return vkt.PlotlyResult(fig.to_json())

    @vkt.WebView("Axial Capacity Checks", duration_guess=3)
    def view_design_checks(self, params, **kwargs):
        pile_diameter_mm = params.pile_section.pile_diameter
        pile_length_mm = params.pile_section.pile_length
        pile_cap_thickness_mm = params.cap_section.pile_cap_thickness
        pile_centres_horizontal_mm = params.pile_section.pile_centres_horizontal
        pile_centres_vertical_mm = params.pile_section.pile_centres_vertical
        column_size_mm = params.cap_section.column_size
        clear_cover_mm = params.cap_section.clear_cover
        bar_diameter_mm = params.cap_section.bar_diameter
        factor_of_safety = params.soil_section.factor_of_safety
        concrete_strength_mpa = params.concrete_section.concrete_strength
        phi_shear = params.concrete_section.phi_shear

        soil = SoilParameters(
            name=params.soil_section.soil_name,
            unit_weight_kN_m3=params.soil_section.unit_weight,
            friction_angle_deg=params.soil_section.friction_angle,
            notes=params.soil_section.soil_notes,
        )

        validation = validate_design_inputs(
            pile_diameter_mm=pile_diameter_mm,
            pile_length_mm=pile_length_mm,
            pile_cap_thickness_mm=pile_cap_thickness_mm,
            pile_centres_horizontal_mm=pile_centres_horizontal_mm,
            pile_centres_vertical_mm=pile_centres_vertical_mm,
            factor_of_safety=factor_of_safety,
            soil=soil,
            column_size_mm=column_size_mm,
            clear_cover_mm=clear_cover_mm,
            bar_diameter_mm=bar_diameter_mm,
            concrete_strength_mpa=concrete_strength_mpa,
            phi_shear=phi_shear,
        )

        if not validation["ok"]:
            return vkt.WebResult(html=create_validation_error_html(validation["errors"], validation["warnings"]))

        capacity = calculate_axial_capacity(
            pile_diameter_mm=pile_diameter_mm,
            pile_length_mm=pile_length_mm,
            factor_of_safety=factor_of_safety,
            soil=soil,
        )

        group_efficiency = calculate_group_efficiency(
            spacing_ratio_h=validation["spacing_ratio_h"],
            spacing_ratio_v=validation["spacing_ratio_v"],
        )
        allowable_group_kN = (
            group_efficiency["efficiency"] * N_PILES_IN_GROUP * capacity["allowable_single_kN"]
        )

        comparison_rows = build_reaction_rows(
            reaction_rows=params.reaction_loads_section.load_cases,
            allowable_single_kN=capacity["allowable_single_kN"],
            allowable_group_kN=allowable_group_kN,
        )

        allowable_group_display = f"{allowable_group_kN:.2f} kN"

        total_axial_demand_kN = sum(row["axial_demand_kN"] for row in comparison_rows)
        governing_single = find_governing_row(comparison_rows, "single_utilization")
        governing_group = find_governing_row(comparison_rows, "group_utilization")
        governing_punching_demand_kN = governing_group["axial_demand_kN"] if governing_group is not None else (
            governing_single["axial_demand_kN"] if governing_single is not None else 0.0
        )
        punching_shear = calculate_punching_shear(
            pile_cap_thickness_mm=pile_cap_thickness_mm,
            governing_axial_demand_kN=governing_punching_demand_kN,
            column_size_mm=column_size_mm,
            concrete_strength_mpa=concrete_strength_mpa,
            clear_cover_mm=clear_cover_mm,
            bar_diameter_mm=bar_diameter_mm,
            phi_shear=phi_shear,
        )

        comparison_table_rows = []
        for row in comparison_rows:
            group_capacity_cell = f"{row['allowable_group_kN']:.2f}" if row["allowable_group_kN"] is not None else "N/A"
            comparison_table_rows.append(
                f"""
                <tr>
                    <td>{escape(row['case_name'])}</td>
                    <td>{escape(row['node'])}</td>
                    <td>{row['axial_demand_kN']:.2f}</td>
                    <td>{row['allowable_single_kN']:.2f}</td>
                    <td>{format_utilization(row['single_utilization'])}</td>
                    <td>{group_capacity_cell}</td>
                    <td>{format_utilization(row['group_utilization'])}</td>
                    <td><span class="status {format_status_class(row['status'])}">{escape(row['status'])}</span></td>
                </tr>
                """
            )

        governing_single_text = "No reaction rows available."
        if governing_single is not None:
            governing_single_text = (
                f"{escape(governing_single['case_name'])} at {escape(governing_single['node'])}: "
                f"utilization = {governing_single['single_utilization']:.3f}"
            )

        governing_group_text = "No reaction rows available."
        if governing_group is not None:
            governing_group_text = (
                f"{escape(governing_group['case_name'])} at {escape(governing_group['node'])}: "
                f"utilization = {governing_group['group_utilization']:.3f}"
            )

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Pile Axial Capacity Report</title>
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <style>
                body {{
                    font-family: Arial, Helvetica, sans-serif;
                    margin: 0;
                    background: #f3f5f7;
                    color: #111827;
                }}
                .container {{
                    max-width: 1240px;
                    margin: 0 auto;
                    padding: 24px;
                }}
                .hero {{
                    background: #ffffff;
                    border: 1px solid #d1d5db;
                    border-radius: 14px;
                    padding: 28px;
                    margin-bottom: 24px;
                }}
                .grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
                    gap: 16px;
                    margin-bottom: 24px;
                }}
                .card {{
                    background: #ffffff;
                    border: 1px solid #d1d5db;
                    border-radius: 14px;
                    padding: 20px;
                    margin-bottom: 24px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                    gap: 14px;
                    margin-top: 16px;
                }}
                .metric {{
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    padding: 14px;
                }}
                .metric-label {{
                    font-size: 12px;
                    text-transform: uppercase;
                    color: #6b7280;
                    margin-bottom: 6px;
                    letter-spacing: 0.04em;
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: 700;
                    color: #111827;
                }}
                .status {{
                    display: inline-block;
                    padding: 6px 10px;
                    border-radius: 999px;
                    font-size: 12px;
                    font-weight: 700;
                    letter-spacing: 0.03em;
                }}
                .status-pass {{
                    background: #dcfce7;
                    color: #166534;
                }}
                .status-fail {{
                    background: #fee2e2;
                    color: #991b1b;
                }}
                .status-warn {{
                    background: #fef3c7;
                    color: #92400e;
                }}
                h1 {{
                    margin: 0 0 8px;
                    font-size: 30px;
                }}
                h2 {{
                    margin-top: 0;
                    font-size: 22px;
                }}
                h3 {{
                    font-size: 16px;
                    margin-top: 0;
                }}
                p {{
                    line-height: 1.5;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-top: 12px;
                }}
                th, td {{
                    padding: 12px 10px;
                    border-bottom: 1px solid #e5e7eb;
                    text-align: left;
                    vertical-align: top;
                }}
                th {{
                    font-size: 12px;
                    text-transform: uppercase;
                    color: #6b7280;
                    letter-spacing: 0.04em;
                    background: #f9fafb;
                }}
                .equation-block {{
                    background: #f9fafb;
                    border: 1px solid #e5e7eb;
                    border-radius: 10px;
                    padding: 16px;
                    margin-top: 12px;
                }}
                .equation-note {{
                    color: #4b5563;
                    font-size: 14px;
                    margin-top: 10px;
                }}
                ul {{
                    padding-left: 20px;
                    margin-bottom: 0;
                }}
                .mono {{
                    font-family: Menlo, Consolas, monospace;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="hero">
                    <h1>Pile Axial Capacity Report</h1>
                    <p>Single-pile and 3-pile group axial capacities are evaluated from the soil profile and compared directly against each row in the reaction loads table using \\(P = |F_3|\\).</p>
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-label">Allowable Single Pile Capacity</div>
                            <div class="metric-value">{capacity['allowable_single_kN']:.2f} kN</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Allowable 3-Pile Group Capacity</div>
                            <div class="metric-value">{allowable_group_display}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Group Efficiency Coefficient</div>
                            <div class="metric-value">{group_efficiency['efficiency']:.3f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Computed Allowable Bearing</div>
                            <div class="metric-value">{capacity['allowable_bearing_kPa']:.2f} kPa</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Total Absolute Axial Demand</div>
                            <div class="metric-value">{total_axial_demand_kN:.2f} kN</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Design Inputs</h2>
                    <div class="grid">
                        <div>
                            <h3>Pile Geometry</h3>
                            <p><strong>Diameter:</strong> {pile_diameter_mm:.0f} mm</p>
                            <p><strong>Length:</strong> {pile_length_mm:.0f} mm</p>
                            <p><strong>Slenderness:</strong> L/D = {validation['slenderness_ratio']:.2f}</p>
                        </div>
                        <div>
                            <h3>Pile Group Geometry</h3>
                            <p><strong>Horizontal spacing:</strong> {pile_centres_horizontal_mm:.0f} mm</p>
                            <p><strong>Vertical spacing:</strong> {pile_centres_vertical_mm:.0f} mm</p>
                            <p><strong>Spacing ratios:</strong> s_h/D = {validation['spacing_ratio_h']:.2f}, s_v/D = {validation['spacing_ratio_v']:.2f}</p>
                        </div>
                        <div>
                            <h3>Cap and Soil</h3>
                            <p><strong>Pile cap thickness:</strong> {pile_cap_thickness_mm:.0f} mm</p>
                            <p><strong>Thickness ratio:</strong> t/D = {validation['thickness_ratio']:.2f}</p>
                            <p><strong>Column or pedestal size:</strong> {column_size_mm:.0f} mm</p>
                            <p><strong>Clear cover:</strong> {clear_cover_mm:.0f} mm</p>
                            <p><strong>Main bar diameter:</strong> {bar_diameter_mm:.0f} mm</p>
                            <p><strong>Soil:</strong> {escape(soil.name)}</p>
                            <p><strong>Unit weight:</strong> {soil.unit_weight_kN_m3:.2f} kN/m^3</p>
                            <p><strong>Friction angle:</strong> {soil.friction_angle_deg:.2f} deg</p>
                            <p><strong>Factor of safety:</strong> {factor_of_safety:.2f}</p>
                            <p><strong>Concrete strength f'c:</strong> {concrete_strength_mpa:.1f} MPa</p>
                            <p><strong>Shear phi:</strong> {phi_shear:.2f}</p>
                            <p><strong>Notes:</strong> {escape(soil.notes)}</p>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Derived Geotechnical Parameters</h2>
                    <div class="grid">
                        <div class="metric">
                            <div class="metric-label">Earth Pressure Coefficient K</div>
                            <div class="metric-value">{capacity['earth_pressure_coefficient']:.3f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Interface Angle delta</div>
                            <div class="metric-value">{capacity['delta_deg']:.2f} deg</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Beta</div>
                            <div class="metric-value">{capacity['beta']:.3f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Bearing Factor Nq</div>
                            <div class="metric-value">{capacity['nq']:.3f}</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Converse-Labarre theta_h</div>
                            <div class="metric-value">{group_efficiency['theta_h_deg']:.2f} deg</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Converse-Labarre theta_v</div>
                            <div class="metric-value">{group_efficiency['theta_v_deg']:.2f} deg</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Base Area Ab</div>
                            <div class="metric-value">{capacity['base_area_m2']:.4f} m^2</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Shaft Area As</div>
                            <div class="metric-value">{capacity['shaft_area_m2']:.4f} m^2</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Sigma v at Tip</div>
                            <div class="metric-value">{capacity['sigma_v_tip_kPa']:.2f} kPa</div>
                        </div>
                        <div class="metric">
                            <div class="metric-label">Average Sigma v along Shaft</div>
                            <div class="metric-value">{capacity['sigma_v_avg_kPa']:.2f} kPa</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Equations Used</h2>
                    <div class="equation-block">
                        <div>$$K = 1 - \\sin(\\phi) = 1 - \\sin({soil.friction_angle_deg:.2f}^\\circ) = {capacity['earth_pressure_coefficient']:.3f}$$</div>
                        <div>$$\\delta = 0.67\\phi = 0.67 \\times {soil.friction_angle_deg:.2f}^\\circ = {capacity['delta_deg']:.2f}^\\circ$$</div>
                        <div>$$\\beta = K\\tan(\\delta) = {capacity['earth_pressure_coefficient']:.3f} \\times \\tan({capacity['delta_deg']:.2f}^\\circ) = {capacity['beta']:.3f}$$</div>
                        <div>$$N_q = e^{{\\pi\\tan\\phi}}\\tan^2\\left(45^\\circ + \\frac{{\\phi}}{{2}}\\right) = {capacity['nq']:.3f}$$</div>
                        <div>$$\\sigma_{{v,tip}} = \\gamma L = {soil.unit_weight_kN_m3:.2f} \\times {capacity['length_m']:.3f} = {capacity['sigma_v_tip_kPa']:.2f}\\ \\text{{kPa}}$$</div>
                        <div>$$\\sigma_{{v,avg}} = \\frac{{\\gamma L}}{{2}} = \\frac{{{soil.unit_weight_kN_m3:.2f} \\times {capacity['length_m']:.3f}}}{{2}} = {capacity['sigma_v_avg_kPa']:.2f}\\ \\text{{kPa}}$$</div>
                        <div>$$A_b = \\frac{{\\pi D^2}}{{4}} = \\frac{{\\pi \\times {capacity['diameter_m']:.3f}^2}}{{4}} = {capacity['base_area_m2']:.4f}\\ \\text{{m}}^2$$</div>
                        <div>$$A_s = \\pi D L = \\pi \\times {capacity['diameter_m']:.3f} \\times {capacity['length_m']:.3f} = {capacity['shaft_area_m2']:.4f}\\ \\text{{m}}^2$$</div>
                        <div>$$Q_s = \\beta \\sigma_{{v,avg}} A_s = {capacity['beta']:.3f} \\times {capacity['sigma_v_avg_kPa']:.2f} \\times {capacity['shaft_area_m2']:.4f} = {capacity['ultimate_shaft_kN']:.2f}\\ \\text{{kN}}$$</div>
                        <div>$$Q_b = N_q \\sigma_{{v,tip}} A_b = {capacity['nq']:.3f} \\times {capacity['sigma_v_tip_kPa']:.2f} \\times {capacity['base_area_m2']:.4f} = {capacity['ultimate_tip_kN']:.2f}\\ \\text{{kN}}$$</div>
                        <div>$$Q_{{u,1}} = Q_s + Q_b = {capacity['ultimate_shaft_kN']:.2f} + {capacity['ultimate_tip_kN']:.2f} = {capacity['ultimate_single_kN']:.2f}\\ \\text{{kN}}$$</div>
                        <div>$$Q_{{all,1}} = \\frac{{Q_{{u,1}}}}{{FS}} = \\frac{{{capacity['ultimate_single_kN']:.2f}}}{{{factor_of_safety:.2f}}} = {capacity['allowable_single_kN']:.2f}\\ \\text{{kN}}$$</div>
                        <div>$$q_{{all}} = \\frac{{N_q \\sigma_{{v,tip}}}}{{FS}} = \\frac{{{capacity['nq']:.3f} \\times {capacity['sigma_v_tip_kPa']:.2f}}}{{{factor_of_safety:.2f}}} = {capacity['allowable_bearing_kPa']:.2f}\\ \\text{{kPa}}$$</div>
                        <div>$$\\theta_h = \\tan^{{-1}}\\left(\\frac{{D}}{{s_h}}\\right) = \\tan^{{-1}}\\left(\\frac{{1}}{{{validation['spacing_ratio_h']:.3f}}}\\right) = {group_efficiency['theta_h_deg']:.2f}^\\circ$$</div>
                        <div>$$\\theta_v = \\tan^{{-1}}\\left(\\frac{{D}}{{s_v}}\\right) = \\tan^{{-1}}\\left(\\frac{{1}}{{{validation['spacing_ratio_v']:.3f}}}\\right) = {group_efficiency['theta_v_deg']:.2f}^\\circ$$</div>
                        <div>$$\\theta_{{avg}} = \\frac{{\\theta_h + \\theta_v}}{{2}} = \\frac{{{group_efficiency['theta_h_deg']:.2f} + {group_efficiency['theta_v_deg']:.2f}}}{{2}} = {group_efficiency['theta_avg_deg']:.2f}^\\circ$$</div>
                        <div>$$\\eta_g = 1 - \\frac{{\\theta_{{avg}}}}{{90}} = 1 - \\frac{{{group_efficiency['theta_avg_deg']:.2f}}}{{90}} = {group_efficiency['efficiency']:.3f}$$</div>
                        <div>$$Q_{{all,3}} = \\eta_g \\times 3Q_{{all,1}} = {group_efficiency['efficiency']:.3f} \\times 3 \\times {capacity['allowable_single_kN']:.2f} = {allowable_group_kN:.2f}\\ \\text{{kN}}$$</div>
                        <div>$$d = h - c_c - \\frac{{d_b}}{{2}} = {pile_cap_thickness_mm:.0f} - {clear_cover_mm:.0f} - \\frac{{{bar_diameter_mm:.0f}}}{{2}} = {punching_shear['effective_depth_mm']:.1f}\\ \\text{{mm}}$$</div>
                        <div>$$b_o = 4(c + d) = 4({column_size_mm:.0f} + {punching_shear['effective_depth_mm']:.1f}) = {punching_shear['critical_perimeter_mm']:.1f}\\ \\text{{mm}}$$</div>
                        <div>$$v_u = \\frac{{V_u}}{{b_o d}} = \\frac{{{governing_punching_demand_kN:.2f} \\times 1000}}{{{punching_shear['critical_perimeter_mm']:.1f} \\times {punching_shear['effective_depth_mm']:.1f}}} = {punching_shear['vu_mpa']:.3f}\\ \\text{{MPa}}$$</div>
                        <div>$$v_c = 0.17\\sqrt{{f'_c}} = 0.17\\sqrt{{{concrete_strength_mpa:.1f}}} = {punching_shear['vc_mpa']:.3f}\\ \\text{{MPa}}$$</div>
                        <div>$$\\phi v_c = {phi_shear:.2f} \\times {punching_shear['vc_mpa']:.3f} = {punching_shear['phi_vc_mpa']:.3f}\\ \\text{{MPa}}$$</div>
                    </div>
                    <p class="equation-note">Pile group efficiency is evaluated with the Converse-Labarre method based on the horizontal and vertical pile spacing ratios.</p>
                </div>

                <div class="card">
                    <h2>Capacity Summary</h2>
                    <table>
                        <tr><th>Result</th><th>Value</th></tr>
                        <tr><td>Ultimate shaft resistance Qs</td><td>{capacity['ultimate_shaft_kN']:.2f} kN</td></tr>
                        <tr><td>Ultimate tip resistance Qb</td><td>{capacity['ultimate_tip_kN']:.2f} kN</td></tr>
                        <tr><td>Ultimate single-pile capacity Qu,1</td><td>{capacity['ultimate_single_kN']:.2f} kN</td></tr>
                        <tr><td>Allowable single-pile capacity Qall,1</td><td>{capacity['allowable_single_kN']:.2f} kN</td></tr>
                        <tr><td>Computed allowable bearing qall</td><td>{capacity['allowable_bearing_kPa']:.2f} kPa</td></tr>
                        <tr><td>Group efficiency coefficient eta_g</td><td>{group_efficiency['efficiency']:.3f}</td></tr>
                        <tr><td>Allowable 3-pile group capacity Qall,3</td><td>{allowable_group_display}</td></tr>
                    </table>
                </div>

                <div class="card">
                    <h2>Punching Shear Check</h2>
                    <table>
                        <tr><th>Result</th><th>Value</th></tr>
                        <tr><td>Governing punching demand V_u</td><td>{governing_punching_demand_kN:.2f} kN</td></tr>
                        <tr><td>Effective depth d</td><td>{punching_shear['effective_depth_mm']:.1f} mm</td></tr>
                        <tr><td>Critical perimeter b_o</td><td>{punching_shear['critical_perimeter_mm']:.1f} mm</td></tr>
                        <tr><td>Demand stress v_u</td><td>{punching_shear['vu_mpa']:.3f} MPa</td></tr>
                        <tr><td>Concrete shear stress v_c</td><td>{punching_shear['vc_mpa']:.3f} MPa</td></tr>
                        <tr><td>Design shear strength phi v_c</td><td>{punching_shear['phi_vc_mpa']:.3f} MPa</td></tr>
                        <tr><td>Punching utilization</td><td>{punching_shear['utilization']:.3f}</td></tr>
                        <tr><td>Status</td><td><span class="status {format_status_class(punching_shear['status'])}">{punching_shear['status']}</span></td></tr>
                    </table>
                </div>

                <div class="card">
                    <h2>Reaction Load Comparison</h2>
                    <table>
                        <tr>
                            <th>Case</th>
                            <th>Node</th>
                            <th>Axial Demand P = |F3| (kN)</th>
                            <th>Allowable Single (kN)</th>
                            <th>Single Utilization</th>
                            <th>Allowable Group (kN)</th>
                            <th>Group Utilization</th>
                            <th>Status</th>
                        </tr>
                        {''.join(comparison_table_rows)}
                    </table>
                </div>

                <div class="card">
                    <h2>Governing Cases</h2>
                    <p><strong>Worst single-pile utilization:</strong> {governing_single_text}</p>
                    <p><strong>Worst 3-pile group utilization:</strong> {governing_group_text}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return vkt.WebResult(html=html)
