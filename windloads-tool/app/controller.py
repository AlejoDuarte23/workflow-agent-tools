import math
import json

import viktor as vkt
from viktor.geometry import Point, Line, RectangularExtrusion, Polygon, Material, Cone, Group, Vector, SquareBeam
from viktor.views import Label
from app.truss_beam import RectangularTrussBeam


def notched_profile(width=2000.0, height=2400.0, notch=400.0) -> list[vkt.Point]:
    """
    Profile is defined in the local XY plane (z=0).
    This outline is a width x height rectangle with a notch (notch x notch)
    removed at the top-right corner.

    Clockwise + closed (first point repeated at the end) as required by Extrusion.
    """
    w = float(width)
    h = float(height)
    n = float(notch)

    # Center the profile around the local origin (0, 0) so the extrusion line passes through the center
    xL, xR = -w / 2, w / 2
    yB, yT = -h / 2, h / 2

    xN = xR - n      # notch inner x
    yN = yT - n      # notch bottom y

    # Clockwise loop
    return [
        vkt.Point(xL, yB),
        vkt.Point(xL, yT),
        vkt.Point(xN, yT),
        vkt.Point(xN, yN),
        vkt.Point(xR, yN),
        vkt.Point(xR, yB),
        vkt.Point(xL, yB),
    ]


def create_load_arrow(origin: Point, arrow_length: float, material=None) -> Group:
    """
    Create a horizontal load arrow pointing in +Y direction.
    
    Parameters:
        origin: The tip point of the arrow (where it points to)
        arrow_length: Total length of the arrow
        material: Viktor Material for the arrow
    """
    # Arrow proportions
    cone_length = arrow_length * 0.25
    cone_diameter = arrow_length * 0.16  # diameter, not radius
    shaft_size = arrow_length * 0.03

    # The arrow tip is at origin, pointing in +Y direction
    # Cone origin is at the base of the cone
    cone_base = Point(origin.x, origin.y - cone_length, origin.z)
    shaft_start = Point(origin.x, origin.y - arrow_length, origin.z)
    shaft_end = cone_base

    # Create cone (arrow head) pointing in +Y
    # Cone(diameter, height, *, origin, orientation, material)
    arrow_head = Cone(
        cone_diameter,
        cone_length,
        origin=cone_base,
        orientation=Vector(0, 1, 0),
        material=material
    )

    # Create shaft (rectangular extrusion)
    shaft_line = Line(shaft_start, shaft_end)
    arrow_shaft = RectangularExtrusion(
        shaft_size,
        shaft_size,
        shaft_line,
        material=material
    )

    return Group([arrow_head, arrow_shaft])

def _risk_importance_factor(risk_category: str) -> float:
    """
    Optional wind importance factor (Iw) used in some legacy/alternate workflows.

    ASCE 7-10 commonly uses risk-category-specific wind speed maps.
    If your input wind speed V already comes from the correct map for the selected
    risk category, use Iw = 1.0 (recommended).
    """
    return {
        "I": 0.87,
        "II": 1.00,
        "III": 1.15,
        "IV": 1.15,
    }.get(risk_category, 1.00)


def _exposure_constants(exposure: str) -> tuple[float, float]:
    """
    Power-law exposure constants used for Kz:
      Kz = 2.01 * (z/zg)^(2/alpha)

    Returns (alpha, zg_m).
    """
    data = {
        "B": (7.0, 1200.0),
        "C": (9.5, 900.0),
        "D": (11.5, 700.0),
    }
    alpha, zg_ft = data.get(exposure, data["C"])
    zg_m = zg_ft * 0.3048
    return alpha, zg_m


def _kz(z_ref_m: float, exposure: str, z_min_m: float = 4.6) -> float:
    """
    Velocity pressure exposure coefficient (Kz) using:
      Kz = 2.01 * (z/zg)^(2/alpha)   for z >= z_min
      Kz = 2.01 * (z_min/zg)^(2/alpha) for z < z_min

    For open truss/gantry work, z_ref should be a representative height for the
    loaded face (often the centroid height of projected solid area).
    """
    alpha, zg_m = _exposure_constants(exposure)
    z = max(float(z_ref_m), float(z_min_m))
    return 2.01 * (z / zg_m) ** (2.0 / alpha)


def _velocity_pressure_kpa(
    wind_speed_ms: float,
    z_ref_m: float,
    exposure: str,
    kzt: float,
    kd: float,
    iw: float,
) -> dict[str, float]:
    """
    Velocity pressure qz in kPa (SI form).
    Base sea-level coefficient (0.613) corresponds to 0.5 * rho0 with rho0=1.225 kg/m^3.

      qz = 0.613 * Kz * Kzt * Kd * V^2 * Iw   [Pa]

    Returns dict with kz and qz_kpa.
    """
    kz = _kz(z_ref_m, exposure)
    rho = 1.225
    coeff = 0.613

    q_pa = coeff * kz * float(kzt) * float(kd) * (float(wind_speed_ms) ** 2) * float(iw)
    return {
        "kz": kz,
        "rho": rho,
        "coeff": coeff,
        "qz_kpa": q_pa / 1000.0,
    }



def _unit(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
    n = math.sqrt(vx * vx + vy * vy + vz * vz)
    if n <= 0.0:
        return 0.0, 0.0, 0.0
    return vx / n, vy / n, vz / n


def _projected_area_and_centroid_height(
    nodes: dict[int, dict[str, float]],
    lines: dict[int, dict[str, int]],
    member_proj_width_m: float,
    wind_dir_unit: tuple[float, float, float],
) -> dict[str, float]:
    """
    Computes:
    - Af_m2: projected solid area based on line members (approx)
    - z_centroid_m: centroid height of projected area (y-axis used as vertical)

    Member projected area approximation:
      A_i = b * L_i * sin(theta)
    where theta is angle between member axis and wind direction.
    """
    wx, wy, wz = wind_dir_unit
    b = max(float(member_proj_width_m), 0.0)

    Af_members = 0.0
    Az_moment = 0.0  # sum(A_i * y_mid)

    for _, ld in lines.items():
        ni = nodes[ld["NodeI"]]
        nj = nodes[ld["NodeJ"]]

        dx = float(nj["x"] - ni["x"])
        dy = float(nj["y"] - ni["y"])
        dz = float(nj["z"] - ni["z"])

        L = math.sqrt(dx * dx + dy * dy + dz * dz)
        if L <= 0.0:
            continue

        ux, uy, uz = dx / L, dy / L, dz / L
        dot = ux * wx + uy * wy + uz * wz
        sin_theta = math.sqrt(max(0.0, 1.0 - dot * dot))

        Ai = b * L * sin_theta
        if Ai <= 0.0:
            continue

        y_mid = 0.5 * (float(ni["y"]) + float(nj["y"]))
        Af_members += Ai
        Az_moment += Ai * y_mid

    # Centroid height
    if Af_members > 0.0:
        z_centroid = Az_moment / Af_members
    else:
        z_centroid = 0.0

    return {
        "Af_members_m2": Af_members,
        "Af_total_m2": Af_members,
        "z_centroid_m": z_centroid,
    }


def _gross_area(L_m: float, H_m: float) -> float:
    """
    Gross envelope area Ag of the loaded face (L × H).
    Wind normal to side face (transverse wind on long face).
    """
    return max(L_m, 0.0) * max(H_m, 0.0)


def _wind_direction() -> tuple[float, float, float]:
    """
    Wind direction unit vector (normal to side face, along +z).
    Acts on L × H face.

    Coordinate system:
      x = length direction
      y = vertical
      z = width direction
    """
    return _unit(0.0, 0.0, 1.0)

class Parametrization(vkt.Parametrization):
    # Wind Load Generation
    text_1 = vkt.Text("# Wind Load Generation")
    # Site Data
    text_2 = vkt.Text("""## Site Data
Enter the site location and elevation information""")
    risk_category = vkt.OptionField("Risk Category", options=["I", "II", "III", "IV"], default="II")
    site_elevation_m = vkt.NumberField("Site Elevation", suffix="m", default=138.0, min=-500, max=9000)
    lb_2 = vkt.LineBreak()

    # Structure Data (mm)
    text_4 = vkt.Text("""## Structure Data
Define the bridge geometry and member sizing (metric)""")
    bridge_length = vkt.NumberField("Bridge Length, L", min=100, default=20000, suffix="mm")
    bridge_width = vkt.NumberField("Bridge Width, B", min=100, default=4500, suffix="mm")
    bridge_height = vkt.NumberField("Bridge Height, H", min=100, default=3000, suffix="mm")
    roof_pitch_angle = vkt.NumberField("Roof Pitch Angle, θ", suffix="°", default=12, min=0, max=60)
    n_divisions = vkt.NumberField("Number of Divisions", min=1, default=4)

    cross_section = vkt.OptionField(
        "Cross-Section Size",
        options=["HSS200×200×8", "HSS250×250×10", "HSS300×300×12", "HSS350×350×16"],
        default="HSS200×200×8",
    )
    lb_3 = vkt.LineBreak()

    # Wind Data (SI + kPa output)
    text_6 = vkt.Text("""## Wind Data
Specify wind parameters (SI). Results are shown in kPa and kN.""")

    exposure_category = vkt.OptionField("Exposure Category", options=["B", "C", "D"], default="C")
    wind_speed_ms = vkt.NumberField("Basic Wind Speed, V", suffix="m/s", default=47.0, min=0, max=120)

    topographic_factor_kzt = vkt.NumberField("Topographic Factor, Kzt", default=1.0, min=0.5, max=2.0)
    directionality_factor_kd = vkt.NumberField("Directionality Factor, Kd", default=0.85, min=0.5, max=1.0)
    gust_effect_factor_g = vkt.NumberField("Gust Effect Factor, G (rigid default = 0.85)", default=0.85, min=0.5, max=1.2)

    # Lattice framework / open-structure inputs
    text_8 = vkt.Text("## Open Truss / Lattice Framework")

    force_coefficient_cf = vkt.NumberField(
        "Force Coefficient, Cf",
        default=1.6,
        min=0.1,
        max=5.0,
    )

    lb_4 = vkt.LineBreak()

    # Export
    download_json = vkt.DownloadButton("Download JSON", method="download_json_data")

class Controller(vkt.Controller):
    parametrization = Parametrization

    @vkt.GeometryView("3D Model", x_axis_to_right=True)
    def create_render(self, params, **kwargs):
        beam = RectangularTrussBeam(
            length=float(params.bridge_length) / 1000.0,
            width=float(params.bridge_width) / 1000.0,
            height=float(params.bridge_height) / 1000.0,
            n_diagonals=int(params.n_divisions),
        )

        nodes, lines, chord_tl_ids, chord_tr_ids = beam.build()
        nodes, lines = beam.clean_model()
        nodes, lines = beam.remove_top_edge_nodes(chord_tl_ids, chord_tr_ids)

        sections_group = []
        cs_size_m = float(params.cross_section.replace("HSS", "").split("×")[0]) / 1000.0
        B_m = float(params.bridge_width) / 1000.0
        L_m = float(params.bridge_length) / 1000.0

        # Bridge material with red color
        bridge_material = Material(color=vkt.Color.from_hex("#C41E3A"), roughness=0.8, metalness=0.3)

        for line_id, line_data in lines.items():
            ni = nodes[line_data["NodeI"]]
            nj = nodes[line_data["NodeJ"]]

            pi = Point(ni["x"], ni["y"], ni["z"])
            pj = Point(nj["x"], nj["y"], nj["z"])

            axis = Line(pi, pj)
            sections_group.append(
                RectangularExtrusion(cs_size_m, cs_size_m, axis, identifier=str(line_id), material=bridge_material)
            )

        # --- Add concrete abutments ---
        height = float(params.bridge_height) / 1000.0
        concrete_material = Material(color=vkt.Color(80, 80, 80), roughness=0.9, metalness=0.1)
        
        # Left abutment
        node2 = vkt.Point(-0.4, -B_m, -height/2 + cs_size_m)
        node1 = vkt.Point(-0.400, 2*B_m, -height/2 + cs_size_m)
        center_line = vkt.Line(node1, node2)
        profile = notched_profile(width=1.000, height=height, notch=cs_size_m)
        solid = vkt.Extrusion(profile, center_line, profile_rotation=0, material=concrete_material)
        sections_group.append(vkt.Group([solid, center_line]))
        
        # Right abutment
        node1 = vkt.Point(L_m + 0.4, -B_m, -height/2 + cs_size_m) 
        node2 = vkt.Point(L_m + 0.400, 2*B_m, -height/2 + cs_size_m)
        center_line = vkt.Line(node1, node2)
        profile = notched_profile(width=1.000, height=height, notch=cs_size_m)
        solid = vkt.Extrusion(profile, center_line, profile_rotation=180, material=concrete_material)
        sections_group.append(vkt.Group([solid, center_line]))

        # --- Add bridge deck ---
        deck_thickness = 0.2  # meters
        
        deck_material = Material(color=vkt.Color(100, 100, 100), roughness=0.9, metalness=0.1)
        deck = SquareBeam(L_m, B_m, deck_thickness, material=deck_material)
        deck.translate((L_m / 2, B_m / 2, deck_thickness / 2))  # Position at z=0
        sections_group.append(deck)
        
        # --- Add asphalt layer on top ---
        asphalt_thickness = 0.05  # meters
        asphalt_material = Material(color=vkt.Color(40, 40, 40), roughness=1.0, metalness=0.0)
        asphalt = SquareBeam(L_m, B_m, asphalt_thickness, material=asphalt_material)
        asphalt.translate((L_m / 2, B_m / 2, deck_thickness / 2 + asphalt_thickness / 2))
        sections_group.append(asphalt)

        # ---------- translucent red rectangle on the side (xz plane => y constant) ----------
        xs = [float(n["x"]) for n in nodes.values()]
        ys = [float(n["y"]) for n in nodes.values()]
        zs = [float(n["z"]) for n in nodes.values()]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        z_min, z_max = min(zs), max(zs)

        # pick which side: y_min or y_max, offset 2x bridge width away
        side_y = y_min - 2 * B_m

        # small inset so it doesn't exactly coincide with members
        pad_x = 0.02 * (x_max - x_min) if x_max > x_min else 0.0
        pad_z = 0.02 * (z_max - z_min) if z_max > z_min else 0.0
        x0, x1 = x_min + pad_x, x_max - pad_x
        z0, z1 = z_min + pad_z, z_max - pad_z

        red_transparent = Material(color=(255, 0, 0), opacity=0.25, roughness=1.0, metalness=0.0)  # :contentReference[oaicite:1]{index=1}

        # Polygon is defined in the xy plane; we map:
        #   polygon.y := desired z
        # then rotate about x by +90° to land in xz plane (y becomes constant)
        profile_xy = [
            Point(x0, z0, 0),
            Point(x0, z1, 0),
            Point(x1, z1, 0),
            Point(x1, z0, 0),
        ]
        poly = Polygon(profile_xy, surface_orientation=True)

        # make it a thin plate so it renders from both sides
        thickness = 0.001  # 1 mm
        plate = poly.extrude(Line(Point(0, 0, 0), Point(0, 0, thickness)), material=red_transparent)

        plate = plate.rotate(angle=math.pi / 2, direction=(1, 0, 0), point=(0, 0, 0))  # :contentReference[oaicite:2]{index=2}
        plate = plate.translate((0, side_y, 0))  # :contentReference[oaicite:3]{index=3}

        sections_group.append(plate)

        # ---------- Load arrows around the perimeter of the plate ----------
        red_arrow_material = Material(color=(200, 0, 0), opacity=1.0, roughness=0.5, metalness=0.3)
        arrow_length = B_m * 0.6  # Smaller arrows

        # Arrow tip positions (at the plate, pointing toward structure in +Y)
        arrow_tip_y = side_y + 0.01  # Slightly in front of plate

        # Number of arrows along each edge
        n_arrows_x = 7  # Along length (x direction)
        n_arrows_z = 3  # Along height (z direction)

        # Generate arrow positions along the perimeter
        corner_positions = []

        # Bottom edge (z = z0)
        for i in range(n_arrows_x):
            x = x0 + (x1 - x0) * i / (n_arrows_x - 1)
            corner_positions.append(Point(x, arrow_tip_y, z0))

        # Top edge (z = z1)
        for i in range(n_arrows_x):
            x = x0 + (x1 - x0) * i / (n_arrows_x - 1)
            corner_positions.append(Point(x, arrow_tip_y, z1))

        # Left edge (x = x0), excluding corners already added
        for i in range(1, n_arrows_z - 1):
            z = z0 + (z1 - z0) * i / (n_arrows_z - 1)
            corner_positions.append(Point(x0, arrow_tip_y, z))

        # Right edge (x = x1), excluding corners already added
        for i in range(1, n_arrows_z - 1):
            z = z0 + (z1 - z0) * i / (n_arrows_z - 1)
            corner_positions.append(Point(x1, arrow_tip_y, z))

        for corner in corner_positions:
            arrow = create_load_arrow(corner, arrow_length, material=red_arrow_material)
            sections_group.append(arrow)

        # ---------- Pressure label at center of plate ----------
        # Calculate pressure
        H_m = float(params.bridge_height) / 1000.0
        z_centroid_m = H_m / 2.0
        vp = _velocity_pressure_kpa(
            wind_speed_ms=float(params.wind_speed_ms),
            z_ref_m=z_centroid_m,
            exposure=str(params.exposure_category),
            kzt=float(params.topographic_factor_kzt),
            kd=float(params.directionality_factor_kd),
            iw=1.0,
        )
        qz_kpa = vp["qz_kpa"]
        G = float(params.gust_effect_factor_g)
        p_kpa = qz_kpa * G

        # Center of the plate, moved up
        center_x = (x0 + x1) / 2.0
        center_z = z1 + (z1 - z0) * 0.1  # Above top edge
        label_point = Point(center_x, side_y, center_z)

        pressure_label = Label(
            label_point,
            f"p = {p_kpa:.2f} kPa",
            size_factor=1.0,
            color=vkt.Color(0, 0, 0)
        )
        # -------------------------------------------------------------------------------

        return vkt.GeometryResult(geometry=sections_group, labels=[pressure_label])    

    def download_json_data(self, params, **kwargs):
        # Unit conversions (mm -> m)
        L_m = float(params.bridge_length) / 1000.0
        B_m = float(params.bridge_width) / 1000.0
        H_m = float(params.bridge_height) / 1000.0

        # Roof geometry (kept because you already had it; not used for z_ref in open-truss method)
        theta_deg = float(params.roof_pitch_angle)
        rise_m = (B_m / 2.0) * math.tan(math.radians(theta_deg))
        eave_m = H_m
        ridge_m = eave_m + rise_m

        # Wind direction (normal to side face, L × H)
        wind_dir = _wind_direction()

        # Build model once to get Af and centroid height (z_ref)
        beam = RectangularTrussBeam(
            length=L_m,
            width=B_m,
            height=H_m,
            n_diagonals=int(params.n_divisions),
        )
        nodes, lines, chord_tl_ids, chord_tr_ids = beam.build()
        nodes, lines = beam.clean_model()
        nodes, lines = beam.remove_top_edge_nodes(chord_tl_ids, chord_tr_ids)

        # Member projected width: use HSS outer size as projected width
        member_proj_width_m = float(params.cross_section.replace("HSS", "").split("×")[0]) / 1000.0

        # Projected solid area Af and centroid height z_ref (centroid at H/2)
        proj = _projected_area_and_centroid_height(
            nodes=nodes,
            lines=lines,
            member_proj_width_m=member_proj_width_m,
            wind_dir_unit=wind_dir,
        )

        Af_m2 = proj["Af_total_m2"]
        z_centroid_m = H_m / 2.0

        # Gross area and solidity ratio
        Ag_m2 = _gross_area(L_m, H_m)
        epsilon = (Af_m2 / Ag_m2) if Ag_m2 > 0.0 else 0.0

        # Velocity pressure qz (Iw = 1.0)
        vp = _velocity_pressure_kpa(
            wind_speed_ms=float(params.wind_speed_ms),
            z_ref_m=z_centroid_m,
            exposure=str(params.exposure_category),
            kzt=float(params.topographic_factor_kzt),
            kd=float(params.directionality_factor_kd),
            iw=1.0,
        )
        kz = vp["kz"]
        qz_kpa = vp["qz_kpa"]

        # Design pressure term including gust effect (p = qz * G)
        G = float(params.gust_effect_factor_g)
        p_kpa = qz_kpa * G

        # Global force: F = (qz*G) * Cf * Af = p * Cf * Af
        Cf = float(params.force_coefficient_cf)
        F_kN = p_kpa * Cf * Af_m2  # since 1 kPa = 1 kN/m^2

        # Reference line load distributed along length
        w_ref_kN_per_m = (F_kN / L_m) if L_m > 0.0 else 0.0

        # Panel point approximation
        n_panels = max(int(params.n_divisions), 1)
        panel_length_m = (L_m / n_panels) if n_panels > 0 else 0.0
        P_int_kN = w_ref_kN_per_m * panel_length_m
        P_end_kN = 0.5 * P_int_kN

        data = {
            # Inputs
            "risk_category": str(params.risk_category),
            "site_elevation_m": float(params.site_elevation_m),

            "bridge_length_mm": float(params.bridge_length),
            "bridge_width_mm": float(params.bridge_width),
            "bridge_height_mm": float(params.bridge_height),
            "n_divisions": int(params.n_divisions),
            "cross_section": str(params.cross_section),

            "exposure_category": str(params.exposure_category),
            "wind_speed_ms": float(params.wind_speed_ms),
            "kzt": float(params.topographic_factor_kzt),
            "kd": float(params.directionality_factor_kd),
            "g": float(params.gust_effect_factor_g),

            "Af_members_m2": round(proj["Af_members_m2"], 4),
            "Af_total_m2": round(Af_m2, 4),
            "Ag_m2": round(Ag_m2, 4),
            "solidity_ratio_epsilon": round(epsilon, 4),


            # Wind pressures
            "velocity_pressure_coeff": round(vp["coeff"], 4),
            "kz": round(kz, 4),
            "qz_kpa": round(qz_kpa, 4),
            "p_kpa": round(p_kpa, 4),

            # Lattice force
            "cf": round(Cf, 4),
        }

        json_string = json.dumps(data, indent=2)
        json_file = vkt.File.from_data(json_string)
        return vkt.DownloadResult(json_file, "wind_load_data.json")