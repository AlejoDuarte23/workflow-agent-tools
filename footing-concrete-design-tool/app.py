import viktor as vkt
import math
import report_template


# ==============================================
# HELPER CALCULATION FUNCTIONS (ACI 318-19)
# ==============================================

def calculate_foundation_weights(B, L, H, b1, b2, ph, gamma_concrete, gamma_fill):
    """
    Fill definition (your spreadsheet):
      V_fill = (B*L*ph) - (b1*b2*ph)
    i.e., fill sits on the footing plan area up to height ph, excluding pedestal footprint.
    """
    slab_weight = B * L * H * gamma_concrete
    pedestal_weight = b1 * b2 * ph * gamma_concrete
    fill_vol = (B * L - b1 * b2) * ph
    fill_weight = fill_vol * gamma_fill
    total_weight = slab_weight + pedestal_weight + fill_weight
    return {
        "slab_weight": slab_weight,
        "pedestal_weight": pedestal_weight,
        "fill_weight": fill_weight,
        "total_weight": total_weight,
        "fill_vol": fill_vol,
    }


def calculate_effective_depth(H, cover, db):
    """H in m, cover/db in mm -> d in m."""
    return H - cover / 1000.0 - db / 2000.0


def calculate_factored_actions(loads, total_weight, ph, H):
    """
    Transfer loads from pedestal top to footing slab level (centroid).
    Input convention:
      F3 negative = compression (down)
      Foundation self-weight increases compression
    Returns:
      Fx_footing, Fy_footing (kN)
      Fz_footing (kN, negative compression)
      Mx_footing (= M1) (kN·m)
      My_footing (= M2) (kN·m)
      ex, ey (m), computed as My/|P|, Mx/|P|
    """
    transfer_distance = ph + H / 2.0

    # Axial at footing level (compression stays negative)
    Fz_footing = loads["F3"] - total_weight

    # Horizontal forces (kept, but you can ignore in bearing if desired)
    Fx_footing = loads["F1"]
    Fy_footing = loads["F2"]

    # Moments at footing level
    # M1 about X, increased by Fy*lever arm
    # M2 about Y, increased by Fx*lever arm
    Mx_footing = loads["M1"] + Fy_footing * transfer_distance
    My_footing = loads["M2"] + Fx_footing * transfer_distance

    P = abs(Fz_footing)
    if P > 1e-6:
        ex = My_footing / P
        ey = Mx_footing / P
    else:
        ex = 0.0
        ey = 0.0

    return {
        "Fx_footing": Fx_footing,
        "Fy_footing": Fy_footing,
        "Fz_footing": Fz_footing,
        "Mx_footing": Mx_footing,
        "My_footing": My_footing,
        "ex": ex,
        "ey": ey,
    }


def bearing_pressure_check_full_partial(Fz_footing, B, L, ex, ey, q_allow_kPa):
    """
    Bearing pressure check at footing level using:
      - Full contact linear corner pressure if |ex|<=L/6 and |ey|<=B/6
      - Partial contact effective area if uplift would occur
    Units:
      Fz_footing in kN (negative compression), B,L,ex,ey in m, q_allow in kPa (=kN/m²)
    Returns:
      case: "Full contact" / "Partial contact"
      qmax, qmin in kPa
      A_eff in m² (only for partial contact)
      passes (qmax <= q_allow)
    """
    P = abs(Fz_footing)  # kN
    if P < 1e-9:
        return {
            "case": "No axial load",
            "qmax": 0.0,
            "qmin": 0.0,
            "A_eff": B * L,
            "passes": True,
            "full_contact": True,
        }

    full_contact = (abs(ex) <= L / 6.0) and (abs(ey) <= B / 6.0)

    if full_contact:
        q0 = P / (B * L)  # kPa since kN/m²
        # Corner pressures: q = q0*(1 ± 6ex/L ± 6ey/B)
        kx = 6.0 * ex / L
        ky = 6.0 * ey / B

        q1 = q0 * (1.0 + kx + ky)
        q2 = q0 * (1.0 + kx - ky)
        q3 = q0 * (1.0 - kx + ky)
        q4 = q0 * (1.0 - kx - ky)

        qmax = max(q1, q2, q3, q4)
        qmin = min(q1, q2, q3, q4)

        return {
            "case": "Full contact",
            "qmax": qmax,
            "qmin": qmin,
            "A_eff": B * L,
            "passes": qmax <= q_allow_kPa,
            "full_contact": True,
            "q_corners": (q1, q2, q3, q4),
        }

    # Partial contact (effective area)
    L_eff = L - 2.0 * abs(ex)
    B_eff = B - 2.0 * abs(ey)
    A_eff = L_eff * B_eff

    if L_eff <= 0.0 or B_eff <= 0.0 or A_eff <= 1e-9:
        # Extreme eccentricity -> no compression area (physically unstable)
        return {
            "case": "Partial contact (invalid effective area)",
            "qmax": float("inf"),
            "qmin": 0.0,
            "A_eff": max(0.0, A_eff),
            "passes": False,
            "full_contact": False,
        }

    q_eff = P / A_eff  # kPa
    return {
        "case": "Partial contact",
        "qmax": q_eff,
        "qmin": 0.0,
        "A_eff": A_eff,
        "passes": q_eff <= q_allow_kPa,
        "full_contact": False,
    }


def check_punching_shear(Fz_footing, B, L, b1, b2, d, fc):
    """
    Two-way (punching) shear check (as in your original approach).
    Returns kN capacities.
    """
    b0 = 2.0 * (b1 + d) + 2.0 * (b2 + d)
    lambda_s = min(1.0, math.sqrt(2.0 / (1.0 + d * 1000.0 / 254.0)))

    Vu = abs(Fz_footing / (B * L)) * (B * L - (b1 + d) * (b2 + d))

    Vc1 = 0.75 * 0.33 * lambda_s * math.sqrt(fc) * b0 * d * 1000.0

    beta = max(b1, b2) / min(b1, b2)
    Vc2 = 0.75 * 0.17 * (1.0 + 2.0 / beta) * lambda_s * math.sqrt(fc) * b0 * d * 1000.0

    alpha_s = 40.0
    Vc3 = 0.75 * 0.083 * (2.0 + alpha_s * d / b0) * lambda_s * math.sqrt(fc) * b0 * d * 1000.0

    Vc_min = min(Vc1, Vc2, Vc3)
    passes = Vu <= Vc_min

    return {
        "passes": passes,
        "Vu": Vu,
        "Vc_min": Vc_min,
        "Vc1": Vc1,
        "Vc2": Vc2,
        "Vc3": Vc3,
        "b0": b0,
        "lambda_s": lambda_s,
    }


def check_one_way_shear(Fz_footing, B, L, b1, b2, d, fc):
    """
    One-way (beam) shear check with direction-specific bw.
    """
    sigma_u = abs(Fz_footing) / (B * L)  # kPa

    Vu_x = sigma_u * B * max(0.0, (L / 2.0 - b2 / 2.0 - d))
    Vu_y = sigma_u * L * max(0.0, (B / 2.0 - b1 / 2.0 - d))

    phi = 0.75
    d_mm = d * 1000.0
    bw_x_mm = B * 1000.0
    bw_y_mm = L * 1000.0

    Vc_x = phi * 0.17 * math.sqrt(fc) * bw_x_mm * d_mm / 1000.0  # kN
    Vc_y = phi * 0.17 * math.sqrt(fc) * bw_y_mm * d_mm / 1000.0  # kN

    util_x = Vu_x / Vc_x if Vc_x > 0 else 0.0
    util_y = Vu_y / Vc_y if Vc_y > 0 else 0.0

    passes = (Vu_x <= Vc_x) and (Vu_y <= Vc_y)

    return {
        "passes": passes,
        "sigma_u": sigma_u,
        "Vu_x": Vu_x,
        "Vu_y": Vu_y,
        "Vc_x": Vc_x,
        "Vc_y": Vc_y,
        "util_x": util_x,
        "util_y": util_y,
    }


def _As_required_rect_section(Mu_kNm, b_m, d_m, fc_MPa, fy_MPa, phi=0.9):
    """
    Solve As from phi*Mn >= Mu for singly reinforced rectangular strip.
    Units:
      Mu_kNm: kN·m
      b_m, d_m: m
      fc, fy: MPa (N/mm²)
    Returns:
      (As_mm2 or None, discriminant)
    """
    b = b_m * 1000.0
    d = d_m * 1000.0
    Mu = abs(Mu_kNm) * 1e6  # N·mm

    fc = fc_MPa
    fy = fy_MPa

    A = (fy ** 2) / (2.0 * 0.85 * fc * b)
    B = -fy * d
    C = Mu / phi

    disc = B * B - 4.0 * A * C
    if disc <= 0.0:
        return None, disc

    As1 = (-B - math.sqrt(disc)) / (2.0 * A)
    As2 = (-B + math.sqrt(disc)) / (2.0 * A)

    As_pos = [x for x in (As1, As2) if x > 0.0]
    if not As_pos:
        return None, disc

    return min(As_pos), disc


def calculate_required_rebar(Fz_footing, B, L, b1, b2, d, H, fc, fy):
    """
    Strip method moments at face of pedestal + unit-consistent As solution.
    """
    sigma_u = abs(Fz_footing) / (B * L)  # kPa

    wu_x = sigma_u * L  # kN/m
    wu_y = sigma_u * B  # kN/m

    Mu_x = wu_x * ((L / 2.0 - b2 / 2.0) ** 2) / 2.0  # kN·m
    Mu_y = wu_y * ((B / 2.0 - b1 / 2.0) ** 2) / 2.0  # kN·m

    # Match your sheet intent: use b = L for x-dir strip and b = B for y-dir strip
    As_calc_x, disc_x = _As_required_rect_section(Mu_x, b_m=L, d_m=d, fc_MPa=fc, fy_MPa=fy, phi=0.9)
    As_calc_y, disc_y = _As_required_rect_section(Mu_y, b_m=B, d_m=d, fc_MPa=fc, fy_MPa=fy, phi=0.9)

    As_min_x = 0.0018 * (L * 1000.0) * (H * 1000.0)
    As_min_y = 0.0018 * (B * 1000.0) * (H * 1000.0)

    As_req_x = max(As_calc_x, As_min_x) if As_calc_x is not None else As_min_x
    As_req_y = max(As_calc_y, As_min_y) if As_calc_y is not None else As_min_y

    return {
        "As_req_x": As_req_x,
        "As_req_y": As_req_y,
        "Mu_x": Mu_x,
        "Mu_y": Mu_y,
        "As_min_x": As_min_x,
        "As_min_y": As_min_y,
        "disc_x": disc_x,
        "disc_y": disc_y,
        "flexure_solution_real_x": As_calc_x is not None,
        "flexure_solution_real_y": As_calc_y is not None,
    }


def calculate_rebar_spacing_strip_method(B, L, H, cover_mm, db_mm, As_req_x, As_req_y):
    """
    Strip method spacing (matches your screenshots):

    n = ceil(As_target / Ab)
    s_clear = ((strip_len*1000) - 2*cc - n*db) / (n-1)
    s_c2c = s_clear + db

    Where:
      As_target = max(As_min, As_required) is already handled upstream by As_req_x / As_req_y
      X-dir uses strip length = L
      Y-dir uses strip length = B

    Returns spacing in mm and #bars for each direction.
    """
    # bar area (mm²)
    Ab = math.pi * (db_mm ** 2) / 4.0

    # helper
    def _dir(As_target_mm2, strip_len_m):
        n = max(2, math.ceil(As_target_mm2 / Ab))  # at least 2 bars to define spacing
        strip_len_mm = strip_len_m * 1000.0

        s_clear = (strip_len_mm - 2.0 * cover_mm - n * db_mm) / (n - 1)
        s_c2c = s_clear + db_mm
        return n, s_clear, s_c2c

    n_x, s_clear_x, s_c2c_x = _dir(As_req_x, L)  # X-dir along B -> distributed across L
    n_y, s_clear_y, s_c2c_y = _dir(As_req_y, B)  # Y-dir along L -> distributed across B

    return {
        "Ab": Ab,
        "n_x": n_x,
        "s_clear_x": s_clear_x,
        "s_c2c_x": s_c2c_x,
        "n_y": n_y,
        "s_clear_y": s_clear_y,
        "s_c2c_y": s_c2c_y,
    }


# ==============================================
# PARAMETRIZATION CLASS
# ==============================================

class Parametrization(vkt.Parametrization):

    section_intro = vkt.Section("Introduction")
    section_intro.intro = vkt.Text("""
## Multi-Node Concrete Footing Design Tool (ACI 318-19)
Checks foundation weights, factored actions, two-way shear (punching), one-way shear, flexure, and rebar spacing.

### Effective Depth
$$d = H - \\text{cover} - \\frac{d_b}{2}$$

### Two-Way Shear (Punching) - ACI 22.6
$$b_0 = 2(b_1 + d) + 2(b_2 + d)$$
$$\\phi V_c = 0.75 \\times \\min\\left(0.33\\lambda_s\\sqrt{f'_c}, 0.17(1+\\frac{2}{\\beta})\\lambda_s\\sqrt{f'_c}, 0.083(2+\\frac{\\alpha_s d}{b_0})\\lambda_s\\sqrt{f'_c}\\right) b_0 d$$

### One-Way Shear (Beam) - ACI 22.5
$$V_{u,x} = \\sigma_u B\\left(\\frac{L}{2} - \\frac{b_2}{2} - d\\right), \\quad \\phi V_{c,x} = 0.75 \\times 0.17\\sqrt{f'_c} b_w d$$

### Flexure - Strip Method
$$M_u = \\frac{w_u}{2}\\left(\\frac{L}{2} - \\frac{b_2}{2}\\right)^2$$
$$\\phi M_n = \\phi A_s f_y\\left(d - \\frac{a}{2}\\right), \\quad a = \\frac{A_s f_y}{0.85f'_c b}$$
$$A_{s,\\text{req}} = \\max\\left(A_{s,\\text{calc}}, A_{s,\\min}\\right), \\quad A_{s,\\min} = 0.0018bh$$

### Rebar Spacing
$$n = \\lceil\\frac{A_s}{A_b}\\rceil, \\quad s_{\\text{c/c}} = \\frac{L_{\\text{strip}} - 2\\times\\text{cover} - n d_b}{n-1} + d_b$$
""")

    section_nodes = vkt.Section("Node Coordinates")
    section_nodes.node_coordinates = vkt.Table(
        "Node Locations",
        default=[
            {"node_name": "N1", "x": 0.0, "y": 0.0, "z": 0.0},
            {"node_name": "N2", "x": 5.0, "y": 0.0, "z": 0.0},
            {"node_name": "N3", "x": 10.0, "y": 0.0, "z": 0.0},
            {"node_name": "N4", "x": 0.0, "y": 5.0, "z": 0.0},
        ],
    )
    section_nodes.node_coordinates.node_name = vkt.TextField("Node Name")
    section_nodes.node_coordinates.x = vkt.NumberField("X", num_decimals=2, suffix="m")
    section_nodes.node_coordinates.y = vkt.NumberField("Y", num_decimals=2, suffix="m")
    section_nodes.node_coordinates.z = vkt.NumberField("Z", num_decimals=2, suffix="m")

    section_geometry = vkt.Section("Footing & Pedestal Dimensions")
    section_geometry.node_geometry = vkt.Table(
        "Footing and Pedestal Geometry",
        default=[
            {"node_name": "N1", "B": 2.2, "L": 2.4, "H": 0.6, "b1": 0.4, "b2": 0.5, "ph": 1.0},
            {"node_name": "N2", "B": 2.2, "L": 2.4, "H": 0.6, "b1": 0.4, "b2": 0.5, "ph": 1.0},
            {"node_name": "N3", "B": 2.2, "L": 2.4, "H": 0.6, "b1": 0.4, "b2": 0.5, "ph": 1.0},
            {"node_name": "N4", "B": 2.2, "L": 2.4, "H": 0.6, "b1": 0.4, "b2": 0.5, "ph": 1.0},
        ],
    )
    section_geometry.node_geometry.node_name = vkt.TextField("Node Name")
    section_geometry.node_geometry.B = vkt.NumberField("Footing Width (B)", num_decimals=2, suffix="m")
    section_geometry.node_geometry.L = vkt.NumberField("Footing Length (L)", num_decimals=2, suffix="m")
    section_geometry.node_geometry.H = vkt.NumberField("Footing Thickness (H)", num_decimals=2, suffix="m")
    section_geometry.node_geometry.b1 = vkt.NumberField("Pedestal Width (b1)", num_decimals=3, suffix="m")
    section_geometry.node_geometry.b2 = vkt.NumberField("Pedestal Length (b2)", num_decimals=3, suffix="m")
    section_geometry.node_geometry.ph = vkt.NumberField("Pedestal Height (ph)", num_decimals=2, suffix="m")

    load_cases_section = vkt.Section("Load Cases")
    load_cases_section.load_cases = vkt.Table(
        "Node Reactions & Load Combinations",
        default=[
            {"case_name": "LC1", "node_name": "N1",
             "F1": 3.0, "F2": 2.0, "F3": -1750.0,
             "M1": 100.0, "M2": 100.0, "M3": 0.0},
            {"case_name": "LC2", "node_name": "N2",
             "F1": 3.0, "F2": 2.0, "F3": -1700.0,
             "M1": 80.0, "M2": 60.0, "M3": 0.0},
            {"case_name": "LC3", "node_name": "N3",
             "F1": 3.0, "F2": 2.0, "F3": -1700.0,
             "M1": 80.0, "M2": 60.0, "M3": 0.0},
            {"case_name": "LC4", "node_name": "N4",
             "F1": 3.0, "F2": 2.0, "F3": -1700.0,
             "M1": 80.0, "M2": 60.0, "M3": 0.0},
        ],
    )
    load_cases_section.load_cases.case_name = vkt.TextField("Load Case Name")
    load_cases_section.load_cases.node_name = vkt.TextField("Node Name")
    load_cases_section.load_cases.F1 = vkt.NumberField("F1 (Longitudinal)", suffix="kN", num_decimals=2)
    load_cases_section.load_cases.F2 = vkt.NumberField("F2 (Transverse)", suffix="kN", num_decimals=2)
    load_cases_section.load_cases.F3 = vkt.NumberField("F3 (Axial)", suffix="kN", num_decimals=2)
    load_cases_section.load_cases.M1 = vkt.NumberField("M1 (About X)", suffix="kN·m", num_decimals=2)
    load_cases_section.load_cases.M2 = vkt.NumberField("M2 (About Y)", suffix="kN·m", num_decimals=2)
    load_cases_section.load_cases.M3 = vkt.NumberField("M3 (Torsion)", suffix="kN·m", num_decimals=2)

    section_concrete = vkt.Section("Concrete Properties")
    section_concrete.gamma_concrete = vkt.NumberField("Concrete Unit Weight (γ_c)", default=24.0, min=0, suffix="kN/m³", num_decimals=1)
    section_concrete.fc = vkt.NumberField("Concrete Strength (f'c)", default=28.0, min=0, suffix="MPa", num_decimals=1)
    section_concrete.fy = vkt.NumberField("Steel Yield Strength (fy)", default=420.0, min=0, suffix="MPa", num_decimals=1)
    section_concrete.gamma_fill = vkt.NumberField("Fill Unit Weight (γ_fill)", default=19.5, min=0, suffix="kN/m³", num_decimals=1)
    section_concrete.cover = vkt.NumberField("Concrete Cover", default=90, min=0, suffix="mm", num_decimals=0)
    section_concrete.db = vkt.NumberField("Rebar Diameter (db)", default=12, min=0, suffix="mm", num_decimals=0)



# ==============================================
# CONTROLLER CLASS
# ==============================================

class Controller(vkt.Controller):
    parametrization = Parametrization

    def create_geometry_lookup(self, params):
        # Build coordinate lookup
        coords_by_node = {}
        for row in params.section_nodes.node_coordinates:
            coords_by_node[row["node_name"]] = {
                "x": row["x"], "y": row["y"], "z": row["z"],
            }

        # Build geometry lookup, merging with coordinates
        geometry_by_node = {}
        for row in params.section_geometry.node_geometry:
            node_name = row["node_name"]
            coords = coords_by_node.get(node_name, {"x": 0.0, "y": 0.0, "z": 0.0})
            geometry_by_node[node_name] = {
                "x": coords["x"], "y": coords["y"], "z": coords["z"],
                "B": row["B"], "L": row["L"], "H": row["H"],
                "b1": row["b1"], "b2": row["b2"], "ph": row["ph"],
            }
        return geometry_by_node

    def create_loads_lookup(self, params):
        loads_by_node = {}
        for row in params.load_cases_section.load_cases:
            loads_by_node.setdefault(row["node_name"], []).append({
                "case_name": row["case_name"],
                "F1": row["F1"], "F2": row["F2"], "F3": row["F3"],
                "M1": row["M1"], "M2": row["M2"], "M3": row["M3"],
            })
        return loads_by_node

    def build_node_loadcase_item(self, node_name, case_name, geometry, d, lc_result):
        """Build a flattened data item for a single node-loadcase combination."""
        lc_group = vkt.DataGroup()

        # Geometry info
        g = geometry
        lc_group.add(vkt.DataItem("Geometry", f"Footing {g['B']:.2f}×{g['L']:.2f}×{g['H']:.2f}m, Pedestal {g['b1']:.3f}×{g['b2']:.3f}×{g['ph']:.2f}m, d={d:.3f}m"))

        # Foundation weights (compact)
        w = lc_result["weights"]
        lc_group.add(vkt.DataItem("Total Weight", w["total_weight"], suffix="kN", number_of_decimals=1))
        lc_group.add(vkt.DataItem("  (Slab+Pedestal+Fill)", f"{w['slab_weight']:.1f} + {w['pedestal_weight']:.1f} + {w['fill_weight']:.1f} kN"))

        # Factored actions (key values only)
        fa = lc_result["factored_actions"]
        lc_group.add(vkt.DataItem("Fz (Axial)", abs(fa["Fz_footing"]), suffix="kN", number_of_decimals=1))
        lc_group.add(vkt.DataItem("Eccentricity", f"ex={fa['ex']:.4f}m, ey={fa['ey']:.4f}m"))
        lc_group.add(vkt.DataItem("Moments", f"Mx={fa['Mx_footing']:.1f} kN·m, My={fa['My_footing']:.1f} kN·m"))

        # Punching shear
        pu = lc_result["punching_shear"]
        pu_status = vkt.DataStatus.SUCCESS if pu["passes"] else vkt.DataStatus.ERROR
        util_pu = pu["Vu"] / pu["Vc_min"] if pu["Vc_min"] > 0 else 0.0
        lc_group.add(vkt.DataItem(
            "Punching Shear",
            f"Vu={pu['Vu']:.1f} kN ≤ Vc={pu['Vc_min']:.1f} kN (Util={util_pu:.2f})",
            status=pu_status,
            status_message="PASS" if pu["passes"] else "FAIL"
        ))

        # One-way shear
        ow = lc_result["one_way_shear"]
        ow_status = vkt.DataStatus.SUCCESS if ow["passes"] else vkt.DataStatus.ERROR
        lc_group.add(vkt.DataItem(
            "One-Way Shear X",
            f"Vu={ow['Vu_x']:.1f} kN ≤ Vc={ow['Vc_x']:.1f} kN (Util={ow['util_x']:.2f})",
            status=ow_status if ow['Vu_x'] > ow['Vc_x'] else vkt.DataStatus.SUCCESS,
            status_message="PASS" if ow['Vu_x'] <= ow['Vc_x'] else "FAIL"
        ))
        lc_group.add(vkt.DataItem(
            "One-Way Shear Y",
            f"Vu={ow['Vu_y']:.1f} kN ≤ Vc={ow['Vc_y']:.1f} kN (Util={ow['util_y']:.2f})",
            status=ow_status if ow['Vu_y'] > ow['Vc_y'] else vkt.DataStatus.SUCCESS,
            status_message="PASS" if ow['Vu_y'] <= ow['Vc_y'] else "FAIL"
        ))

        # Flexure
        fx = lc_result["flexure"]
        lc_group.add(vkt.DataItem("Flexure X-dir", f"As_req={fx['As_req_x']:.0f} mm² (Mu={fx['Mu_x']:.1f} kN·m)"))
        lc_group.add(vkt.DataItem("Flexure Y-dir", f"As_req={fx['As_req_y']:.0f} mm² (Mu={fx['Mu_y']:.1f} kN·m)"))

        # Rebar Spacing
        sp = lc_result["spacing"]
        lc_group.add(vkt.DataItem("Rebar Spacing X-dir", f"n={sp['n_x']} bars @ {sp['s_c2c_x']:.0f}mm c/c (clear={sp['s_clear_x']:.0f}mm)"))
        lc_group.add(vkt.DataItem("Rebar Spacing Y-dir", f"n={sp['n_y']} bars @ {sp['s_c2c_y']:.0f}mm c/c (clear={sp['s_clear_y']:.0f}mm)"))

        overall_status = vkt.DataStatus.SUCCESS if lc_result["overall_pass"] else vkt.DataStatus.ERROR
        status_msg = "All checks PASS" if lc_result["overall_pass"] else "Some checks FAIL"

        return vkt.DataItem(
            f"{node_name} - {case_name}",
            subgroup=lc_group,
            status=overall_status,
            status_message=status_msg
        )

    @vkt.DataView("Design Check Results", duration_guess=5)
    def view_design_results(self, params, **kwargs):

        geometry_by_node = self.create_geometry_lookup(params)
        loads_by_node = self.create_loads_lookup(params)

        fc = params.section_concrete.fc
        fy = params.section_concrete.fy
        gamma_concrete = params.section_concrete.gamma_concrete
        gamma_fill = params.section_concrete.gamma_fill
        cover = params.section_concrete.cover
        db = params.section_concrete.db

        main_group = vkt.DataGroup()

        # Flatten structure: each node-loadcase becomes a top-level item
        for node_name, geometry in geometry_by_node.items():
            if node_name not in loads_by_node:
                warning_group = vkt.DataGroup()
                warning_group.add(vkt.DataItem("Status", "No load cases defined"))
                main_group.add(vkt.DataItem(node_name, subgroup=warning_group, status=vkt.DataStatus.WARNING))
                continue

            d = calculate_effective_depth(geometry["H"], cover, db)

            weights = calculate_foundation_weights(
                geometry["B"], geometry["L"], geometry["H"],
                geometry["b1"], geometry["b2"], geometry["ph"],
                gamma_concrete, gamma_fill
            )

            for lc in loads_by_node[node_name]:
                if abs(lc["F3"]) < 1e-9:
                    continue

                factored = calculate_factored_actions(lc, weights["total_weight"], geometry["ph"], geometry["H"])

                punch = check_punching_shear(
                    factored["Fz_footing"], geometry["B"], geometry["L"],
                    geometry["b1"], geometry["b2"], d, fc,
                )

                oneway = check_one_way_shear(
                    factored["Fz_footing"], geometry["B"], geometry["L"],
                    geometry["b1"], geometry["b2"], d, fc,
                )

                flex = calculate_required_rebar(
                    factored["Fz_footing"], geometry["B"], geometry["L"],
                    geometry["b1"], geometry["b2"], d, geometry["H"], fc, fy,
                )

                spacing = calculate_rebar_spacing_strip_method(
                    geometry["B"], geometry["L"], geometry["H"],
                    cover_mm=cover, db_mm=db,
                    As_req_x=flex["As_req_x"], As_req_y=flex["As_req_y"],
                )

                combo_pass = punch["passes"] and oneway["passes"]

                lc_result = {
                    "case_name": lc["case_name"],
                    "weights": weights,
                    "factored_actions": factored,
                    "punching_shear": punch,
                    "one_way_shear": oneway,
                    "flexure": flex,
                    "spacing": spacing,
                    "overall_pass": combo_pass,
                }

                main_group.add(self.build_node_loadcase_item(node_name, lc["case_name"], geometry, d, lc_result))

        return vkt.DataResult(main_group)

    @vkt.WebView("Calculation Report", duration_guess=10)
    def view_calculation_report(self, params, **kwargs):
        """Display Mathcad-like calculation report with equations and results"""

        geometry_by_node = self.create_geometry_lookup(params)
        loads_by_node = self.create_loads_lookup(params)

        fc = params.section_concrete.fc
        fy = params.section_concrete.fy
        gamma_concrete = params.section_concrete.gamma_concrete
        gamma_fill = params.section_concrete.gamma_fill
        cover = params.section_concrete.cover
        db = params.section_concrete.db

        # Store results for each node-loadcase combination
        results_by_node_lc = {}

        # Process all node-loadcase combinations
        for node_name, geometry in geometry_by_node.items():
            if node_name not in loads_by_node:
                continue

            d = calculate_effective_depth(geometry["H"], cover, db)

            weights = calculate_foundation_weights(
                geometry["B"], geometry["L"], geometry["H"],
                geometry["b1"], geometry["b2"], geometry["ph"],
                gamma_concrete, gamma_fill
            )

            for lc in loads_by_node[node_name]:
                if abs(lc["F3"]) < 1e-9:
                    continue

                factored = calculate_factored_actions(lc, weights["total_weight"], geometry["ph"], geometry["H"])

                punch = check_punching_shear(
                    factored["Fz_footing"], geometry["B"], geometry["L"],
                    geometry["b1"], geometry["b2"], d, fc,
                )

                oneway = check_one_way_shear(
                    factored["Fz_footing"], geometry["B"], geometry["L"],
                    geometry["b1"], geometry["b2"], d, fc,
                )

                flex = calculate_required_rebar(
                    factored["Fz_footing"], geometry["B"], geometry["L"],
                    geometry["b1"], geometry["b2"], d, geometry["H"], fc, fy,
                )

                spacing = calculate_rebar_spacing_strip_method(
                    geometry["B"], geometry["L"], geometry["H"],
                    cover_mm=cover, db_mm=db,
                    As_req_x=flex["As_req_x"], As_req_y=flex["As_req_y"],
                )

                combo_pass = punch["passes"] and oneway["passes"]

                results_by_node_lc[(node_name, lc["case_name"])] = {
                    "weights": weights,
                    "factored_actions": factored,
                    "punching_shear": punch,
                    "one_way_shear": oneway,
                    "flexure": flex,
                    "spacing": spacing,
                    "overall_pass": combo_pass,
                }

        # Build HTML report
        html = report_template.get_report_header()
        html += report_template.format_design_parameters(fc, fy, gamma_concrete, gamma_fill, cover, db)
        html += report_template.get_design_equations()
        html += report_template.format_node_geometry_table(geometry_by_node)
        html += report_template.format_load_cases_table(loads_by_node)
        html += report_template.format_design_results(results_by_node_lc)
        html += report_template.get_report_footer()

        return vkt.WebResult(html=html)

    @vkt.PlotlyView("Footing Plan View - Critical Zones", duration_guess=5)
    def view_footing_plan(self, params, **kwargs):
        """2D plan view showing footings with critical shear zones highlighted"""
        import plotly.graph_objects as go

        geometry_by_node = self.create_geometry_lookup(params)
        loads_by_node = self.create_loads_lookup(params)

        fc = params.section_concrete.fc
        fy = params.section_concrete.fy
        gamma_concrete = params.section_concrete.gamma_concrete
        gamma_fill = params.section_concrete.gamma_fill
        cover = params.section_concrete.cover
        db = params.section_concrete.db

        # Create Plotly figure
        fig = go.Figure()

        # Colors
        footing_color = "rgba(200, 200, 200, 0.5)"
        pedestal_color = "rgba(80, 80, 80, 0.8)"
        punching_pass_color = "rgba(76, 175, 80, 0.3)"
        punching_fail_color = "rgba(244, 67, 54, 0.3)"
        oneway_color = "rgba(33, 150, 243, 0.2)"

        # Track bounds for layout
        all_x = []
        all_y = []

        for node_name, geometry in geometry_by_node.items():
            if node_name not in loads_by_node or not loads_by_node[node_name]:
                continue

            # Get node position
            cx = geometry['x']
            cy = geometry['y']

            B = geometry['B']
            L = geometry['L']
            H = geometry['H']
            b1 = geometry['b1']
            b2 = geometry['b2']
            ph = geometry['ph']

            d = calculate_effective_depth(H, cover, db)

            # Calculate weights and checks for worst case (first load case)
            lc = loads_by_node[node_name][0]
            weights = calculate_foundation_weights(B, L, H, b1, b2, ph, gamma_concrete, gamma_fill)
            factored = calculate_factored_actions(lc, weights["total_weight"], ph, H)
            punch = check_punching_shear(factored["Fz_footing"], B, L, b1, b2, d, fc)
            oneway = check_one_way_shear(factored["Fz_footing"], B, L, b1, b2, d, fc)

            # Footing outline
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
                    text=f"{node_name}<br>Footing: {B:.2f}m × {L:.2f}m × {H:.2f}m<br>d = {d:.3f}m",
                    showlegend=False,
                )
            )

            # Punching shear critical perimeter (at d from pedestal face)
            punch_color = punching_pass_color if punch["passes"] else punching_fail_color
            b1_crit = b1 + 2 * d
            b2_crit = b2 + 2 * d
            px0, px1 = cx - b1_crit / 2, cx + b1_crit / 2
            py0, py1 = cy - b2_crit / 2, cy + b2_crit / 2

            fig.add_trace(
                go.Scatter(
                    x=[px0, px1, px1, px0, px0],
                    y=[py0, py0, py1, py1, py0],
                    mode="lines",
                    fill="toself",
                    fillcolor=punch_color,
                    line=dict(color="rgba(200,0,0,0.8)" if not punch["passes"] else "rgba(0,150,0,0.8)",
                              width=2, dash="dash"),
                    name=f"{node_name} Punching Zone",
                    hoverinfo="text",
                    text=f"{node_name} Punching Shear<br>b₀ = {punch['b0']:.2f}m<br>Vu = {punch['Vu']:.1f}kN<br>φVc = {punch['Vc_min']:.1f}kN<br>Status: {'PASS' if punch['passes'] else 'FAIL'}",
                    showlegend=False,
                )
            )

            # One-way shear critical sections (at d from pedestal face)
            # X-direction (vertical lines)
            x_crit_left = cx - b2 / 2 - d
            x_crit_right = cx + b2 / 2 + d
            if abs(x_crit_left - x0) > 0.01:  # Only show if there's space
                fig.add_trace(
                    go.Scatter(
                        x=[x_crit_left, x_crit_left],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color="rgba(33,150,243,0.6)", width=2, dash="dot"),
                        name=f"{node_name} One-Way X",
                        hoverinfo="text",
                        text=f"{node_name} One-Way Shear X<br>Vu = {oneway['Vu_x']:.1f}kN<br>φVc = {oneway['Vc_x']:.1f}kN<br>Status: {'PASS' if oneway['Vu_x'] <= oneway['Vc_x'] else 'FAIL'}",
                        showlegend=False,
                    )
                )
            if abs(x_crit_right - x1) > 0.01:
                fig.add_trace(
                    go.Scatter(
                        x=[x_crit_right, x_crit_right],
                        y=[y0, y1],
                        mode="lines",
                        line=dict(color="rgba(33,150,243,0.6)", width=2, dash="dot"),
                        name=f"{node_name} One-Way X",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            # Y-direction (horizontal lines)
            y_crit_bottom = cy - b1 / 2 - d
            y_crit_top = cy + b1 / 2 + d
            if abs(y_crit_bottom - y0) > 0.01:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y_crit_bottom, y_crit_bottom],
                        mode="lines",
                        line=dict(color="rgba(33,150,243,0.6)", width=2, dash="dot"),
                        name=f"{node_name} One-Way Y",
                        hoverinfo="text",
                        text=f"{node_name} One-Way Shear Y<br>Vu = {oneway['Vu_y']:.1f}kN<br>φVc = {oneway['Vc_y']:.1f}kN<br>Status: {'PASS' if oneway['Vu_y'] <= oneway['Vc_y'] else 'FAIL'}",
                        showlegend=False,
                    )
                )
            if abs(y_crit_top - y1) > 0.01:
                fig.add_trace(
                    go.Scatter(
                        x=[x0, x1],
                        y=[y_crit_top, y_crit_top],
                        mode="lines",
                        line=dict(color="rgba(33,150,243,0.6)", width=2, dash="dot"),
                        name=f"{node_name} One-Way Y",
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

            # Pedestal rectangle
            ped_x0, ped_x1 = cx - b1 / 2, cx + b1 / 2
            ped_y0, ped_y1 = cy - b2 / 2, cy + b2 / 2
            fig.add_trace(
                go.Scatter(
                    x=[ped_x0, ped_x1, ped_x1, ped_x0, ped_x0],
                    y=[ped_y0, ped_y0, ped_y1, ped_y1, ped_y0],
                    mode="lines",
                    fill="toself",
                    fillcolor=pedestal_color,
                    line=dict(color="rgba(50,50,50,1)", width=2),
                    name=f"{node_name} Pedestal",
                    hoverinfo="text",
                    text=f"{node_name}<br>Pedestal: {b1:.3f}m × {b2:.3f}m × {ph:.2f}m",
                    showlegend=False,
                )
            )

            # Add node label
            fig.add_annotation(
                x=cx,
                y=cy,
                text=f"<b>{node_name}</b>",
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(50,50,50,0.8)",
                borderpad=4,
            )

            all_x.extend([x0, x1])
            all_y.extend([y0, y1])

        # Calculate plot bounds
        if all_x and all_y:
            margin = 1.0
            x_range = [min(all_x) - margin, max(all_x) + margin]
            y_range = [min(all_y) - margin, max(all_y) + margin]
        else:
            x_range = [-5, 10]
            y_range = [-5, 10]

        # Add legend traces (dummy traces for legend only)
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=footing_color, line=dict(color="rgba(100,100,100,1)", width=2)),
                name="Footing Outline",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="markers",
                marker=dict(size=10, color=pedestal_color, line=dict(color="rgba(50,50,50,1)", width=2)),
                name="Pedestal",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="rgba(0,150,0,0.8)", width=2, dash="dash"),
                name="Punching Critical Perimeter (Pass)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="rgba(200,0,0,0.8)", width=2, dash="dash"),
                name="Punching Critical Perimeter (Fail)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None],
                mode="lines",
                line=dict(color="rgba(33,150,243,0.6)", width=2, dash="dot"),
                name="One-Way Shear Critical Section",
            )
        )

        # Layout
        fig.update_layout(
            title="Footing Plan View - Critical Shear Zones (ACI 318-19)",
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
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="rgba(0,0,0,0.3)",
                borderwidth=1,
            ),
        )

        return vkt.PlotlyResult(fig.to_json())