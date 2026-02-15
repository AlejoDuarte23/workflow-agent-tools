import viktor as vkt
import math


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


# ==============================================
# PARAMETRIZATION CLASS
# ==============================================

class Parametrization(vkt.Parametrization):

    section_intro = vkt.Section("Introduction")
    section_intro.intro = vkt.Text("""
## Multi-Node Concrete Footing Design Tool (ACI 318-19)
Checks:
- Foundation weights (slab + pedestal + optional fill)
- Factored actions at footing slab level (centroid)
- Bearing pressure (full-contact / partial-contact effective area)
- Two-way shear (punching)
- One-way shear
- Flexure (required As)
""")

    section_geometry = vkt.Section("Node Geometry & Dimensions")
    section_geometry.node_geometry = vkt.Table(
        "Footing and Pedestal Geometry",
        default=[
            {"node_name": "N1", "x": 0.0, "y": 0.0, "z": 0.0,
             "B": 2.2, "L": 2.4, "H": 0.6, "b1": 0.4, "b2": 0.5, "ph": 1.0},
        ],
    )
    section_geometry.node_geometry.node_name = vkt.TextField("Node Name")
    section_geometry.node_geometry.x = vkt.NumberField("X", num_decimals=2, suffix="m")
    section_geometry.node_geometry.y = vkt.NumberField("Y", num_decimals=2, suffix="m")
    section_geometry.node_geometry.z = vkt.NumberField("Z", num_decimals=2, suffix="m")
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

    section_soil = vkt.Section("Soil Properties")
    section_soil.qa = vkt.NumberField("Allowable Bearing Capacity (q_allow)", default=190.0, min=0.0, suffix="kPa", num_decimals=1)


# ==============================================
# CONTROLLER CLASS
# ==============================================

class Controller(vkt.Controller):
    parametrization = Parametrization

    def create_geometry_lookup(self, params):
        geometry_by_node = {}
        for row in params.section_geometry.node_geometry:
            geometry_by_node[row["node_name"]] = {
                "x": row["x"], "y": row["y"], "z": row["z"],
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

        # Bearing pressure
        br = lc_result["bearing"]
        br_status = vkt.DataStatus.SUCCESS if br["passes"] else vkt.DataStatus.ERROR
        lc_group.add(vkt.DataItem(
            "Bearing Check",
            f"{br['case']}: qmax={br['qmax']:.1f} kPa ≤ {lc_result['q_allow']:.1f} kPa",
            status=br_status,
            status_message="PASS" if br["passes"] else "FAIL"
        ))

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

        q_allow = params.section_soil.qa  # kPa

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

                bearing = bearing_pressure_check_full_partial(
                    factored["Fz_footing"],
                    geometry["B"], geometry["L"],
                    factored["ex"], factored["ey"],
                    q_allow_kPa=q_allow,
                )

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

                combo_pass = bearing["passes"] and punch["passes"] and oneway["passes"]

                lc_result = {
                    "case_name": lc["case_name"],
                    "weights": weights,
                    "factored_actions": factored,
                    "bearing": bearing,
                    "q_allow": q_allow,
                    "punching_shear": punch,
                    "one_way_shear": oneway,
                    "flexure": flex,
                    "overall_pass": combo_pass,
                }

                main_group.add(self.build_node_loadcase_item(node_name, lc["case_name"], geometry, d, lc_result))

        return vkt.DataResult(main_group)