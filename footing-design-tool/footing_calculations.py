"""
Footing Design Calculation Helper Module

This module contains all the iteration and structural check logic
for concrete footing design according to ACI 318 / NSR-10.
"""

import math
from typing import Annotated
from collections.abc import Callable


def create_bearing_capacity_interpolator(
    bearing_table: Annotated[
        list[dict], "Table with 'depth' (m) and 'bearing_capacity' (kPa) entries"
    ],
) -> Annotated[
    Callable[[float], float],
    "Function that interpolates bearing capacity (kPa) for given depth (m)",
]:
    depth_bearing_pairs = []
    for row in bearing_table:
        depth = float(row.get("depth", 0) or 0)
        bc = float(row.get("bearing_capacity", 0) or 0)
        if depth > 0 and bc > 0:
            depth_bearing_pairs.append((depth, bc))

    # Sort by depth ascending
    depth_bearing_pairs.sort(key=lambda x: x[0])

    def get_bearing_capacity(foundation_depth: float) -> float:
        """Get interpolated bearing capacity based on foundation depth."""
        if not depth_bearing_pairs:
            return 150.0  # Default fallback
        if foundation_depth <= depth_bearing_pairs[0][0]:
            return depth_bearing_pairs[0][1]
        if foundation_depth >= depth_bearing_pairs[-1][0]:
            return depth_bearing_pairs[-1][1]

        # Linear interpolation between points
        for i in range(len(depth_bearing_pairs) - 1):
            d1, bc1 = depth_bearing_pairs[i]
            d2, bc2 = depth_bearing_pairs[i + 1]
            if d1 <= foundation_depth <= d2:
                ratio = (foundation_depth - d1) / (d2 - d1)
                return bc1 + ratio * (bc2 - bc1)

        return depth_bearing_pairs[-1][1]

    return get_bearing_capacity


def check_punching_shear(
    fz_total: Annotated[float, "Total vertical load (kN)"],
    B: Annotated[float, "Footing width (m)"],
    L: Annotated[float, "Footing length (m)"],
    b1: Annotated[float, "Pedestal width (m)"],
    b2: Annotated[float, "Pedestal length (m)"],
    d: Annotated[float, "Effective depth (m)"],
    fc: Annotated[float, "Concrete compressive strength (MPa)"],
) -> tuple[
    Annotated[bool, "True if shear capacity check passes"],
    Annotated[float, "Applied punching shear force Vu (kN)"],
    Annotated[float, "Punching shear capacity Vc (kN)"],
]:
    """
    Check two-way (punching) shear capacity according to NSR-10 C.11.11.2.1.

    Returns:
        Tuple of (passes_check, Vu_punch, Vc_punch)
    """
    bo = 2 * (b1 + d) + 2 * (b2 + d)
    Ao = (b1 + d) * (b2 + d)
    sigma_u = fz_total / (B * L)
    Vu_punch = sigma_u * (B * L - Ao)

    # Three capacity equations
    Vc1 = 0.75 * 0.33 * math.sqrt(fc) * bo * d * 1000
    beta = max(b1, b2) / min(b1, b2) if min(b1, b2) > 0 else 1.0
    Vc2 = 0.75 * 0.17 * (1 + 2 / beta) * math.sqrt(fc) * bo * d * 1000
    Vc3 = 0.75 * 0.083 * (40 * d / bo + 2) * math.sqrt(fc) * bo * d * 1000

    Vc_punch = min(Vc1, Vc2, Vc3)

    return abs(Vu_punch) <= Vc_punch, Vu_punch, Vc_punch


def check_one_way_shear(
    fz_total: Annotated[float, "Total vertical load (kN)"],
    B: Annotated[float, "Footing width (m)"],
    L: Annotated[float, "Footing length (m)"],
    b1: Annotated[float, "Pedestal width (m)"],
    b2: Annotated[float, "Pedestal length (m)"],
    d: Annotated[float, "Effective depth (m)"],
    fc: Annotated[float, "Concrete compressive strength (MPa)"],
) -> tuple[
    Annotated[bool, "True if one-way shear check passes"],
    Annotated[float, "Maximum shear stress (kPa)"],
    Annotated[float, "Allowable shear stress (kPa)"],
]:
    """
    Check one-way (beam) shear capacity according to NSR-10 C.11.11.1.1.

    Returns:
        Tuple of (passes_check, max_sigma_V, Vc_stress)
    """
    sigma_u = fz_total / (B * L)
    Vc_stress = 0.75 * math.sqrt(fc) / 6 * 1000  # kPa

    # Check in both directions
    Vu_B = sigma_u * B * (L / 2 - b2 / 2 - d)
    sigma_V_B = Vu_B / (B * d) if (B * d) > 0 else 0

    Vu_L = sigma_u * L * (B / 2 - b1 / 2 - d)
    sigma_V_L = Vu_L / (L * d) if (L * d) > 0 else 0

    max_sigma_V = max(abs(sigma_V_B), abs(sigma_V_L))

    return max_sigma_V <= Vc_stress, max_sigma_V, Vc_stress


def check_bearing_pressure(
    fz_total: Annotated[float, "Total vertical load (kN)"],
    B: Annotated[float, "Footing width (m)"],
    L: Annotated[float, "Footing length (m)"],
    ex: Annotated[float, "Eccentricity in X direction (m)"],
    ey: Annotated[float, "Eccentricity in Y direction (m)"],
    bearing_capacity: Annotated[float, "Allowable bearing capacity (kPa)"],
) -> tuple[
    Annotated[bool, "True if bearing pressure check passes"],
    Annotated[float, "Maximum bearing pressure (kPa)"],
    Annotated[float, "Effective footing width (m)"],
    Annotated[float, "Effective footing length (m)"],
]:
    """
    Check bearing pressure using Meyerhof effective area method.

    Returns:
        Tuple of (passes_check, sigma_max, B_eff, L_eff)
    """
    # Check kern condition: B >= 6*ex, L >= 6*ey
    if B < 6 * ex or L < 6 * ey:
        return False, 0, 0, 0

    # Calculate effective dimensions
    B_eff = B - 2 * ex
    L_eff = L - 2 * ey

    if B_eff <= 0 or L_eff <= 0:
        return False, 0, 0, 0

    A_eff = B_eff * L_eff
    sigma_max = fz_total / A_eff

    return sigma_max <= bearing_capacity, sigma_max, B_eff, L_eff


def calculate_eccentricities(
    load_combo: Annotated[
        dict, "Load combo dict with F1, F2, F3, M1, M2 reaction forces (kN, kN·m)"
    ],
    footing_weight: Annotated[float, "Weight of footing concrete (kN)"],
    fill_weight: Annotated[float, "Weight of fill material (kN)"],
    pedestal_weight: Annotated[float, "Weight of pedestal concrete (kN)"],
    pedestal_height: Annotated[float, "Height of pedestal (m)"],
    h: Annotated[float, "Footing thickness (m)"],
) -> tuple[
    Annotated[float, "Total vertical load at footing base (kN)"],
    Annotated[float, "Eccentricity in X direction (m)"],
    Annotated[float, "Eccentricity in Y direction (m)"],
    Annotated[float, "Moment about X-axis at base (kN·m)"],
    Annotated[float, "Moment about Y-axis at base (kN·m)"],
]:
    """
    Calculate eccentricities and moments at footing base.

    Returns:
        Tuple of (fz_total, ex, ey, mx_base, my_base)
    """
    fz_reaction = abs(load_combo["F3"])
    fx = load_combo["F1"]
    fy = load_combo["F2"]
    mx_reaction = abs(load_combo["M1"])
    my_reaction = abs(load_combo["M2"])

    # Total vertical load at base
    fz_total = fz_reaction + footing_weight + fill_weight + pedestal_weight

    # Transfer moments to base of footing
    lever_arm = pedestal_height + h / 2
    my_base = my_reaction + abs(fx) * lever_arm
    mx_base = mx_reaction + abs(fy) * lever_arm

    # Calculate eccentricities
    ex = my_base / fz_total if fz_total > 0 else 0
    ey = mx_base / fz_total if fz_total > 0 else 0

    return fz_total, ex, ey, mx_base, my_base


def check_design_for_all_load_combos(
    load_combos: Annotated[list[dict], "List of load combination dictionaries"],
    B: Annotated[float, "Footing width (m)"],
    L: Annotated[float, "Footing length (m)"],
    h: Annotated[float, "Footing thickness (m)"],
    d: Annotated[float, "Effective depth (m)"],
    pedestal_b: Annotated[float, "Pedestal width (m)"],
    pedestal_h: Annotated[float, "Pedestal length (m)"],
    pedestal_height: Annotated[float, "Pedestal height (m)"],
    footing_weight: Annotated[float, "Weight of footing (kN)"],
    fill_weight: Annotated[float, "Weight of fill material (kN)"],
    pedestal_weight: Annotated[float, "Weight of pedestal (kN)"],
    bearing_capacity: Annotated[float, "Allowable bearing capacity (kPa)"],
    fc: Annotated[float, "Concrete compressive strength (MPa)"],
) -> tuple[
    Annotated[bool, "True if all load combos pass all checks"],
    Annotated[dict | None, "Governing load combo results dict or None if failed"],
]:
    """
    Check if a design geometry passes all structural checks for all load combinations.

    Returns:
        Tuple of (all_pass, governing_results_dict or None)
    """
    max_sigma = 0
    governing_combo = None
    governing_results = {}

    for lc in load_combos:
        fz_reaction = abs(lc["F3"])
        if fz_reaction < 0.001:
            continue  # Skip zero load combos

        # Calculate eccentricities
        fz_total, ex, ey, mx_base, my_base = calculate_eccentricities(
            lc, footing_weight, fill_weight, pedestal_weight, pedestal_height, h
        )

        # Check bearing pressure
        bearing_pass, sigma_max, B_eff, L_eff = check_bearing_pressure(
            fz_total, B, L, ex, ey, bearing_capacity
        )
        if not bearing_pass:
            return False, None

        # Check punching shear
        punch_pass, Vu_punch, Vc_punch = check_punching_shear(
            fz_total, B, L, pedestal_b, pedestal_h, d, fc
        )
        if not punch_pass:
            return False, None

        # Check one-way shear
        oneway_pass, max_sigma_V, Vc_stress = check_one_way_shear(
            fz_total, B, L, pedestal_b, pedestal_h, d, fc
        )
        if not oneway_pass:
            return False, None

        # Track governing (max) stress
        if sigma_max > max_sigma:
            max_sigma = sigma_max
            governing_combo = lc["load_combo"]
            governing_results = {
                "sigma_max": sigma_max,
                "ex": ex,
                "ey": ey,
                "mx_base": mx_base,
                "my_base": my_base,
                "Vu_punch": Vu_punch,
                "Vc_punch": Vc_punch,
                "load_combo": governing_combo,
            }

    return True, governing_results


def find_optimal_footing_design(
    load_combos: Annotated[list[dict], "List of load combination dictionaries"],
    fc: Annotated[float, "Concrete compressive strength (MPa)"],
    gamma_concrete: Annotated[float, "Concrete unit weight (kN/m³)"],
    gamma_fill: Annotated[float, "Fill material unit weight (kN/m³)"],
    bearing_capacity_func: Annotated[
        Callable[[float], float],
        "Function returning bearing capacity (kPa) for depth (m)",
    ],
    pedestal_sizes: Annotated[list[float], "Pedestal square dimensions to iterate (m)"],
    pedestal_heights: Annotated[list[float], "Pedestal heights to iterate (m)"],
    thickness_options: Annotated[list[float], "Footing thicknesses to iterate (m)"],
    footing_dims: Annotated[list[float], "Footing plan dimensions to iterate (m)"],
) -> Annotated[
    dict | None, "Optimal design parameters dict or None if no compliant design found"
]:
    """
    Find optimal footing design by iterating through all combinations.
    Returns design with minimum concrete weight.

    Returns:
        Dict with optimal design parameters or None if no compliant design found
    """
    compliant_designs = []

    for ped_size in pedestal_sizes:
        pedestal_b = ped_size
        pedestal_h = ped_size

        for pedestal_height in pedestal_heights:
            pedestal_volume = pedestal_b * pedestal_h * pedestal_height
            pedestal_weight = pedestal_volume * gamma_concrete

            for h in thickness_options:
                d = h - 0.090  # Effective depth (90mm cover)

                # Calculate foundation depth and get bearing capacity
                foundation_depth = pedestal_height + h
                bearing_capacity = bearing_capacity_func(foundation_depth)

                for B in footing_dims:
                    # Skip if B is smaller than pedestal + clearance
                    if B < pedestal_b + 0.2:
                        continue

                    for L in footing_dims:
                        # Skip if L is smaller than pedestal + clearance
                        if L < pedestal_h + 0.2:
                            continue

                        # Calculate weights
                        footing_volume = B * L * h
                        footing_weight = footing_volume * gamma_concrete
                        fill_volume = max(
                            0, (B * L * pedestal_height) - pedestal_volume
                        )
                        fill_weight = fill_volume * gamma_fill

                        # Check all load combos for this geometry
                        all_pass, governing_results = check_design_for_all_load_combos(
                            load_combos,
                            B,
                            L,
                            h,
                            d,
                            pedestal_b,
                            pedestal_h,
                            pedestal_height,
                            footing_weight,
                            fill_weight,
                            pedestal_weight,
                            bearing_capacity,
                            fc,
                        )

                        if not all_pass:
                            continue

                        # Design is compliant for all load combos
                        total_weight = footing_weight + fill_weight + pedestal_weight
                        concrete_weight = footing_weight + pedestal_weight

                        compliant_designs.append(
                            {
                                "pedestal_size": ped_size,
                                "pedestal_height": pedestal_height,
                                "h": h,
                                "B": B,
                                "L": L,
                                "d": d,
                                "footing_weight": footing_weight,
                                "fill_weight": fill_weight,
                                "pedestal_weight": pedestal_weight,
                                "total_weight": total_weight,
                                "concrete_weight": concrete_weight,
                                "foundation_depth": foundation_depth,
                                "bearing_capacity": bearing_capacity,
                                "governing_combo": governing_results.get("load_combo"),
                                "sigma_max": governing_results.get("sigma_max", 0),
                                "ex": governing_results.get("ex", 0),
                                "ey": governing_results.get("ey", 0),
                                "mx_base": governing_results.get("mx_base", 0),
                                "my_base": governing_results.get("my_base", 0),
                                "Vu_punch": governing_results.get("Vu_punch", 0),
                                "Vc_punch": governing_results.get("Vc_punch", 0),
                            }
                        )

    # Return optimal design (minimum concrete weight)
    if compliant_designs:
        return min(compliant_designs, key=lambda x: x["concrete_weight"])
    return None


def get_top_n_designs(
    load_combos: Annotated[list[dict], "List of load combination dictionaries"],
    fc: Annotated[float, "Concrete compressive strength (MPa)"],
    gamma_concrete: Annotated[float, "Concrete unit weight (kN/m³)"],
    gamma_fill: Annotated[float, "Fill material unit weight (kN/m³)"],
    bearing_capacity_func: Annotated[
        Callable[[float], float],
        "Function returning bearing capacity (kPa) for depth (m)",
    ],
    pedestal_sizes: Annotated[list[float], "Pedestal square dimensions to iterate (m)"],
    pedestal_heights: Annotated[list[float], "Pedestal heights to iterate (m)"],
    thickness_options: Annotated[list[float], "Footing thicknesses to iterate (m)"],
    footing_dims: Annotated[list[float], "Footing plan dimensions to iterate (m)"],
    n: Annotated[int, "Number of top designs to return"] = 5,
) -> Annotated[
    list[dict], "List of design dicts sorted by concrete weight (ascending)"
]:
    """
    Get top N designs sorted by concrete weight.

    Returns:
        List of design dicts sorted by concrete weight (ascending)
    """
    compliant_designs = []

    for ped_size in pedestal_sizes:
        pedestal_b = ped_size
        pedestal_h = ped_size

        for pedestal_height in pedestal_heights:
            pedestal_volume = pedestal_b * pedestal_h * pedestal_height
            pedestal_weight = pedestal_volume * gamma_concrete

            for h in thickness_options:
                d = h - 0.090
                foundation_depth = pedestal_height + h
                bearing_capacity = bearing_capacity_func(foundation_depth)

                for B in footing_dims:
                    if B < pedestal_b + 0.2:
                        continue

                    for L in footing_dims:
                        if L < pedestal_h + 0.2:
                            continue

                        footing_volume = B * L * h
                        footing_weight = footing_volume * gamma_concrete
                        fill_volume = max(
                            0, (B * L * pedestal_height) - pedestal_volume
                        )
                        fill_weight = fill_volume * gamma_fill

                        all_pass, governing_results = check_design_for_all_load_combos(
                            load_combos,
                            B,
                            L,
                            h,
                            d,
                            pedestal_b,
                            pedestal_h,
                            pedestal_height,
                            footing_weight,
                            fill_weight,
                            pedestal_weight,
                            bearing_capacity,
                            fc,
                        )

                        if not all_pass:
                            continue

                        total_weight = footing_weight + fill_weight + pedestal_weight
                        concrete_weight = footing_weight + pedestal_weight

                        compliant_designs.append(
                            {
                                "pedestal_size": ped_size,
                                "pedestal_height": pedestal_height,
                                "h": h,
                                "B": B,
                                "L": L,
                                "d": d,
                                "footing_weight": footing_weight,
                                "fill_weight": fill_weight,
                                "pedestal_weight": pedestal_weight,
                                "total_weight": total_weight,
                                "concrete_weight": concrete_weight,
                                "foundation_depth": foundation_depth,
                                "bearing_capacity": bearing_capacity,
                                "governing_combo": governing_results.get("load_combo"),
                                "sigma_max": governing_results.get("sigma_max", 0),
                            }
                        )

    # Sort by concrete weight and return top N
    sorted_designs = sorted(compliant_designs, key=lambda x: x["concrete_weight"])
    return sorted_designs[:n]
