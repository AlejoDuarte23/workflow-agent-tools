import viktor as vkt
import math
import numpy as np


def calculate_bearing_pressure(F1, F2, F3, M1, M2, M3,
                             B, L, h,  # footing dimensions
                             pedestal_b, pedestal_h, pedestal_height,
                             gamma_concrete=24.0, gamma_fill=18.0):
    """
    Calculate bearing pressure for eccentric footings
    
    Parameters:
    F1, F2, F3: Forces at top of pedestal (kN)
    M1, M2, M3: Moments at top of pedestal (kN-m)
    B, L, h: Footing width, length, and thickness (m)
    pedestal_b, pedestal_h, pedestal_height: Pedestal dimensions (m)
    gamma_concrete, gamma_fill: Unit weights (kN/m³)
    
    Returns:
    Dictionary with bearing pressures and eccentricities
    """
    
    # Calculate foundation weights
    pedestal_volume = pedestal_b * pedestal_h * pedestal_height
    footing_volume = B * L * h
    fill_volume = (B * L - pedestal_b * pedestal_h) * pedestal_height
    
    pedestal_weight = pedestal_volume * gamma_concrete
    footing_weight = footing_volume * gamma_concrete
    fill_weight = fill_volume * gamma_fill
    
    total_foundation_weight = pedestal_weight + footing_weight + fill_weight
    
    # Transfer loads to neutral axis of footing (bottom of slab)
    transfer_distance = pedestal_height + h/2
    
    # Updated loads at neutral axis (F3 is vertical, negative means compression)
    F3_na = F3 - total_foundation_weight  # F3 is negative, foundation weight increases compression
    M1_na = M1 + F2 * transfer_distance
    M2_na = M2 + F1 * transfer_distance
    
    # Footing properties
    A = B * L  # Area
    Ix = B * L**3 / 12  # Moment of inertia about x-axis
    Iy = L * B**3 / 12  # Moment of inertia about y-axis
    Sx = Ix / (L/2)  # Section modulus about x-axis
    Sy = Iy / (B/2)  # Section modulus about y-axis
    
    # Calculate eccentricities (using absolute value of F3 for compression)
    F3_abs = abs(F3_na)
    ex = M2_na / F3_abs if F3_abs != 0 else 0  # Eccentricity in x-direction
    ey = M1_na / F3_abs if F3_abs != 0 else 0  # Eccentricity in y-direction
    
    # Kern boundaries
    kern_x = B / 6
    kern_y = L / 6

    # Determine if we have single or double eccentricity (threshold = 0.01m)
    ECCENTRICITY_THRESHOLD = 0.01
    has_ex = abs(ex) >= ECCENTRICITY_THRESHOLD
    has_ey = abs(ey) >= ECCENTRICITY_THRESHOLD

    # Calculate bearing pressures based on eccentricity case
    has_tension = False

    if has_ex and has_ey:
        # Two eccentricities case
        # CRITICAL: Must ensure L > 6*|ey| AND B > 6*|ex| to avoid tension

        if L > 6 * abs(ey) and B > 6 * abs(ex):
            # Footing dimensions are adequate - no tension
            # Use biaxial formula: q = P/(LB) * (1 ± 6ex/B ± 6ey/L)
            q_center = F3_abs / A

            # Corner pressures using biaxial formula
            # q = P/A * (1 ± 6ex/B ± 6ey/L)
            q1 = q_center * (1 + 6*ex/B + 6*ey/L)  # (+B/2, +L/2)
            q2 = q_center * (1 - 6*ex/B + 6*ey/L)  # (-B/2, +L/2)
            q3 = q_center * (1 + 6*ex/B - 6*ey/L)  # (+B/2, -L/2)
            q4 = q_center * (1 - 6*ex/B - 6*ey/L)  # (-B/2, -L/2)

            qmax = max(q1, q2, q3, q4)
            qmin = min(q1, q2, q3, q4)

            case_x = "within_kern"
            case_y = "within_kern"
        else:
            # Footing dimensions are insufficient - tension will occur
            # Mark this geometry as invalid
            has_tension = True
            qmax = float('inf')  # Set to infinity to reject this geometry
            qmin = 0
            case_x = "outside_kern" if B <= 6 * abs(ex) else "within_kern"
            case_y = "outside_kern" if L <= 6 * abs(ey) else "within_kern"
    else:
        # Single eccentricity case (or one is negligible < 0.01m)
        if has_ex:
            # Only ex is significant
            e = ex
            dimension = B
            case_x = "within_kern" if abs(ex) <= kern_x else "outside_kern"
            case_y = "within_kern"
        elif has_ey:
            # Only ey is significant
            e = ey
            dimension = L
            case_x = "within_kern"
            case_y = "within_kern" if abs(ey) <= kern_y else "outside_kern"
        else:
            # No significant eccentricity (concentric load)
            e = 0
            dimension = B
            case_x = "within_kern"
            case_y = "within_kern"

        # Single eccentricity formula: q = P/A * (1 ± 6e/dimension)
        q_center = F3_abs / A

        if abs(e) <= dimension / 6:
            # Within kern
            qmax = q_center * (1 + 6 * abs(e) / dimension)
            qmin = q_center * (1 - 6 * abs(e) / dimension)
        else:
            # Outside kern - use effective dimension
            dimension_eff = 3 * (dimension/2 - abs(e)) if (dimension/2 - abs(e)) > 0 else dimension/6

            if has_ex:
                A_eff = dimension_eff * L
            elif has_ey:
                A_eff = B * dimension_eff
            else:
                A_eff = A

            qmax = 2 * F3_abs / A_eff if A_eff > 0 else float('inf')
            qmin = 0
    
    # Calculate resultant eccentricity
    e_resultant = math.sqrt(ex**2 + ey**2)
    
    # Determine overall case description
    if case_x == "within_kern" and case_y == "within_kern":
        overall_case = "Within Kern (Full Compression)"
    else:
        overall_case = "Outside Kern (Partial Compression)"
    
    results = {
        'loads_at_neutral_axis': {
            'F3': F3_na,
            'M1': M1_na,
            'M2': M2_na
        },
        'foundation_weight': total_foundation_weight,
        'eccentricities': {
            'ex': ex,
            'ey': ey,
            'e_resultant': e_resultant
        },
        'kern_limits': {
            'kern_x': kern_x,
            'kern_y': kern_y
        },
        'bearing_pressures': {
            'qmax': qmax,
            'qmin': qmin,
            'q_average': F3_abs / A
        },
        'pressure_case': {
            'x_direction': case_x,
            'y_direction': case_y,
            'overall': overall_case
        },
        'footing_properties': {
            'area': A,
            'Sx': Sx,
            'Sy': Sy
        },
        'has_tension': has_tension
    }
    
    return results


def interpolate_bearing_capacity(depth, bearing_table):
    """
    Interpolate bearing capacity based on depth from the bearing capacity table
    
    Parameters:
    depth: Total foundation depth (m)
    bearing_table: List of dicts with 'depth' and 'bearing_capacity' keys
    
    Returns:
    Interpolated bearing capacity (kPa)
    """
    if not bearing_table or len(bearing_table) == 0:
        return 100.0  # Default value
    
    # Sort table by depth
    sorted_table = sorted(bearing_table, key=lambda x: x['depth'])
    
    depths = [row['depth'] for row in sorted_table]
    capacities = [row['bearing_capacity'] for row in sorted_table]
    
    # If depth is outside range, use nearest value
    if depth <= depths[0]:
        return capacities[0]
    if depth >= depths[-1]:
        return capacities[-1]
    
    # Linear interpolation
    return np.interp(depth, depths, capacities)


def optimize_footing_for_node(load_cases, min_footing_length, gamma_concrete, gamma_fill, bearing_table):
    """
    Find the optimal (lightest) footing geometry that satisfies bearing capacity for all load cases
    
    Parameters:
    load_cases: List of load case dictionaries
    min_footing_length: Minimum footing length to start optimization (m)
    gamma_concrete, gamma_fill: Unit weights (kN/m³)
    bearing_table: Bearing capacity vs depth table
    
    Returns:
    Dictionary with optimal geometry and governing load case
    """
    
    # Define design space
    L_values = [min_footing_length + i * 0.5 for i in range(20)]  # Up to min + 10m
    B_values = [min_footing_length + i * 0.5 for i in range(20)]  # Up to min + 10m
    h_values = [0.3, 0.4, 0.5]
    pedestal_sizes = [0.3, 0.4, 0.5]  # Max pedestal size: 0.5m
    pedestal_heights = [0.7, 1.0, 1.5, 2.0, 2.5]
    
    best_solution = None
    min_weight = float('inf')
    
    # Iterate through all combinations
    for L in L_values:
        for B in B_values:
            for h in h_values:
                for ped_size in pedestal_sizes:
                    for ped_height in pedestal_heights:
                        
                        # Calculate total depth for bearing capacity interpolation
                        total_depth = ped_height + h
                        allowable_bearing = interpolate_bearing_capacity(total_depth, bearing_table)
                        
                        # Check all load cases
                        all_cases_ok = True
                        max_pressure = 0
                        governing_case_idx = 0
                        
                        for idx, load_case in enumerate(load_cases):
                            result = calculate_bearing_pressure(
                                F1=load_case['F1'],
                                F2=load_case['F2'],
                                F3=load_case['F3'],
                                M1=load_case['M1'],
                                M2=load_case['M2'],
                                M3=load_case['M3'],
                                B=B,
                                L=L,
                                h=h,
                                pedestal_b=ped_size,
                                pedestal_h=ped_size,
                                pedestal_height=ped_height,
                                gamma_concrete=gamma_concrete,
                                gamma_fill=gamma_fill
                            )

                            # Check if footing has tension (invalid for two eccentricity case)
                            if result['has_tension']:
                                all_cases_ok = False
                                break

                            qmax = result['bearing_pressures']['qmax']

                            # Track maximum pressure and governing case
                            if qmax > max_pressure:
                                max_pressure = qmax
                                governing_case_idx = idx

                            # Check if bearing capacity is exceeded
                            if qmax > allowable_bearing:
                                all_cases_ok = False
                                break
                        
                        # If all cases pass, calculate weight
                        if all_cases_ok:
                            # Calculate total weight
                            pedestal_volume = ped_size * ped_size * ped_height
                            footing_volume = B * L * h
                            fill_volume = (B * L - ped_size * ped_size) * ped_height
                            
                            total_weight = (pedestal_volume + footing_volume) * gamma_concrete + fill_volume * gamma_fill
                            
                            # Update best solution if lighter
                            if total_weight < min_weight:
                                min_weight = total_weight
                                best_solution = {
                                    'B': B,
                                    'L': L,
                                    'h': h,
                                    'pedestal_size': ped_size,
                                    'pedestal_height': ped_height,
                                    'total_weight': total_weight,
                                    'governing_case_idx': governing_case_idx,
                                    'governing_case_name': load_cases[governing_case_idx]['name'],
                                    'max_pressure': max_pressure,
                                    'allowable_bearing': allowable_bearing,
                                    'total_depth': total_depth
                                }
    
    return best_solution


class Parametrization(vkt.Parametrization):

    # Introduction Section
    section_intro = vkt.Section("Introduction")
    section_intro.intro = vkt.Text("""
## Footing Sizing and Optimization Tool
Optimizes footing geometry to minimize weight while satisfying bearing capacity for all load combinations. Parametrization follows **footing-design-tool** convention.

### Foundation Weight
$$W_{\\text{total}} = (b_{\\text{ped}} h_{\\text{ped}} H_{\\text{ped}} + BLh)\\gamma_c + (BL - b_{\\text{ped}} h_{\\text{ped}})H_{\\text{ped}}\\gamma_f$$

### Eccentricities and Bearing Pressure
$$e_x = \\frac{M2_{\\text{NA}}}{|F3_{\\text{NA}}|}, \\quad e_y = \\frac{M1_{\\text{NA}}}{|F3_{\\text{NA}}|}$$

**Single Eccentricity** (one eccentricity < 0.01m):

**(a) Within Kern** $(e \\leq B/6)$:
$$q_{\\text{max}} = \\frac{P}{A}\\left(1 + \\frac{6e}{B}\\right), \\quad q_{\\text{min}} = \\frac{P}{A}\\left(1 - \\frac{6e}{B}\\right)$$

**(b) Outside Kern** $(e > B/6)$:
$$q_{\\text{max}} = \\frac{2P}{3(0.5B - e)L}, \\quad q_{\\text{min}} = 0$$

**Two Eccentricities** (both ≥ 0.01m):
$$q = \\frac{P}{LB}\\left(1 \\pm \\frac{6e_{x}}{B}\\pm \\frac{6e_{y}}{L}\\right)$$

**Note:** For two eccentricities, dimensions must satisfy $L > 6|e_y|$ and $B > 6|e_x|$ to avoid tension.

### Optimization
Minimize $W_{\\text{total}}$ subject to: $q_{\\text{max}} \\leq q_{\\text{allowable}}(H_{\\text{ped}} + h)$ for all load cases.
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
        {"case_name": "LC1", "node": "N1", "F1": 0.0, "F2": 0.0, "F3": -15.0, "M1": 10.0, "M2": 8.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N1", "F1": 5.0, "F2": 3.0, "F3": -18.0, "M1": 15.0, "M2": 12.0, "M3": 0.0},
        {"case_name": "LC1", "node": "N2", "F1": 0.0, "F2": 0.0, "F3": -20.0, "M1": 15.0, "M2": 12.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N2", "F1": 8.0, "F2": 4.0, "F3": -23.0, "M1": 20.0, "M2": 18.0, "M3": 0.0},
        {"case_name": "LC1", "node": "N3", "F1": 0.0, "F2": 0.0, "F3": -20.0, "M1": 15.0, "M2": 12.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N3", "F1": 8.0, "F2": 4.0, "F3": -23.0, "M1": 20.0, "M2": 18.0, "M3": 0.0},
        {"case_name": "LC1", "node": "N4", "F1": 0.0, "F2": 0.0, "F3": -15.0, "M1": 10.0, "M2": 8.0, "M3": 0.0},
        {"case_name": "LC2", "node": "N4", "F1": 5.0, "F2": 3.0, "F3": -18.0, "M1": 15.0, "M2": 12.0, "M3": 0.0},
    ])
    load_cases_section.load_cases.case_name = vkt.TextField("Load Case Name")
    load_cases_section.load_cases.node = vkt.TextField("Node Name")
    load_cases_section.load_cases.F1 = vkt.NumberField("F1", suffix="kN")
    load_cases_section.load_cases.F2 = vkt.NumberField("F2", suffix="kN")
    load_cases_section.load_cases.F3 = vkt.NumberField("F3", suffix="kN")
    load_cases_section.load_cases.M1 = vkt.NumberField("M1", suffix="kN-m")
    load_cases_section.load_cases.M2 = vkt.NumberField("M2", suffix="kN-m")
    load_cases_section.load_cases.M3 = vkt.NumberField("M3", suffix="kN-m")

    # Bearing capacity table
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
    section_bearing.bearing_table.bearing_capacity = vkt.NumberField("Bearing Capacity (kPa)", num_decimals=1)
    
    # Optimization settings
    optimization_section = vkt.Section("Optimization Settings")
    optimization_section.min_footing_length = vkt.NumberField("Minimum Footing Length", default=1.0, min=0.5, suffix="m")
    
    # Material properties
    material_section = vkt.Section("Material Properties")
    material_section.gamma_concrete = vkt.NumberField("Concrete Unit Weight", default=24.0, min=0, suffix="kN/m³")
    material_section.gamma_fill = vkt.NumberField("Fill Unit Weight", default=18.0, min=0, suffix="kN/m³")

    # Download section
    download_section = vkt.Section("Export Results")
    download_section.download_btn = vkt.DownloadButton("Download Optimization Results (JSON)", method="download_results")

class Controller(vkt.Controller):
    parametrization = Parametrization
    
    def download_results(self, params, **kwargs):
        """Download optimization results as JSON file"""
        import json
        from io import BytesIO
        
        # Group load cases by node
        nodes_dict = {}
        for node_row in params.nodes_section.nodes:
            node_name = node_row['node_name']
            nodes_dict[node_name] = []
        
        # Assign load cases to nodes
        for lc_row in params.load_cases_section.load_cases:
            node_name = lc_row['node']
            if node_name in nodes_dict:
                nodes_dict[node_name].append({
                    'name': lc_row['case_name'],
                    'F1': lc_row['F1'],
                    'F2': lc_row['F2'],
                    'F3': lc_row['F3'],
                    'M1': lc_row['M1'],
                    'M2': lc_row['M2'],
                    'M3': lc_row['M3']
                })
        
        # Run optimization for all nodes
        results_data = {}
        
        for node_name, load_cases in nodes_dict.items():
            if not load_cases:
                results_data[node_name] = {
                    'status': 'No load cases defined'
                }
                continue
            
            # Run optimization
            optimal = optimize_footing_for_node(
                load_cases=load_cases,
                min_footing_length=params.optimization_section.min_footing_length,
                gamma_concrete=params.material_section.gamma_concrete,
                gamma_fill=params.material_section.gamma_fill,
                bearing_table=params.section_bearing.bearing_table
            )
            
            if optimal is None:
                results_data[node_name] = {
                    'status': 'No feasible solution found'
                }
            else:
                # Get detailed results for governing load case
                governing_lc = load_cases[optimal['governing_case_idx']]
                
                # Calculate bearing pressure details for governing case
                bearing_result = calculate_bearing_pressure(
                    F1=governing_lc['F1'],
                    F2=governing_lc['F2'],
                    F3=governing_lc['F3'],
                    M1=governing_lc['M1'],
                    M2=governing_lc['M2'],
                    M3=governing_lc['M3'],
                    B=optimal['B'],
                    L=optimal['L'],
                    h=optimal['h'],
                    pedestal_b=optimal['pedestal_size'],
                    pedestal_h=optimal['pedestal_size'],
                    pedestal_height=optimal['pedestal_height'],
                    gamma_concrete=params.material_section.gamma_concrete,
                    gamma_fill=params.material_section.gamma_fill
                )
                
                results_data[node_name] = {
                    'footing_geometry': {
                        'length_L_m': optimal['L'],
                        'width_B_m': optimal['B'],
                        'slab_thickness_h_m': optimal['h'],
                        'pedestal_base_m': optimal['pedestal_size'],
                        'pedestal_height_m': optimal['pedestal_height'],
                        'total_depth_m': optimal['total_depth'],
                        'footing_area_m2': optimal['L'] * optimal['B']
                    },
                    'loads': {
                        'Fz_kN': governing_lc['F3']
                    },
                    'bearing_pressure': {
                        'qmax_kPa': bearing_result['bearing_pressures']['qmax']
                    }
                }
        
        # Convert to JSON
        json_str = json.dumps(results_data, indent=2)
        json_bytes = BytesIO(json_str.encode('utf-8'))
        
        return vkt.DownloadResult(vkt.File.from_data(json_bytes.getvalue()), 'footing_optimization_results.json')
    
    @vkt.PlotlyView("Footing Layout (2D)", duration_guess=5)
    def view_footing_layout(self, params, **kwargs):
        """2D plan view showing all footings and pedestals with optimal designs"""
        import plotly.graph_objects as go

        # Group load cases by node
        nodes_dict = {}
        for node_row in params.nodes_section.nodes:
            node_name = node_row['node_name']
            nodes_dict[node_name] = {
                'x': node_row['x'],
                'y': node_row['y'],
                'load_cases': []
            }

        # Assign load cases to nodes
        for lc_row in params.load_cases_section.load_cases:
            node_name = lc_row['node']
            if node_name in nodes_dict:
                nodes_dict[node_name]['load_cases'].append({
                    'name': lc_row['case_name'],
                    'F1': lc_row['F1'],
                    'F2': lc_row['F2'],
                    'F3': lc_row['F3'],
                    'M1': lc_row['M1'],
                    'M2': lc_row['M2'],
                    'M3': lc_row['M3']
                })

        # Run optimization for all nodes
        optimization_results = {}
        for node_name, node_data in nodes_dict.items():
            if node_data['load_cases']:
                optimal = optimize_footing_for_node(
                    load_cases=node_data['load_cases'],
                    min_footing_length=params.optimization_section.min_footing_length,
                    gamma_concrete=params.material_section.gamma_concrete,
                    gamma_fill=params.material_section.gamma_fill,
                    bearing_table=params.section_bearing.bearing_table
                )
                optimization_results[node_name] = optimal

        # Create Plotly figure
        fig = go.Figure()

        # Colors
        footing_color = "rgba(180, 180, 180, 0.6)"
        pedestal_color = "rgba(100, 100, 100, 0.8)"

        # Track bounds for layout
        all_x = []
        all_y = []

        for node_name, node_data in nodes_dict.items():
            cx = node_data['x']
            cy = node_data['y']

            if node_name in optimization_results and optimization_results[node_name]:
                opt = optimization_results[node_name]
                B = opt['B']
                L = opt['L']
                h = opt['h']
                ped = opt['pedestal_size']
                ped_h = opt['pedestal_height']

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
                        text=f"{node_name}<br>Footing: {B:.2f}m × {L:.2f}m<br>Thickness: {h * 1000:.0f}mm<br>Depth: {opt['total_depth'] * 1000:.0f}mm<br>Weight: {opt['total_weight']:.1f}kN",
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

    @vkt.WebView("Calculation Report", duration_guess=10)
    def view_calculation_report(self, params, **kwargs):
        """Display Mathcad-like calculation report with equations and results"""
        
        # Group load cases by node
        nodes_dict = {}
        for node_row in params.nodes_section.nodes:
            node_name = node_row['node_name']
            nodes_dict[node_name] = []
        
        # Assign load cases to nodes
        for lc_row in params.load_cases_section.load_cases:
            node_name = lc_row['node']
            if node_name in nodes_dict:
                nodes_dict[node_name].append({
                    'name': lc_row['case_name'],
                    'F1': lc_row['F1'],
                    'F2': lc_row['F2'],
                    'F3': lc_row['F3'],
                    'M1': lc_row['M1'],
                    'M2': lc_row['M2'],
                    'M3': lc_row['M3']
                })
        
        # Run optimization for all nodes
        optimization_results = {}
        for node_name, load_cases in nodes_dict.items():
            if load_cases:
                optimal = optimize_footing_for_node(
                    load_cases=load_cases,
                    min_footing_length=params.optimization_section.min_footing_length,
                    gamma_concrete=params.material_section.gamma_concrete,
                    gamma_fill=params.material_section.gamma_fill,
                    bearing_table=params.section_bearing.bearing_table
                )
                optimization_results[node_name] = optimal
        
        # Generate HTML report
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Footing Design Calculation Report</title>
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
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Multi-Node Footing Design Calculation Report</h1>
                
                <div class="section">
                    <h2>Design Parameters</h2>
                    <div class="input-params">
                        <p><strong>Material Properties:</strong></p>
                        <p>\\(\\gamma_{{\\text{{concrete}}}} = {params.material_section.gamma_concrete}\\) kN/m³</p>
                        <p>\\(\\gamma_{{\\text{{fill}}}} = {params.material_section.gamma_fill}\\) kN/m³</p>
                        <p><strong>Minimum Footing Length:</strong> {params.optimization_section.min_footing_length} m</p>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Allowable Bearing Capacity vs Depth</h2>
                    <table>
                        <tr>
                            <th>Depth (m)</th>
                            <th>Allowable Bearing Capacity (kPa)</th>
                        </tr>
        """
        
        # Add bearing capacity table
        for row in params.section_bearing.bearing_table:
            html += f"""
                        <tr>
                            <td>{row['depth']:.2f}</td>
                            <td>{row['bearing_capacity']:.1f}</td>
                        </tr>
            """
        
        html += """
                    </table>
                    <p class="description">Note: Bearing capacity is interpolated based on total foundation depth (pedestal height + slab thickness)</p>
                </div>
                
                <div class="section">
                    <h2>Design Equations</h2>
                    
                    <h3>Foundation Weight Calculation</h3>
                    <div class="equation-block">
                        <div class="equation">$$V_{\\text{pedestal}} = b_{\\text{ped}} \\times h_{\\text{ped}} \\times H_{\\text{ped}}$$</div>
                        <div class="equation">$$V_{\\text{footing}} = B \\times L \\times h$$</div>
                        <div class="equation">$$V_{\\text{fill}} = (B \\times L - b_{\\text{ped}} \\times h_{\\text{ped}}) \\times H_{\\text{ped}}$$</div>
                        <div class="equation">$$W_{\\text{foundation}} = (V_{\\text{pedestal}} + V_{\\text{footing}}) \\times \\gamma_{\\text{concrete}} + V_{\\text{fill}} \\times \\gamma_{\\text{fill}}$$</div>
                    </div>
                    
                    <h3>Load Transfer to Neutral Axis</h3>
                    <div class="equation-block">
                        <div class="equation">$$d_{\\text{transfer}} = H_{\\text{ped}} + \\frac{{h}}{{2}}$$</div>
                        <div class="equation">$$F3_{\\text{NA}} = F3 - W_{\\text{foundation}}$$</div>
                        <div class="equation">$$M1_{\\text{NA}} = M1 + F2 \\times d_{\\text{transfer}}$$</div>
                        <div class="equation">$$M2_{\\text{NA}} = M2 + F1 \\times d_{\\text{transfer}}$$</div>
                    </div>
                    
                    <h3>Footing Properties</h3>
                    <div class="equation-block">
                        <div class="equation">$$A = B \\times L$$</div>
                        <div class="equation">$$I_x = \\frac{{B \\times L^3}}{{12}}$$</div>
                        <div class="equation">$$I_y = \\frac{{L \\times B^3}}{{12}}$$</div>
                        <div class="equation">$$S_x = \\frac{{I_x}}{{L/2}} = \\frac{{B \\times L^2}}{{6}}$$</div>
                        <div class="equation">$$S_y = \\frac{{I_y}}{{B/2}} = \\frac{{L \\times B^2}}{{6}}$$</div>
                    </div>
                    
                    <h3>Eccentricity Calculation</h3>
                    <div class="equation-block">
                        <div class="equation">$$e_x = \\frac{{M2_{\\text{NA}}}}{{|F3_{\\text{NA}}|}}$$</div>
                        <div class="equation">$$e_y = \\frac{{M1_{\\text{NA}}}}{{|F3_{\\text{NA}}|}}$$</div>
                        <div class="equation">$$e_{\\text{resultant}} = \\sqrt{{e_x^2 + e_y^2}}$$</div>
                    </div>
                    
                    <h3>Kern Limits</h3>
                    <div class="equation-block">
                        <div class="equation">$$\\text{Kern}_x = \\frac{{B}}{{6}}$$</div>
                        <div class="equation">$$\\text{Kern}_y = \\frac{{L}}{{6}}$$</div>
                        <div class="description">If eccentricity is within kern limits, entire footing is in compression</div>
                    </div>

                    <h3>Bearing Pressure Distribution Cases</h3>
                    <div style="display: flex; justify-content: space-around; margin: 20px 0; gap: 20px;">

                        <!-- Case (a): Single Eccentricity - Within Kern -->
                        <div style="flex: 1; text-align: center; border: 1px solid #ccc; padding: 15px; background-color: #fafafa;">
                            <h4 style="margin-top: 0;">(a) Single Eccentricity<br>Within Kern</h4>
                            <p style="font-size: 14px; color: #666;">$$e \\leq B/6$$ or $$e \\leq L/6$$</p>
                            <p style="font-size: 13px; color: #666; font-style: italic;">(one e < 0.01m)</p>
                            <div class="equation-block" style="font-size: 13px; margin-top: 10px;">
                                <div>$$q_{{\\text{{max}}}} = \\frac{{P}}{{A}}\\left(1 + \\frac{{6e}}{{D}}\\right)$$</div>
                                <div>$$q_{{\\text{{min}}}} = \\frac{{P}}{{A}}\\left(1 - \\frac{{6e}}{{D}}\\right)$$</div>
                                <p style="font-size: 11px; margin-top: 5px;">D = B or L</p>
                            </div>
                        </div>

                        <!-- Case (b): Single Eccentricity - Outside Kern -->
                        <div style="flex: 1; text-align: center; border: 1px solid #ccc; padding: 15px; background-color: #fafafa;">
                            <h4 style="margin-top: 0;">(b) Single Eccentricity<br>Outside Kern</h4>
                            <p style="font-size: 14px; color: #666;">$$e > B/6$$ or $$e > L/6$$</p>
                            <p style="font-size: 13px; color: #666; font-style: italic;">(one e < 0.01m)</p>
                            <div class="equation-block" style="font-size: 13px; margin-top: 10px;">
                                <div>$$D_{{\\text{{eff}}}} = 3\\left(\\frac{{D}}{{2}} - e\\right)$$</div>
                                <div>$$q_{{\\text{{max}}}} = \\frac{{2P}}{{A_{{\\text{{eff}}}}}}$$</div>
                                <div>$$q_{{\\text{{min}}}} = 0$$</div>
                            </div>
                        </div>

                        <!-- Case (c): Two Eccentricities (Biaxial) -->
                        <div style="flex: 1; text-align: center; border: 1px solid #ccc; padding: 15px; background-color: #fafafa;">
                            <h4 style="margin-top: 0;">(c) Two Eccentricities<br>(Biaxial)</h4>
                            <p style="font-size: 14px; color: #666;">$$e_x \\geq 0.01$$ and $$e_y \\geq 0.01$$</p>
                            <p style="font-size: 13px; color: #666; font-style: italic;">Required: L > 6|e<sub>y</sub>| and B > 6|e<sub>x</sub>|</p>
                            <div class="equation-block" style="font-size: 13px; margin-top: 10px;">
                                <div>$$q = \\frac{{P}}{{LB}}\\left(1 \\pm \\frac{{6e_{{x}}}}{{B}}\\pm \\frac{{6e_{{y}}}}{{L}}\\right)$$</div>
                                <p style="font-size: 11px; margin-top: 5px;">Evaluate at 4 corners</p>
                            </div>
                        </div>
                    </div>
                </div>
        """
        
        # Add load cases section
        html += """
                <div class="section">
                    <h2>Load Cases</h2>
        """
        
        for node_name, load_cases in nodes_dict.items():
            if load_cases:
                html += f"""
                    <h3>Node: {node_name}</h3>
                    <table>
                        <tr>
                            <th>Load Case</th>
                            <th>F1 (kN)</th>
                            <th>F2 (kN)</th>
                            <th>F3 (kN)</th>
                            <th>M1 (kN-m)</th>
                            <th>M2 (kN-m)</th>
                            <th>M3 (kN-m)</th>
                        </tr>
                """
                for lc in load_cases:
                    html += f"""
                        <tr>
                            <td>{lc['name']}</td>
                            <td>{lc['F1']:.2f}</td>
                            <td>{lc['F2']:.2f}</td>
                            <td>{lc['F3']:.2f}</td>
                            <td>{lc['M1']:.2f}</td>
                            <td>{lc['M2']:.2f}</td>
                            <td>{lc['M3']:.2f}</td>
                        </tr>
                    """
                html += """
                    </table>
                """
        
        html += """
                </div>
                
                <div class="section">
                    <h2>Optimization Results - Optimal Footing Geometry</h2>
                    <table>
                        <tr>
                            <th>Node</th>
                            <th>L (m)</th>
                            <th>B (m)</th>
                            <th>h (m)</th>
                            <th>Ped. Size (m)</th>
                            <th>Ped. Height (m)</th>
                            <th>Weight (kN)</th>
                            <th>Governing LC</th>
                            <th>Max Pressure (kPa)</th>
                            <th>Allowable (kPa)</th>
                            <th>Utilization</th>
                        </tr>
        """
        
        # Add optimization results
        for node_name in nodes_dict.keys():
            if node_name in optimization_results and optimization_results[node_name]:
                opt = optimization_results[node_name]
                utilization = opt['max_pressure'] / opt['allowable_bearing']
                util_color = '#ffffff' if utilization <= 1.0 else '#f5f5f5'
                html += f"""
                        <tr style="background-color: {util_color};">
                            <td><strong>{node_name}</strong></td>
                            <td>{opt['L']:.2f}</td>
                            <td>{opt['B']:.2f}</td>
                            <td>{opt['h']:.2f}</td>
                            <td>{opt['pedestal_size']:.2f}</td>
                            <td>{opt['pedestal_height']:.2f}</td>
                            <td>{opt['total_weight']:.1f}</td>
                            <td>{opt['governing_case_name']}</td>
                            <td>{opt['max_pressure']:.1f}</td>
                            <td>{opt['allowable_bearing']:.1f}</td>
                            <td>{utilization:.3f}</td>
                        </tr>
                """
            else:
                html += f"""
                        <tr style="background-color: #f5f5f5;">
                            <td><strong>{node_name}</strong></td>
                            <td colspan="10" style="text-align: center;">No feasible solution found</td>
                        </tr>
                """
        
        html += """
                    </table>
                    <p class="description">Note: Utilization ratio should be less than or equal to 1.0 for acceptable design</p>
                </div>
                
                <div class="section">
                    <h2>Design Summary</h2>
        """
        
        # Add individual node summaries
        for node_name in nodes_dict.keys():
            if node_name in optimization_results and optimization_results[node_name]:
                opt = optimization_results[node_name]
                utilization = opt['max_pressure'] / opt['allowable_bearing']
                
                if utilization <= 1.0:
                    status_text = "ACCEPTABLE"
                else:
                    status_text = "OVERSTRESSED"
                
                html += f"""
                    <div class="result-box">
                        <h3>Node {node_name} - {status_text}</h3>
                        <p><strong>Optimal Footing Dimensions:</strong></p>
                        <ul>
                            <li>Length (L) = {opt['L']:.2f} m</li>
                            <li>Width (B) = {opt['B']:.2f} m</li>
                            <li>Slab Thickness (h) = {opt['h']:.2f} m</li>
                            <li>Pedestal Size = {opt['pedestal_size']:.2f} m × {opt['pedestal_size']:.2f} m</li>
                            <li>Pedestal Height = {opt['pedestal_height']:.2f} m</li>
                        </ul>
                        <p><strong>Performance:</strong></p>
                        <ul>
                            <li>Total Foundation Weight = <span class="result-value">{opt['total_weight']:.1f} kN</span></li>
                            <li>Governing Load Case = {opt['governing_case_name']}</li>
                            <li>Maximum Bearing Pressure = {opt['max_pressure']:.1f} kPa</li>
                            <li>Allowable Bearing Capacity = {opt['allowable_bearing']:.1f} kPa</li>
                            <li>Utilization Ratio = {utilization:.3f} ({utilization*100:.1f}%)</li>
                            <li>Total Foundation Depth = {opt['total_depth']:.2f} m</li>
                        </ul>
                    </div>
                """
            else:
                html += f"""
                    <div class="result-box">
                        <h3>Node {node_name} - NO SOLUTION</h3>
                        <p>No feasible footing geometry found within the design space.</p>
                        <p><strong>Recommendations:</strong></p>
                        <ul>
                            <li>Increase minimum footing length</li>
                            <li>Improve soil bearing capacity</li>
                            <li>Reduce applied loads</li>
                        </ul>
                    </div>
                """
        
        html += """
                </div>
                
                <div class="section" style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #cccccc;">
                    <p style="text-align: center; color: #666666;">
                        <em>Generated by VIKTOR Footing Optimization App</em>
                    </p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return vkt.WebResult(html=html)
    
    @vkt.DataView("Optimization Results", duration_guess=10)
    def view_results(self, params, **kwargs):
        """Display multi-node optimization results"""
        
        vkt.progress_message("Starting multi-node optimization...", 0)
        
        # Group load cases by node
        nodes_dict = {}
        for node_row in params.nodes_section.nodes:
            node_name = node_row['node_name']
            nodes_dict[node_name] = []
        
        # Assign load cases to nodes
        for lc_row in params.load_cases_section.load_cases:
            node_name = lc_row['node']
            if node_name in nodes_dict:
                nodes_dict[node_name].append({
                    'name': lc_row['case_name'],
                    'F1': lc_row['F1'],
                    'F2': lc_row['F2'],
                    'F3': lc_row['F3'],
                    'M1': lc_row['M1'],
                    'M2': lc_row['M2'],
                    'M3': lc_row['M3']
                })
        
        # Optimize each node
        data = vkt.DataGroup()
        total_nodes = len(nodes_dict)
        
        for idx, (node_name, load_cases) in enumerate(nodes_dict.items()):
            vkt.progress_message(f"Optimizing {node_name}...", (idx / total_nodes) * 100)
            
            if not load_cases:
                # No load cases for this node
                node_group = vkt.DataGroup()
                node_group.add(
                    vkt.DataItem("Status", "No load cases defined", status=vkt.DataStatus.WARNING)
                )
                data.add(vkt.DataItem(node_name, subgroup=node_group))
                continue
            
            # Run optimization
            optimal = optimize_footing_for_node(
                load_cases=load_cases,
                min_footing_length=params.optimization_section.min_footing_length,
                gamma_concrete=params.material_section.gamma_concrete,
                gamma_fill=params.material_section.gamma_fill,
                bearing_table=params.section_bearing.bearing_table
            )
            
            # Create result group for this node
            node_group = vkt.DataGroup()
            
            if optimal is None:
                node_group.add(
                    vkt.DataItem(
                        "Status",
                        "No feasible solution found",
                        status=vkt.DataStatus.ERROR,
                        status_message="Try increasing min footing length or bearing capacity"
                    )
                )
            else:
                # Optimal geometry
                geometry_group = vkt.DataGroup()
                geometry_group.add(
                    vkt.DataItem("Footing Length (L)", optimal['L'], suffix="m", number_of_decimals=2),
                    vkt.DataItem("Footing Width (B)", optimal['B'], suffix="m", number_of_decimals=2),
                    vkt.DataItem("Slab Thickness (h)", optimal['h'], suffix="m", number_of_decimals=2),
                    vkt.DataItem("Pedestal Size", optimal['pedestal_size'], suffix="m", number_of_decimals=2),
                    vkt.DataItem("Pedestal Height", optimal['pedestal_height'], suffix="m", number_of_decimals=2)
                )
                node_group.add(vkt.DataItem("Optimal Geometry", subgroup=geometry_group))
                
                # Performance metrics
                node_group.add(
                    vkt.DataItem(
                        "Total Weight",
                        optimal['total_weight'],
                        suffix="kN",
                        number_of_decimals=2,
                        status=vkt.DataStatus.SUCCESS,
                        status_message="Optimized for minimum weight"
                    )
                )
                
                node_group.add(
                    vkt.DataItem(
                        "Governing Load Case",
                        optimal['governing_case_name'],
                        status=vkt.DataStatus.INFO
                    )
                )
                
                # Bearing pressure check
                pressure_group = vkt.DataGroup()
                pressure_group.add(
                    vkt.DataItem("Max Bearing Pressure", optimal['max_pressure'], suffix="kPa", number_of_decimals=2),
                    vkt.DataItem("Allowable Bearing", optimal['allowable_bearing'], suffix="kPa", number_of_decimals=2),
                    vkt.DataItem("Utilization Ratio", optimal['max_pressure'] / optimal['allowable_bearing'], number_of_decimals=3)
                )
                node_group.add(vkt.DataItem("Bearing Pressure", subgroup=pressure_group))
                
                node_group.add(
                    vkt.DataItem("Foundation Depth", optimal['total_depth'], suffix="m", number_of_decimals=2)
                )
            
            data.add(vkt.DataItem(node_name, subgroup=node_group))
        
        vkt.progress_message("Optimization complete!", 100)

        return vkt.DataResult(data)
