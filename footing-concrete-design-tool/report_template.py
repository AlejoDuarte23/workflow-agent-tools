"""
HTML Report Template for Concrete Footing Design Tool
"""

def get_report_header():
    """Return HTML header with CSS styling"""
    return """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Concrete Footing Design Calculation Report</title>
    <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
    <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    <style>
        body {
            font-family: 'Arial', 'Helvetica', sans-serif;
            margin: 20px;
            background-color: #ffffff;
            color: #000000;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
        }
        h1 {
            color: #000000;
            border-bottom: 2px solid #000000;
            padding-bottom: 10px;
            font-size: 24px;
            font-weight: bold;
        }
        h2 {
            color: #000000;
            margin-top: 30px;
            border-bottom: 1px solid #666666;
            padding-bottom: 5px;
            font-size: 18px;
            font-weight: bold;
        }
        h3 {
            color: #333333;
            margin-top: 20px;
            font-size: 16px;
            font-weight: bold;
        }
        .equation-block {
            background-color: #f5f5f5;
            padding: 15px;
            margin: 15px 0;
            border-left: 3px solid #666666;
        }
        .equation {
            margin: 10px 0;
            font-size: 16px;
        }
        .description {
            color: #666666;
            font-style: italic;
            margin: 10px 0;
            font-size: 14px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            border: 1px solid #cccccc;
        }
        th {
            background-color: #e0e0e0;
            color: #000000;
            padding: 10px;
            text-align: left;
            font-weight: bold;
            border: 1px solid #cccccc;
        }
        td {
            padding: 8px;
            border: 1px solid #cccccc;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .result-box {
            background-color: #ffffff;
            border: 1px solid #cccccc;
            padding: 15px;
            margin: 15px 0;
        }
        .pass {
            background-color: #e8f5e9;
        }
        .fail {
            background-color: #ffebee;
        }
        .result-value {
            font-weight: bold;
        }
        .section {
            margin: 30px 0;
        }
        .input-params {
            background-color: #f5f5f5;
            padding: 15px;
            border-left: 3px solid #666666;
            margin: 10px 0;
        }
        .check-status {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 12px;
        }
        .check-pass {
            background-color: #4caf50;
            color: white;
        }
        .check-fail {
            background-color: #f44336;
            color: white;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Multi-Node Concrete Footing Design Calculation Report (ACI 318-19)</h1>
"""


def get_design_equations():
    """Return HTML for design equations section"""
    return """
        <div class="section">
            <h2>Design Equations (ACI 318-19)</h2>

            <h3>Foundation Weight Calculation</h3>
            <div class="equation-block">
                <div class="equation">$$V_{\\text{slab}} = B \\times L \\times H$$</div>
                <div class="equation">$$V_{\\text{pedestal}} = b_1 \\times b_2 \\times p_h$$</div>
                <div class="equation">$$V_{\\text{fill}} = (B \\times L - b_1 \\times b_2) \\times p_h$$</div>
                <div class="equation">$$W_{\\text{total}} = V_{\\text{slab}} \\times \\gamma_c + V_{\\text{pedestal}} \\times \\gamma_c + V_{\\text{fill}} \\times \\gamma_f$$</div>
            </div>

            <h3>Load Transfer to Footing Slab Level</h3>
            <div class="equation-block">
                <div class="equation">$$d_{\\text{transfer}} = p_h + \\frac{H}{2}$$</div>
                <div class="equation">$$F3_{\\text{footing}} = F3 - W_{\\text{total}}$$</div>
                <div class="equation">$$M1_{\\text{footing}} = M1 + F2 \\times d_{\\text{transfer}}$$</div>
                <div class="equation">$$M2_{\\text{footing}} = M2 + F1 \\times d_{\\text{transfer}}$$</div>
                <div class="description">Where F3 is negative for compression</div>
            </div>

            <h3>Eccentricity Calculation</h3>
            <div class="equation-block">
                <div class="equation">$$e_x = \\frac{M2_{\\text{footing}}}{|F3_{\\text{footing}}|}$$</div>
                <div class="equation">$$e_y = \\frac{M1_{\\text{footing}}}{|F3_{\\text{footing}}|}$$</div>
            </div>

            <h3>Effective Depth</h3>
            <div class="equation-block">
                <div class="equation">$$d = H - \\text{cover} - \\frac{d_b}{2}$$</div>
                <div class="description">Where cover and d<sub>b</sub> are in mm, H in m</div>
            </div>

            <h3>Two-Way Shear (Punching) - ACI 318-19 Section 22.6</h3>
            <div class="equation-block">
                <div class="equation">$$b_0 = 2(b_1 + d) + 2(b_2 + d)$$</div>
                <div class="equation">$$V_u = \\sigma_u \\times [B \\times L - (b_1 + d)(b_2 + d)]$$</div>
                <div class="equation">$$\\phi V_c = \\min(V_{c1}, V_{c2}, V_{c3})$$</div>
                <div class="equation">$$V_{c1} = 0.75 \\times 0.33 \\lambda_s \\sqrt{f'_c} \\times b_0 \\times d$$</div>
                <div class="equation">$$V_{c2} = 0.75 \\times 0.17(1 + \\frac{2}{\\beta}) \\lambda_s \\sqrt{f'_c} \\times b_0 \\times d$$</div>
                <div class="equation">$$V_{c3} = 0.75 \\times 0.083(2 + \\frac{\\alpha_s d}{b_0}) \\lambda_s \\sqrt{f'_c} \\times b_0 \\times d$$</div>
                <div class="description">Check: V<sub>u</sub> ≤ φV<sub>c</sub></div>
            </div>

            <h3>One-Way Shear (Beam) - ACI 318-19 Section 22.5</h3>
            <div class="equation-block">
                <div class="equation">$$V_{u,x} = \\sigma_u \\times B \\times \\max(0, \\frac{L}{2} - \\frac{b_2}{2} - d)$$</div>
                <div class="equation">$$V_{u,y} = \\sigma_u \\times L \\times \\max(0, \\frac{B}{2} - \\frac{b_1}{2} - d)$$</div>
                <div class="equation">$$\\phi V_{c,x} = 0.75 \\times 0.17 \\sqrt{f'_c} \\times b_{w,x} \\times d$$</div>
                <div class="equation">$$\\phi V_{c,y} = 0.75 \\times 0.17 \\sqrt{f'_c} \\times b_{w,y} \\times d$$</div>
                <div class="description">Check: V<sub>u</sub> ≤ φV<sub>c</sub> in both directions</div>
            </div>

            <h3>Flexure Design - Strip Method</h3>
            <div class="equation-block">
                <div class="equation">$$w_{u,x} = \\sigma_u \\times L \\quad \\text{(kN/m)}$$</div>
                <div class="equation">$$w_{u,y} = \\sigma_u \\times B \\quad \\text{(kN/m)}$$</div>
                <div class="equation">$$M_{u,x} = \\frac{w_{u,x}}{2} \\left(\\frac{L}{2} - \\frac{b_2}{2}\\right)^2$$</div>
                <div class="equation">$$M_{u,y} = \\frac{w_{u,y}}{2} \\left(\\frac{B}{2} - \\frac{b_1}{2}\\right)^2$$</div>
                <div class="description">Moment at face of pedestal</div>
            </div>

            <h3>Required Reinforcement</h3>
            <div class="equation-block">
                <div class="equation">$$\\phi M_n = \\phi A_s f_y \\left(d - \\frac{a}{2}\\right)$$</div>
                <div class="equation">$$a = \\frac{A_s f_y}{0.85 f'_c b}$$</div>
                <div class="equation">$$A_{s,\\min} = 0.0018 b h$$</div>
                <div class="equation">$$A_{s,\\text{required}} = \\max(A_{s,\\text{calc}}, A_{s,\\min})$$</div>
                <div class="description">Where φ = 0.9 for flexure</div>
            </div>

            <h3>Rebar Spacing - Strip Method</h3>
            <div class="equation-block">
                <div class="equation">$$A_b = \\frac{\\pi d_b^2}{4}$$</div>
                <div class="equation">$$n = \\lceil \\frac{A_{s,\\text{required}}}{A_b} \\rceil \\quad (\\text{minimum 2})$$</div>
                <div class="equation">$$s_{\\text{clear}} = \\frac{L_{\\text{strip}} - 2 \\times \\text{cover} - n \\times d_b}{n - 1}$$</div>
                <div class="equation">$$s_{\\text{c/c}} = s_{\\text{clear}} + d_b$$</div>
                <div class="description">X-direction uses strip length L; Y-direction uses strip length B</div>
            </div>
        </div>
"""


def get_report_footer():
    """Return HTML footer"""
    return """
        <div class="section" style="margin-top: 50px; padding-top: 20px; border-top: 1px solid #cccccc;">
            <p style="text-align: center; color: #666666;">
                <em>Generated by VIKTOR Concrete Footing Design Tool</em>
            </p>
        </div>
    </div>
</body>
</html>
"""


def format_design_parameters(fc, fy, gamma_concrete, gamma_fill, cover, db):
    """Format design parameters section"""
    return f"""
        <div class="section">
            <h2>Design Parameters</h2>
            <div class="input-params">
                <p><strong>Concrete Properties:</strong></p>
                <p>\\(f'_c = {fc}\\) MPa</p>
                <p>\\(f_y = {fy}\\) MPa</p>
                <p>\\(\\gamma_{{\\text{{concrete}}}} = {gamma_concrete}\\) kN/m³</p>
                <p>\\(\\gamma_{{\\text{{fill}}}} = {gamma_fill}\\) kN/m³</p>
                <p><strong>Reinforcement:</strong></p>
                <p>Concrete Cover = {cover} mm</p>
                <p>Rebar Diameter (d<sub>b</sub>) = {db} mm</p>
            </div>
        </div>
"""


def format_node_geometry_table(geometry_by_node):
    """Format node geometry table"""
    html = """
        <div class="section">
            <h2>Node Geometry</h2>
            <table>
                <tr>
                    <th>Node</th>
                    <th>Footing B×L×H (m)</th>
                    <th>Pedestal b1×b2×ph (m)</th>
                    <th>Coordinates (x, y, z)</th>
                </tr>
"""

    for node_name, geom in geometry_by_node.items():
        html += f"""
                <tr>
                    <td><strong>{node_name}</strong></td>
                    <td>{geom['B']:.2f} × {geom['L']:.2f} × {geom['H']:.2f}</td>
                    <td>{geom['b1']:.3f} × {geom['b2']:.3f} × {geom['ph']:.2f}</td>
                    <td>({geom['x']:.2f}, {geom['y']:.2f}, {geom['z']:.2f})</td>
                </tr>
"""

    html += """
            </table>
        </div>
"""
    return html


def format_load_cases_table(loads_by_node):
    """Format load cases table"""
    html = """
        <div class="section">
            <h2>Load Cases</h2>
"""

    for node_name, load_cases in loads_by_node.items():
        if load_cases:
            html += f"""
            <h3>Node: {node_name}</h3>
            <table>
                <tr>
                    <th>Load Case</th>
                    <th>F1 (kN)</th>
                    <th>F2 (kN)</th>
                    <th>F3 (kN)</th>
                    <th>M1 (kN·m)</th>
                    <th>M2 (kN·m)</th>
                    <th>M3 (kN·m)</th>
                </tr>
"""
            for lc in load_cases:
                html += f"""
                <tr>
                    <td>{lc['case_name']}</td>
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
"""
    return html


def format_design_results(results_by_node_lc):
    """Format design results for all node-loadcase combinations"""
    html = """
        <div class="section">
            <h2>Design Check Results</h2>
"""

    for key, result in results_by_node_lc.items():
        node_name, case_name = key
        overall_pass = result['overall_pass']
        row_class = 'pass' if overall_pass else 'fail'
        status_class = 'check-pass' if overall_pass else 'check-fail'
        status_text = 'PASS' if overall_pass else 'FAIL'

        html += f"""
        <div class="result-box {row_class}">
            <h3>{node_name} - {case_name} <span class="check-status {status_class}">{status_text}</span></h3>

            <p><strong>Foundation Weights:</strong></p>
            <ul>
                <li>Total Weight = {result['weights']['total_weight']:.1f} kN</li>
                <li>Slab + Pedestal + Fill = {result['weights']['slab_weight']:.1f} + {result['weights']['pedestal_weight']:.1f} + {result['weights']['fill_weight']:.1f} kN</li>
            </ul>

            <p><strong>Factored Actions at Footing Level:</strong></p>
            <ul>
                <li>F3 (Axial) = {abs(result['factored_actions']['Fz_footing']):.1f} kN</li>
                <li>Eccentricities: e<sub>x</sub> = {result['factored_actions']['ex']:.4f} m, e<sub>y</sub> = {result['factored_actions']['ey']:.4f} m</li>
                <li>Moments: M<sub>x</sub> = {result['factored_actions']['Mx_footing']:.1f} kN·m, M<sub>y</sub> = {result['factored_actions']['My_footing']:.1f} kN·m</li>
            </ul>

            <p><strong>Two-Way Shear (Punching):</strong></p>
            <ul>
                <li>V<sub>u</sub> = {result['punching_shear']['Vu']:.1f} kN</li>
                <li>φV<sub>c</sub> = {result['punching_shear']['Vc_min']:.1f} kN</li>
                <li>Utilization = {result['punching_shear']['Vu'] / result['punching_shear']['Vc_min'] if result['punching_shear']['Vc_min'] > 0 else 0:.3f}</li>
                <li><span class="check-status {'check-pass' if result['punching_shear']['passes'] else 'check-fail'}">{'PASS' if result['punching_shear']['passes'] else 'FAIL'}</span></li>
            </ul>

            <p><strong>One-Way Shear:</strong></p>
            <ul>
                <li><strong>X-direction:</strong> V<sub>u</sub> = {result['one_way_shear']['Vu_x']:.1f} kN, φV<sub>c</sub> = {result['one_way_shear']['Vc_x']:.1f} kN, Util = {result['one_way_shear']['util_x']:.3f}
                    <span class="check-status {'check-pass' if result['one_way_shear']['Vu_x'] <= result['one_way_shear']['Vc_x'] else 'check-fail'}">{'PASS' if result['one_way_shear']['Vu_x'] <= result['one_way_shear']['Vc_x'] else 'FAIL'}</span>
                </li>
                <li><strong>Y-direction:</strong> V<sub>u</sub> = {result['one_way_shear']['Vu_y']:.1f} kN, φV<sub>c</sub> = {result['one_way_shear']['Vc_y']:.1f} kN, Util = {result['one_way_shear']['util_y']:.3f}
                    <span class="check-status {'check-pass' if result['one_way_shear']['Vu_y'] <= result['one_way_shear']['Vc_y'] else 'check-fail'}">{'PASS' if result['one_way_shear']['Vu_y'] <= result['one_way_shear']['Vc_y'] else 'FAIL'}</span>
                </li>
            </ul>

            <p><strong>Flexure - Required Reinforcement:</strong></p>
            <ul>
                <li><strong>X-direction:</strong> A<sub>s,req</sub> = {result['flexure']['As_req_x']:.0f} mm² (M<sub>u</sub> = {result['flexure']['Mu_x']:.1f} kN·m)</li>
                <li><strong>Y-direction:</strong> A<sub>s,req</sub> = {result['flexure']['As_req_y']:.0f} mm² (M<sub>u</sub> = {result['flexure']['Mu_y']:.1f} kN·m)</li>
            </ul>

            <p><strong>Rebar Spacing:</strong></p>
            <ul>
                <li><strong>X-direction:</strong> {result['spacing']['n_x']} bars @ {result['spacing']['s_c2c_x']:.0f} mm c/c (clear = {result['spacing']['s_clear_x']:.0f} mm)</li>
                <li><strong>Y-direction:</strong> {result['spacing']['n_y']} bars @ {result['spacing']['s_c2c_y']:.0f} mm c/c (clear = {result['spacing']['s_clear_y']:.0f} mm)</li>
            </ul>
        </div>
"""

    html += """
        </div>
"""
    return html
