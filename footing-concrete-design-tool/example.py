import viktor as vkt
import math
 
 
class Parametrization(vkt.Parametrization):
    # Factored Actions at Pedestal Level
    section_loads = vkt.Section("Factored Actions (Loads) at pedestal level")
    section_loads.Fz = vkt.NumberField("Axial Force (Fz)", default=1700, suffix="kN")
    section_loads.Fy = vkt.NumberField("Transverse Force (Fy)", default=2, suffix="kN")
    section_loads.Fx = vkt.NumberField("Longitudinal Force (Fx)", default=3, suffix="kN")
    section_loads.Mx = vkt.NumberField("Longitudinal Moment (Mx)", default=80, suffix="kN·m")
    section_loads.My = vkt.NumberField("Transverse Moment (My)", default=60, suffix="kN·m")
    
    # Material Properties
    section_material = vkt.Section("Material Properties")
    section_material.fc = vkt.NumberField("Concrete compressive strength (f'c)", default=28, suffix="MPa")
    section_material.fy = vkt.NumberField("Steel yield strength (fy)", default=420, suffix="MPa")
    section_material.gamma_fill = vkt.NumberField("Fill material unit weight (γ_fill)", default=19.5, suffix="kN/m³")
    
    # Soil Properties
    section_soil = vkt.Section("Soil Properties (Geotechnical Inputs)")
    section_soil.gamma_soil = vkt.NumberField("Soil unit weight (γ_soil)", default=20, suffix="kN/m³")
    section_soil.phi = vkt.NumberField("Soil friction angle (φ)", default=25, suffix="°")
    section_soil.qa = vkt.NumberField("Allowable bearing capacity (q_a)", default=190, suffix="kPa")
    
    # Footing Geometry
    section_footing = vkt.Section("Footing Geometry (Preliminary Sizing)")
    section_footing.B = vkt.NumberField("Footing width (B)", default=2.20, suffix="m")
    section_footing.L = vkt.NumberField("Footing length (L)", default=2.40, suffix="m")
    section_footing.H = vkt.NumberField("Footing thickness (H)", default=0.6, suffix="m")
    
    # Pedestal Geometry
    section_pedestal = vkt.Section("Pedestal/Column Geometry")
    section_pedestal.b1 = vkt.NumberField("Pedestal width (b1)", default=0.400, suffix="m")
    section_pedestal.b2 = vkt.NumberField("Pedestal length (b2)", default=0.500, suffix="m")
    section_pedestal.ph = vkt.NumberField("Pedestal height (ph)", default=1.0, suffix="m")
    
    # Rebar Details
    section_rebar = vkt.Section("Rebar spacing (strip method)")
    section_rebar.db = vkt.NumberField("Bar diameter (db)", default=12, suffix="mm")
 
 
class Controller(vkt.Controller):
    parametrization = Parametrization
    
    @vkt.WebView("Calculation Steps", duration_guess=1)
    def show_calculations(self, params, **kwargs):
        # Extract parameters using exact Excel variable names
        C5 = params.section_loads.Fz
        C6 = params.section_loads.Fy
        C7 = params.section_loads.Fx
        C8 = params.section_loads.Mx
        C9 = params.section_loads.My
        
        C14 = params.section_material.fc
        C15 = params.section_material.fy
        C16 = params.section_material.gamma_fill
        
        C20 = params.section_soil.gamma_soil
        
        C25 = params.section_footing.B
        C26 = params.section_footing.L
        C27 = params.section_footing.H
        
        C31 = params.section_pedestal.b1
        C32 = params.section_pedestal.b2
        C33 = params.section_pedestal.ph
        
        K39 = params.section_rebar.db
        
        # Foundation Weights
        C36 = 24  # Concrete specific weight [kN/m³]
        C37 = C25 * C26 * C27 * C36  # Slab weight [kN]
        C38 = C33 * C32 * C31 * C36  # Pedestal weight [kN]
        C39 = ((C25 * C26 * C27) * (C33 + C27) - (C31 * C32 * C33)) * C16  # Soil weight [kN]
        C40 = C37 + C38 + C39  # Total weight [kN]
        
        # Factored Actions at footing slab level
        C43 = C40 + C5  # Axial Force at footing level [kN]
        C46 = C8 + C7 * (C33 + C27 / 2)  # Longitudinal Moment at footing level [kN·m]
        C47 = C9 + C6 * (C33 + C27 / 2)  # Transverse Moment at footing level [kN·m]
        C48 = C46 / C43  # Eccentricity (x-direction) [m]
        C49 = C47 / C43  # Eccentricity (y-direction) [m]
        
        # Effective Depth and Critical Sections
        C54 = C27 - 0.09  # Effective depth (d) [m]
        C55 = 2 * (C31 + C54) + 2 * (C54 + C32)  # Critical perimeter (b0) [m]
        C56 = C54 * C55  # Critical area (A0) [m²]
        C53 = (C5 / (C25 * C26)) * (1 + 6 * C48 / C25 + 6 * C49 / C26)  # Ultimate pressure (σu) [kPa]
        C57 = C43 - C53 * (C31 + C54) * (C32 + C54)  # Applied punching shear (Vu) [kN]
        C58 = C57 / C56  # Shear stress demand (σv) [kPa]
        
        # Size effect factor and column location
        C60 = 40  # Column Location (as) [40 for interior]
        C61 = min(1, math.sqrt(2 / (1 + (C54 * 1000) / 254)))  # Size effect factor (λs)
        C59 = 0.75 * min(
            0.33 * C61,
            0.17 * (1 + 2 / (max(C31, C32) / min(C31, C32))) * C61,
            0.083 * (2 + C60 * C54 / C55) * C61
        ) * math.sqrt(C14) * 1000  # Concrete shear capacity (stress) (Vc) [kPa]
        
        # Two-Way (Punching) Shear Check
        K8 = (C43 / (C25 * C26) * (C25 * C26 - (C31 + C54) * (C32 + C54)))  # Applied punching shear (Vu) [kN]
        K9 = 0.75 * 0.33 * C61 * math.sqrt(C14) * C55 * C54 * 1000  # Capacity (Condition 1) (Vc1) [kN]
        K10 = 0.75 * 0.17 * (1 + 2 / (max(C31, C32) / min(C31, C32))) * C61 * math.sqrt(C14) * C55 * C54 * 1000  # Capacity (Condition 2) (Vc2) [kN]
        K11 = 0.75 * 0.083 * (2 + C60 * C54 / C55) * C61 * math.sqrt(C14) * C55 * C54 * 1000  # Capacity (Condition 3) (Vc3) [kN]
        
        # One-Way Shear Check
        K16 = C53 * C25 * ((C26 / 2 - C31 / 2 - C54))  # Applied one-way shear (Vu) [kN]
        K17 = K16 / (C25 * C54)  # Shear stress demand (σv) [kPa]
        K18 = 0.75 * 0.17 * math.sqrt(C14) * C25 * C54 * 1000  # Capacity (Condition 1) (Vc1) [kN]
        K19 = 0.75 * 0.17 * math.sqrt(C14) * C26 * C54 * 1000  # Capacity (Condition 2) (Vc2) [kN]
        K20 = min(K18, K19)  # Governing capacity [kN]
        
        # Flexure for Reinforcement
        K25 = C53 * C26  # Factored line load (from pressure along B) (wu) [kN/m]
        K26 = C53 * C25  # Factored line load (from pressure along L) (wu) [kN/m]
        K27 = K26 * (C26 / 2 - C32 / 2) ** 2 / 2  # Factored moment along B (Mu) [kN·m]
        K28 = K25 * (C25 / 2 - C31 / 2) ** 2 / 2  # Factored moment along L (Mu) [kN·m]
        
        # Concrete Bearing at Column/Pedestal Base
        K32 = C43  # Applied load (Pu) [kN]
        K33 = C25 * C26  # Supporting (effective) area (A2) [m²]
        K34 = math.sqrt(K33 / (C31 * C32))  # Area factor (√(A2/A1))
        K35 = min(1000 * 0.65 * 0.85 * C14 * (C31 * C32) * K34, 1000 * 0.65 * 0.85 * C14 * (C31 * C32) * 2)  # Bearing resistance (φPn) [kN]
        
        # Rebar spacing calculations
        K40 = (C27 - C54) * 1000 - K39 / 2  # Cover (cc) [mm]
        K41 = C27 * 1000 - K40 - K39 / 2  # Effective depth (d) [mm]
        
        # X-direction (along B)
        K44 = 0.0018 * (C26 * 1000) * (C27 * 1000)  # min As [mm²]
        K45 = ((1 / (C15 / (0.85 * C14))) * (1 - math.sqrt(1 - (2 * ((K28 * 1000000) / (0.9 * (C26 * 1000) * (K41 ** 2))) * (C15 / (0.85 * C14))) / C15))) * (C26 * 1000) * K41  # required As [mm²]
        K46 = max(K44, K45)  # As,target (x-dir) [mm²]
        K47 = math.ceil(K46 / (math.pi * (K39 ** 2) / 4))  # Required # bars (n) (x-dir)
        K48 = ((C26 * 1000) - 2 * K40 - K47 * K39) / (K47 - 1) if K47 > 1 else 0  # Required clear spacing (s_clear) (x-dir) [mm]
        K49 = K48 + K39 if K48 else 0  # Required spacing (s c/c) (x-dir) [mm]
        
        # Y-direction (along L)
        K51 = 0.0018 * (C26 * 1000) * (C27 * 1000)  # min As [mm²]
        K52 = ((1 / (C15 / (0.85 * C14))) * (1 - math.sqrt(1 - (2 * ((K27 * 1000000) / (0.9 * (C25 * 1000) * (K41 ** 2))) * (C15 / (0.85 * C14))) / C15))) * (C25 * 1000) * K41  # required As [mm²]
        K53 = max(K51, K52)  # As,target (y-dir) [mm²]
        K54 = math.ceil(K53 / (math.pi * (K39 ** 2) / 4))  # Required # bars (n) (y-dir)
        K55 = ((C25 * 1000) - 2 * K40 - K54 * K39) / (K54 - 1) if K54 > 1 else 0  # Required clear spacing (s_clear) (y-dir) [mm]
        K56 = K55 + K39 if K55 else 0  # Required spacing (s c/c) (y-dir) [mm]
        
        # Build HTML with intermediate steps
        html = f"""
        <html>
        <head>
            <meta charset="UTF-8">
            <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
            <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
            <script>
                MathJax = {{
                    tex: {{
                        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
                    }}
                }};
            </script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    background-color: #f5f5f5;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    background-color: #ecf0f1;
                    padding: 10px;
                    border-left: 4px solid #3498db;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 20px;
                }}
                .calc-step {{
                    background-color: white;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .formula {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    margin: 10px 0;
                    border-left: 3px solid #3498db;
                    overflow-x: auto;
                    font-size: 1.05em;
                }}
                .result {{
                    font-weight: bold;
                    color: #27ae60;
                    font-size: 1.1em;
                }}
                .check {{
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 5px;
                }}
                .pass {{
                    background-color: #d4edda;
                    border-left: 4px solid #28a745;
                }}
                .fail {{
                    background-color: #f8d7da;
                    border-left: 4px solid #dc3545;
                }}
            </style>
        </head>
        <body>
            <h1>Concrete Footing Design — Intermediate Calculation Steps</h1>
            <p><em>ACI 318-19 Standard</em></p>
            
            <h2>Step 1: Foundation Weights</h2>
            <div class="calc-step">
                <p><strong>Slab Weight:</strong></p>
                <div class="formula">$$W_{{\\text{{slab}}}} = B \\times L \\times H \\times \\gamma_{{\\text{{concrete}}}} = {C25:.2f} \\times {C26:.2f} \\times {C27:.2f} \\times {C36} = \\textbf{{{C37:.2f} kN}}$$</div>
                
                <p><strong>Pedestal Weight:</strong></p>
                <div class="formula">$$W_{{\\text{{pedestal}}}} = p_h \\times b_2 \\times b_1 \\times \\gamma_{{\\text{{concrete}}}} = {C33:.2f} \\times {C32:.3f} \\times {C31:.3f} \\times {C36} = \\textbf{{{C38:.2f} kN}}$$</div>
                
                <p><strong>Soil Weight:</strong></p>
                <div class="formula">$$W_{{\\text{{soil}}}} = [(B \\times L \\times H) \\times (p_h + H) - (b_1 \\times b_2 \\times p_h)] \\times \\gamma_{{\\text{{fill}}}} = \\textbf{{{C39:.2f} kN}}$$</div>
                
                <p><strong>Total Weight:</strong></p>
                <div class="formula">$$W_{{\\text{{total}}}} = W_{{\\text{{slab}}}} + W_{{\\text{{pedestal}}}} + W_{{\\text{{soil}}}} = {C37:.2f} + {C38:.2f} + {C39:.2f} = \\textbf{{{C40:.2f} kN}}$$</div>
            </div>
            
            <h2>Step 2: Factored Actions at Footing Slab Level</h2>
            <div class="calc-step">
                <p><strong>Axial Force at Footing Level:</strong></p>
                <div class="formula">$$F_{{z,\\text{{footing}}}} = W_{{\\text{{total}}}} + F_z = {C40:.2f} + {C5:.2f} = \\textbf{{{C43:.2f} kN}}$$</div>
                
                <p><strong>Longitudinal Moment at Footing Level:</strong></p>
                <div class="formula">$$M_{{x,\\text{{footing}}}} = M_x + F_x \\times (p_h + H/2) = {C8:.2f} + {C7:.2f} \\times ({C33:.2f} + {C27:.2f}/2) = \\textbf{{{C46:.2f} kN·m}}$$</div>
                
                <p><strong>Transverse Moment at Footing Level:</strong></p>
                <div class="formula">$$M_{{y,\\text{{footing}}}} = M_y + F_y \\times (p_h + H/2) = {C9:.2f} + {C6:.2f} \\times ({C33:.2f} + {C27:.2f}/2) = \\textbf{{{C47:.2f} kN·m}}$$</div>
                
                <p><strong>Eccentricities:</strong></p>
                <div class="formula">$$e_x = \\frac{{M_{{x,\\text{{footing}}}}}}{{F_{{z,\\text{{footing}}}}}} = \\frac{{{C46:.2f}}}{{{C43:.2f}}} = \\textbf{{{C48:.4f} m}}$$</div>
                <div class="formula">$$e_y = \\frac{{M_{{y,\\text{{footing}}}}}}{{F_{{z,\\text{{footing}}}}}} = \\frac{{{C47:.2f}}}{{{C43:.2f}}} = \\textbf{{{C49:.4f} m}}$$</div>
            </div>
            
            <h2>Step 3: Effective Depth and Critical Sections</h2>
            <div class="calc-step">
                <p><strong>Effective Depth:</strong></p>
                <div class="formula">$$d = H - 0.09 = {C27:.2f} - 0.09 = \\textbf{{{C54:.3f} m}}$$</div>
                
                <p><strong>Critical Perimeter (for punching shear):</strong></p>
                <div class="formula">$$b_0 = 2(b_1 + d) + 2(d + b_2) = 2({C31:.3f} + {C54:.3f}) + 2({C54:.3f} + {C32:.3f}) = \\textbf{{{C55:.3f} m}}$$</div>
                
                <p><strong>Critical Area:</strong></p>
                <div class="formula">$$A_0 = d \\times b_0 = {C54:.3f} \\times {C55:.3f} = \\textbf{{{C56:.3f} m²}}$$</div>
                
                <p><strong>Ultimate Pressure:</strong></p>
                <div class="formula">$$\\sigma_u = \\frac{{F_z}}{{B \\times L}} \\times \\left(1 + \\frac{{6e_x}}{{B}} + \\frac{{6e_y}}{{L}}\\right) = \\textbf{{{C53:.2f} kPa}}$$</div>
                
                <p><strong>Size Effect Factor:</strong></p>
                <div class="formula">$$\\lambda_s = \\min\\left(1, \\sqrt{{\\frac{{2}}{{1 + d \\times 1000/254}}}}\\right) = \\textbf{{{C61:.4f}}}$$</div>
            </div>
            
            <h2>Step 4: Two-Way (Punching) Shear Check — ACI 318-19 §22.6</h2>
            <div class="calc-step">
                <p><strong>Applied Punching Shear:</strong></p>
                <div class="formula">$$V_u = \\frac{{F_{{z,\\text{{footing}}}}}}{{B \\times L}} \\times [B \\times L - (b_1 + d) \\times (b_2 + d)] = \\textbf{{{K8:.2f} kN}}$$</div>
                
                <p><strong>Capacity Condition 1 (concrete strength):</strong></p>
                <div class="formula">$$V_{{c1}} = 0.75 \\times 0.33 \\times \\lambda_s \\times \\sqrt{{f'_c}} \\times b_0 \\times d \\times 1000 = \\textbf{{{K9:.2f} kN}}$$</div>
                
                <p><strong>Capacity Condition 2 (aspect ratio):</strong></p>
                <div class="formula">$$V_{{c2}} = 0.75 \\times 0.17 \\times \\left(1 + \\frac{{2}}{{\\beta}}\\right) \\times \\lambda_s \\times \\sqrt{{f'_c}} \\times b_0 \\times d \\times 1000 = \\textbf{{{K10:.2f} kN}}$$</div>
                <p><em>where $\\beta = \\max(b_1,b_2) / \\min(b_1,b_2) = {max(C31, C32)/min(C31, C32):.3f}$</em></p>
                
                <p><strong>Capacity Condition 3 (column location):</strong></p>
                <div class="formula">$$V_{{c3}} = 0.75 \\times 0.083 \\times \\left(2 + \\frac{{\\alpha_s \\times d}}{{b_0}}\\right) \\times \\lambda_s \\times \\sqrt{{f'_c}} \\times b_0 \\times d \\times 1000 = \\textbf{{{K11:.2f} kN}}$$</div>
                <p><em>where $\\alpha_s = {C60}$ (interior column)</em></p>
                
                <div class="check {'pass' if K8 < min(K9, K10, K11) else 'fail'}">
                    <strong>Check: $V_u \\leq \\min(V_{{c1}}, V_{{c2}}, V_{{c3}})$</strong><br>
                    {K8:.2f} kN {'≤' if K8 < min(K9, K10, K11) else '>'} {min(K9, K10, K11):.2f} kN — <strong>{'PASS ✓' if K8 < min(K9, K10, K11) else 'FAIL ✗'}</strong>
                </div>
            </div>
            
            <h2>Step 5: One-Way Shear Check (Beam Action) — ACI 318-19 §22.5</h2>
            <div class="calc-step">
                <p><strong>Applied One-Way Shear:</strong></p>
                <div class="formula">$$V_u = \\sigma_u \\times B \\times \\left(\\frac{{L}}{{2}} - \\frac{{b_1}}{{2}} - d\\right) = {C53:.2f} \\times {C25:.2f} \\times \\left(\\frac{{{C26:.2f}}}{{2}} - \\frac{{{C31:.3f}}}{{2}} - {C54:.3f}\\right) = \\textbf{{{K16:.2f} kN}}$$</div>
                
                <p><strong>Shear Stress Demand:</strong></p>
                <div class="formula">$$\\sigma_v = \\frac{{V_u}}{{B \\times d}} = \\frac{{{K16:.2f}}}{{{C25:.2f} \\times {C54:.3f}}} = \\textbf{{{K17:.2f} kPa}}$$</div>
                
                <p><strong>Capacity (along B direction):</strong></p>
                <div class="formula">$$V_{{c1}} = 0.75 \\times 0.17 \\times \\sqrt{{f'_c}} \\times B \\times d \\times 1000 = \\textbf{{{K18:.2f} kN}}$$</div>
                
                <p><strong>Capacity (along L direction):</strong></p>
                <div class="formula">$$V_{{c2}} = 0.75 \\times 0.17 \\times \\sqrt{{f'_c}} \\times L \\times d \\times 1000 = \\textbf{{{K19:.2f} kN}}$$</div>
                
                <p><strong>Governing Capacity:</strong></p>
                <div class="formula">$$V_c = \\min(V_{{c1}}, V_{{c2}}) = \\textbf{{{K20:.2f} kN}}$$</div>
                
                <div class="check {'pass' if K16 < K20 else 'fail'}">
                    <strong>Check: $V_u \\leq V_c$</strong><br>
                    {K16:.2f} kN {'≤' if K16 < K20 else '>'} {K20:.2f} kN — <strong>{'PASS ✓' if K16 < K20 else 'FAIL ✗'}</strong>
                </div>
            </div>
            
            <h2>Step 6: Flexure for Reinforcement — ACI 318-19 §13.2.7.1 & Ch. 22</h2>
            <div class="calc-step">
                <p><strong>Factored Line Loads:</strong></p>
                <div class="formula">$$w_u \\text{{ (along B)}} = \\sigma_u \\times L = {C53:.2f} \\times {C26:.2f} = \\textbf{{{K25:.2f} kN/m}}$$</div>
                <div class="formula">$$w_u \\text{{ (along L)}} = \\sigma_u \\times B = {C53:.2f} \\times {C25:.2f} = \\textbf{{{K26:.2f} kN/m}}$$</div>
                
                <p><strong>Factored Moments (at critical section):</strong></p>
                <div class="formula">$$M_u \\text{{ (along B)}} = w_u \\times \\frac{{(L/2 - b_2/2)^2}}{{2}} = {K26:.2f} \\times \\frac{{({C26:.2f}/2 - {C32:.3f}/2)^2}}{{2}} = \\textbf{{{K27:.2f} kN·m}}$$</div>
                <div class="formula">$$M_u \\text{{ (along L)}} = w_u \\times \\frac{{(B/2 - b_1/2)^2}}{{2}} = {K25:.2f} \\times \\frac{{({C25:.2f}/2 - {C31:.3f}/2)^2}}{{2}} = \\textbf{{{K28:.2f} kN·m}}$$</div>
            </div>
            
            <h2>Step 7: Concrete Bearing at Column/Pedestal Base — ACI 318-19 §22.8.3.2</h2>
            <div class="calc-step">
                <p><strong>Applied Load:</strong></p>
                <div class="formula">$$P_u = F_{{z,\\text{{footing}}}} = \\textbf{{{K32:.2f} kN}}$$</div>
                
                <p><strong>Supporting Area:</strong></p>
                <div class="formula">$$A_2 = B \\times L = {C25:.2f} \\times {C26:.2f} = \\textbf{{{K33:.2f} m²}}$$</div>
                
                <p><strong>Area Factor:</strong></p>
                <div class="formula">$$\\sqrt{{\\frac{{A_2}}{{A_1}}}} = \\sqrt{{\\frac{{{K33:.2f}}}{{{C31:.3f} \\times {C32:.3f}}}}} = \\textbf{{{K34:.3f}}}$$</div>
                <p><em>Limited to max value of 2.0</em></p>
                
                <p><strong>Bearing Resistance:</strong></p>
                <div class="formula">$$\\phi P_n = \\min\\left(0.65 \\times 0.85 \\times f'_c \\times A_1 \\times \\sqrt{{\\frac{{A_2}}{{A_1}}}}, 0.65 \\times 0.85 \\times f'_c \\times A_1 \\times 2\\right) \\times 1000 = \\textbf{{{K35:.2f} kN}}$$</div>
                
                <div class="check {'pass' if K32 < K35 else 'fail'}">
                    <strong>Check: $P_u \\leq \\phi P_n$</strong><br>
                    {K32:.2f} kN {'≤' if K32 < K35 else '>'} {K35:.2f} kN — <strong>{'PASS ✓' if K32 < K35 else 'FAIL ✗'}</strong>
                </div>
            </div>
            
            <h2>Step 8: Rebar Spacing (Strip Method)</h2>
            <div class="calc-step">
                <p><strong>Cover and Effective Depth:</strong></p>
                <div class="formula">$$c_c = (H - d) \\times 1000 - \\frac{{d_b}}{{2}} = ({C27:.2f} - {C54:.3f}) \\times 1000 - \\frac{{{K39}}}{{2}} = \\textbf{{{K40:.1f} mm}}$$</div>
                <div class="formula">$$d = H \\times 1000 - c_c - \\frac{{d_b}}{{2}} = {C27:.2f} \\times 1000 - {K40:.1f} - \\frac{{{K39}}}{{2}} = \\textbf{{{K41:.1f} mm}}$$</div>
                
                <h3>X-Direction (along B):</h3>
                <div class="formula">$$A_{{s,\\min}} = 0.0018 \\times L \\times H = 0.0018 \\times {C26*1000:.0f} \\times {C27*1000:.0f} = \\textbf{{{K44:.1f} mm²}}$$</div>
                <div class="formula">$$A_{{s,\\text{{required}}}} \\text{{ (from flexure)}} = \\textbf{{{K45:.2f} mm²}}$$</div>
                <div class="formula">$$A_{{s,\\text{{target}}}} = \\max(A_{{s,\\min}}, A_{{s,\\text{{required}}}}) = \\textbf{{{K46:.1f} mm²}}$$</div>
                <div class="formula">$$n = \\left\\lceil \\frac{{A_{{s,\\text{{target}}}}}}{{\\pi \\times d_b^2/4}} \\right\\rceil = \\textbf{{{K47} bars}}$$</div>
                <div class="formula">$$s_{{\\text{{clear}}}} = \\frac{{(L \\times 1000) - 2c_c - n \\times d_b}}{{n - 1}} = \\textbf{{{K48:.1f} mm}}$$</div>
                <div class="formula">$$s_{{\\text{{c/c}}}} = s_{{\\text{{clear}}}} + d_b = \\textbf{{{K49:.1f} mm}}$$</div>
                
                <h3>Y-Direction (along L):</h3>
                <div class="formula">$$A_{{s,\\min}} = 0.0018 \\times L \\times H = \\textbf{{{K51:.1f} mm²}}$$</div>
                <div class="formula">$$A_{{s,\\text{{required}}}} \\text{{ (from flexure)}} = \\textbf{{{K52:.2f} mm²}}$$</div>
                <div class="formula">$$A_{{s,\\text{{target}}}} = \\max(A_{{s,\\min}}, A_{{s,\\text{{required}}}}) = \\textbf{{{K53:.1f} mm²}}$$</div>
                <div class="formula">$$n = \\left\\lceil \\frac{{A_{{s,\\text{{target}}}}}}{{\\pi \\times d_b^2/4}} \\right\\rceil = \\textbf{{{K54} bars}}$$</div>
                <div class="formula">$$s_{{\\text{{clear}}}} = \\frac{{(B \\times 1000) - 2c_c - n \\times d_b}}{{n - 1}} = \\textbf{{{K55:.1f} mm}}$$</div>
                <div class="formula">$$s_{{\\text{{c/c}}}} = s_{{\\text{{clear}}}} + d_b = \\textbf{{{K56:.1f} mm}}$$</div>
            </div>
            
        </body>
        </html>
        """
        
        return vkt.WebResult(html=html)
    
    @vkt.DataView("Design Results", duration_guess=1)
    def get_results(self, params, **kwargs):
        # Extract parameters using exact Excel variable names
        C5 = params.section_loads.Fz
        C6 = params.section_loads.Fy
        C7 = params.section_loads.Fx
        C8 = params.section_loads.Mx
        C9 = params.section_loads.My
        
        C14 = params.section_material.fc
        C15 = params.section_material.fy
        C16 = params.section_material.gamma_fill
        
        C25 = params.section_footing.B
        C26 = params.section_footing.L
        C27 = params.section_footing.H
        
        C31 = params.section_pedestal.b1
        C32 = params.section_pedestal.b2
        C33 = params.section_pedestal.ph
        
        K39 = params.section_rebar.db
        
        # Foundation Weights
        C36 = 24  # Concrete specific weight [kN/m³]
        C37 = C25 * C26 * C27 * C36  # Slab weight [kN]
        C38 = C33 * C32 * C31 * C36  # Pedestal weight [kN]
        C39 = ((C25 * C26 * C27) * (C33 + C27) - (C31 * C32 * C33)) * C16  # Soil weight [kN]
        C40 = C37 + C38 + C39  # Total weight [kN]
        
        # Factored Actions at footing slab level
        C43 = C40 + C5  # Axial Force at footing level [kN]
        C46 = C8 + C7 * (C33 + C27 / 2)  # Longitudinal Moment at footing level [kN·m]
        C47 = C9 + C6 * (C33 + C27 / 2)  # Transverse Moment at footing level [kN·m]
        C48 = C46 / C43  # Eccentricity (x-direction) [m]
        C49 = C47 / C43  # Eccentricity (y-direction) [m]
        
        # Effective Depth and Critical Sections
        C54 = C27 - 0.09  # Effective depth (d) [m]
        C55 = 2 * (C31 + C54) + 2 * (C54 + C32)  # Critical perimeter (b0) [m]
        C56 = C54 * C55  # Critical area (A0) [m²]
        C53 = (C5 / (C25 * C26)) * (1 + 6 * C48 / C25 + 6 * C49 / C26)  # Ultimate pressure (σu) [kPa]
        C57 = C43 - C53 * (C31 + C54) * (C32 + C54)  # Applied punching shear (Vu) [kN]
        C58 = C57 / C56  # Shear stress demand (σv) [kPa]
        
        # Size effect factor and column location
        C60 = 40  # Column Location (as) [40 for interior]
        C61 = min(1, math.sqrt(2 / (1 + (C54 * 1000) / 254)))  # Size effect factor (λs)
        C59 = 0.75 * min(
            0.33 * C61,
            0.17 * (1 + 2 / (max(C31, C32) / min(C31, C32))) * C61,
            0.083 * (2 + C60 * C54 / C55) * C61
        ) * math.sqrt(C14) * 1000  # Concrete shear capacity (stress) (Vc) [kPa]
        
        # Two-Way (Punching) Shear Check
        K8 = (C43 / (C25 * C26) * (C25 * C26 - (C31 + C54) * (C32 + C54)))  # Applied punching shear (Vu) [kN]
        K9 = 0.75 * 0.33 * C61 * math.sqrt(C14) * C55 * C54 * 1000  # Capacity (Condition 1) (Vc1) [kN]
        K10 = 0.75 * 0.17 * (1 + 2 / (max(C31, C32) / min(C31, C32))) * C61 * math.sqrt(C14) * C55 * C54 * 1000  # Capacity (Condition 2) (Vc2) [kN]
        K11 = 0.75 * 0.083 * (2 + C60 * C54 / C55) * C61 * math.sqrt(C14) * C55 * C54 * 1000  # Capacity (Condition 3) (Vc3) [kN]
        
        # One-Way Shear Check
        K16 = C53 * C25 * ((C26 / 2 - C31 / 2 - C54))  # Applied one-way shear (Vu) [kN]
        K17 = K16 / (C25 * C54)  # Shear stress demand (σv) [kPa]
        K18 = 0.75 * 0.17 * math.sqrt(C14) * C25 * C54 * 1000  # Capacity (Condition 1) (Vc1) [kN]
        K19 = 0.75 * 0.17 * math.sqrt(C14) * C26 * C54 * 1000  # Capacity (Condition 2) (Vc2) [kN]
        K20 = min(K18, K19)  # Governing capacity [kN]
        
        # Flexure for Reinforcement
        K25 = C53 * C26  # Factored line load (from pressure along B) (wu) [kN/m]
        K26 = C53 * C25  # Factored line load (from pressure along L) (wu) [kN/m]
        K27 = K26 * (C26 / 2 - C32 / 2) ** 2 / 2  # Factored moment along B (Mu) [kN·m]
        K28 = K25 * (C25 / 2 - C31 / 2) ** 2 / 2  # Factored moment along L (Mu) [kN·m]
        
        # Concrete Bearing at Column/Pedestal Base
        K32 = C43  # Applied load (Pu) [kN]
        K33 = C25 * C26  # Supporting (effective) area (A2) [m²]
        K34 = math.sqrt(K33 / (C31 * C32))  # Area factor (√(A2/A1))
        K35 = min(1000 * 0.65 * 0.85 * C14 * (C31 * C32) * K34, 1000 * 0.65 * 0.85 * C14 * (C31 * C32) * 2)  # Bearing resistance (φPn) [kN]
        
        # Rebar spacing calculations
        K40 = (C27 - C54) * 1000 - K39 / 2  # Cover (cc) [mm]
        K41 = C27 * 1000 - K40 - K39 / 2  # Effective depth (d) [mm]
        
        # X-direction (along B)
        K44 = 0.0018 * (C26 * 1000) * (C27 * 1000)  # min As [mm²]
        K45 = ((1 / (C15 / (0.85 * C14))) * (1 - math.sqrt(1 - (2 * ((K28 * 1000000) / (0.9 * (C26 * 1000) * (K41 ** 2))) * (C15 / (0.85 * C14))) / C15))) * (C26 * 1000) * K41  # required As [mm²]
        K46 = max(K44, K45)  # As,target (x-dir) [mm²]
        K47 = math.ceil(K46 / (math.pi * (K39 ** 2) / 4))  # Required # bars (n) (x-dir)
        K48 = ((C26 * 1000) - 2 * K40 - K47 * K39) / (K47 - 1) if K47 > 1 else 0  # Required clear spacing (s_clear) (x-dir) [mm]
        K49 = K48 + K39 if K48 else 0  # Required spacing (s c/c) (x-dir) [mm]
        
        # Y-direction (along L)
        K51 = 0.0018 * (C26 * 1000) * (C27 * 1000)  # min As [mm²]
        K52 = ((1 / (C15 / (0.85 * C14))) * (1 - math.sqrt(1 - (2 * ((K27 * 1000000) / (0.9 * (C25 * 1000) * (K41 ** 2))) * (C15 / (0.85 * C14))) / C15))) * (C25 * 1000) * K41  # required As [mm²]
        K53 = max(K51, K52)  # As,target (y-dir) [mm²]
        K54 = math.ceil(K53 / (math.pi * (K39 ** 2) / 4))  # Required # bars (n) (y-dir)
        K55 = ((C25 * 1000) - 2 * K40 - K54 * K39) / (K54 - 1) if K54 > 1 else 0  # Required clear spacing (s_clear) (y-dir) [mm]
        K56 = K55 + K39 if K55 else 0  # Required spacing (s c/c) (y-dir) [mm]
        
        # Build DataResult
        data = vkt.DataGroup()
        
        # Two-Way (Punching) Shear Check
        punching_shear = vkt.DataGroup(
            vkt.DataItem("Applied punching shear (Vu)", K8, suffix="kN"),
            vkt.DataItem("Capacity (Condition 1) (Vc1)", K9, suffix="kN"),
            vkt.DataItem("Capacity (Condition 2) (Vc2)", K10, suffix="kN"),
            vkt.DataItem("Capacity (Condition 3) (Vc3)", K11, suffix="kN"),
        )
        data.add(vkt.DataItem("Two-Way (Punching) Shear Check — ACI 318-19 §22.6", subgroup=punching_shear))
        
        # One-Way Shear Check
        one_way_shear = vkt.DataGroup(
            vkt.DataItem("Applied one-way shear (Vu)", K16, suffix="kN"),
            vkt.DataItem("Shear stress demand (σv)", K17, suffix="kPa"),
            vkt.DataItem("Capacity (Condition 1) (Vc1)", K18, suffix="kN"),
            vkt.DataItem("Capacity (Condition 2) (Vc2)", K19, suffix="kN"),
            vkt.DataItem("Governing capacity", K20, suffix="kN"),
        )
        data.add(vkt.DataItem("One-Way Shear Check (Beam Action) — ACI 318-19 §22.5", subgroup=one_way_shear))
        
        # Flexure for Reinforcement
        flexure = vkt.DataGroup(
            vkt.DataItem("Factored line load (from pressure along B) (wu)", K25, suffix="kN/m"),
            vkt.DataItem("Factored line load (from pressure along L) (wu)", K26, suffix="kN/m"),
            vkt.DataItem("Factored moment along B (Mu)", K27, suffix="kN·m"),
            vkt.DataItem("Factored moment along L (Mu)", K28, suffix="kN·m"),
        )
        data.add(vkt.DataItem("Flexure for Reinforcement — ACI 318-19 §13.2.7.1 & Ch. 22", subgroup=flexure))
        
        # Concrete Bearing
        bearing = vkt.DataGroup(
            vkt.DataItem("Applied load (Pu)", K32, suffix="kN"),
            vkt.DataItem("Supporting (effective) area (A2)", K33, suffix="m²"),
            vkt.DataItem("Area factor (√(A2/A1))", K34, suffix="–"),
            vkt.DataItem("Bearing resistance (φPn)", K35, suffix="kN"),
        )
        data.add(vkt.DataItem("Concrete Bearing at Column/Pedestal Base — ACI 318-19 §22.8.3.2", subgroup=bearing))
        
        # Rebar spacing (X-direction)
        rebar_x = vkt.DataGroup(
            vkt.DataItem("Bar diameter (db)", K39, suffix="mm"),
            vkt.DataItem("Cover (cc)", K40, suffix="mm"),
            vkt.DataItem("Effective depth (d)", K41, suffix="mm"),
            vkt.DataItem("min As", K44, suffix="mm²"),
            vkt.DataItem("required As", K45, suffix="mm²"),
            vkt.DataItem("As,target (x-dir)", K46, suffix="mm²"),
            vkt.DataItem("Required # bars (n) (x-dir)", K47, suffix=""),
            vkt.DataItem("Required clear spacing (s_clear) (x-dir)", K48, suffix="mm"),
            vkt.DataItem("Required spacing (s c/c) (x-dir)", K49, suffix="mm"),
        )
        data.add(vkt.DataItem("Rebar spacing (strip method) — X-direction", subgroup=rebar_x))
        
        # Rebar spacing (Y-direction)
        rebar_y = vkt.DataGroup(
            vkt.DataItem("min As", K51, suffix="mm²"),
            vkt.DataItem("required As", K52, suffix="mm²"),
            vkt.DataItem("As,target (y-dir)", K53, suffix="mm²"),
            vkt.DataItem("Required # bars (n) (y-dir)", K54, suffix=""),
            vkt.DataItem("Required clear spacing (s_clear) (y-dir)", K55, suffix="mm"),
            vkt.DataItem("Required spacing (s c/c) (y-dir)", K56, suffix="mm"),
        )
        data.add(vkt.DataItem("Rebar spacing (strip method) — Y-direction", subgroup=rebar_y))
        
        return vkt.DataResult(data)
 