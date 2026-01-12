import json
from pathlib import Path

import viktor as vkt

from app.truss_beam import RectangularTrussBeam
from app.opensees.model import Model
from app.types import CrossSectionInfo, NodesInfoDict, LinesInfoDict, MembersDict, CrossSectionsDict, LoadCase, NodalLoad
from app.plots.deformation import plot_deformed_mesh
from app.plots.loads import plot_loads_3d, plot_wind_loads_3d
from app.plots.model3d import plot_model_3d


class Parametrization(vkt.Parametrization):
    # Step 1: Geometry
    step_1 = vkt.Step("Step 1 - Geometry", views=["show_model"])
    step_1.intro = vkt.Text("""# Rectangular Truss Beam - OpenSees Analysis

Define the truss beam geometry and cross-section parameters below.""")
    step_1.truss_length = vkt.NumberField("Truss Length", min=100, default=10000, suffix="mm")
    step_1.truss_width = vkt.NumberField("Truss Width", min=100, default=1000, suffix="mm")
    step_1.truss_height = vkt.NumberField("Truss Height", min=100, default=1500, suffix="mm")
    step_1.n_divisions = vkt.NumberField("Number of Divisions", min=1, default=6)
    step_1.section_title = vkt.Text("""## Cross-Section
Select a cross section size for the truss members:""")
    step_1.cross_section = vkt.OptionField(
        "Cross-Section Size", 
        options=["SHS50x4", "SHS75x4", "SHS100x4", "SHS150x4"], 
        default="SHS100x4"
    )
    
    # Step 2: Loads (Gravitational + Wind)
    step_2 = vkt.Step("Step 2 - Loads", views=["show_loads", "show_wind_loads"])
    step_2.gravity_intro = vkt.Text("""# Gravitational Loads
Define the gravitational load to apply to the truss beam. The load will be visualized as point loads at all nodes in the XY plane (z=0).""")
    step_2.load_q = vkt.NumberField("Load Q", min=0, default=5, suffix="kPa")
    step_2.wind_intro = vkt.Text("""# Wind Loads
Define the wind pressure to apply to the truss beam. The load will be visualized as point loads at nodes on the windward face (ZX plane at y=width, where z is between 0 and the truss height). Wind direction is in the positive Y direction.""")
    step_2.wind_pressure = vkt.NumberField("Wind Pressure", min=0, default=1, suffix="kPa")

    # Step 3: Run Model
    step_3 = vkt.Step("Step 3 - Run Model", views=["show_deformation"])
    step_3.intro = vkt.Text("""# Run OpenSees Analysis
Run the structural analysis using OpenSees. The model will be generated based on the geometry defined in Step 1.

The analysis runs 7 SLS load combinations and identifies the critical one:
- SLS + Q
- SLS + Q + WL
- SLS + Q - WL
- SLS + 0.6Q + WL
- SLS + 0.6Q - WL
- SLS + WL
- SLS - WL

Where **SLS** = Self-weight, **Q** = Gravitational load, **WL** = Wind load.""")
    step_3.deform_scale = vkt.NumberField("Deformation Scale", min=1, max=500, default=25, step=1)
    step_3.download_text = vkt.Text("""## Download Results
This button downloads the results in a JSON file. It gets the results for each combination along the model parameters""")
    step_3.download_btn = vkt.DownloadButton("Download Results", method="download_results")

    # Step 4: Sensitivity Analysis
    step_4 = vkt.Step("Step 4 - Sensitivity Analysis", views=["show_sensitivity_analysis"])
    step_4.intro = vkt.Text("""# Sensitivity Analysis
Run a sensitivity analysis to see how the truss depth (height) affects the maximum vertical deformation.

The analysis will vary the truss height from the minimum to maximum values and plot the results.""")
    step_4.min_height = vkt.NumberField("Minimum Height", min=100, default=500, suffix="mm")
    step_4.max_height = vkt.NumberField("Maximum Height", min=100, default=3000, suffix="mm")
    step_4.n_steps = vkt.NumberField("Number of Steps", min=3, max=20, default=10)

class Controller(vkt.Controller):
    parametrization = Parametrization
    
    @vkt.PlotlyView("3D Model", duration_guess=5)
    def show_model(self, params, **kwargs):
        """Create 3D visualization of the truss beam using Plotly."""
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for visualization)
        beam = RectangularTrussBeam(
            length=params.step_1.truss_length,
            width=params.step_1.truss_width,
            height=params.step_1.truss_height,
            n_diagonals=int(params.step_1.n_divisions),
        )
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Convert nodes to NodesInfoDict format (with id field)
        nodes_dict: NodesInfoDict = {}
        for node_id, node_data in nodes.items():
            nodes_dict[node_id] = {
                "id": node_id,
                "x": node_data["x"],
                "y": node_data["y"],
                "z": node_data["z"],
            }
        
        # Convert lines to LinesInfoDict format (with Ni, Nj, Type)
        lines_dict: LinesInfoDict = {}
        for line_id, line_data in lines.items():
            lines_dict[line_id] = {
                "id": line_id,
                "Ni": line_data["NodeI"],
                "Nj": line_data["NodeJ"],
                "Type": "Truss Chord",
            }
        
        # Create cross-sections dict
        cross_sections: CrossSectionsDict = {
            selected_cs["id"]: selected_cs
        }
        
        # Create members (assign selected cross-section to all lines)
        members: MembersDict = {}
        for line_id in lines_dict.keys():
            members[line_id] = {
                "line_id": line_id,
                "cross_section_id": selected_cs["id"],
                "material_name": "Steel",
            }
        
        # Create the 3D model visualization plot
        fig = plot_model_3d(
            nodes=nodes_dict,
            lines=lines_dict,
            members=members,
            cross_sections=cross_sections,
        )
        
        return vkt.PlotlyResult(fig)
    
    def download_results(self, params, **kwargs):
        """Download JSON file with max displacements from OpenSees analysis."""
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for OpenSees)
        beam = RectangularTrussBeam(
            length=params.step_1.truss_length,
            width=params.step_1.truss_width,
            height=params.step_1.truss_height,
            n_diagonals=int(params.step_1.n_divisions),
        )
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Convert nodes to NodesInfoDict format (with id field)
        nodes_dict: NodesInfoDict = {}
        for node_id, node_data in nodes.items():
            nodes_dict[node_id] = {
                "id": node_id,
                "x": node_data["x"],
                "y": node_data["y"],
                "z": node_data["z"],
            }
        
        # Convert lines to LinesInfoDict format (with Ni, Nj, Type)
        lines_dict: LinesInfoDict = {}
        for line_id, line_data in lines.items():
            lines_dict[line_id] = {
                "id": line_id,
                "Ni": line_data["NodeI"],
                "Nj": line_data["NodeJ"],
                "Type": "Truss Chord",
            }
        
        # Create cross-sections dict
        cross_sections: CrossSectionsDict = {
            selected_cs["id"]: selected_cs
        }
        
        # Create members (assign selected cross-section to all lines)
        members: MembersDict = {}
        for line_id in lines_dict.keys():
            members[line_id] = {
                "line_id": line_id,
                "cross_section_id": selected_cs["id"],
                "material_name": "Steel",
            }
        
        # Identify support nodes: z=0 AND (x=0 OR x=length)
        from app.opensees.utils import get_nodes_by_x_and_z
        length = params.step_1.truss_length
        width = params.step_1.truss_width
        height = params.step_1.truss_height
        support_nodes_start = get_nodes_by_x_and_z(nodes_dict, x=0, z=0)
        support_nodes_end = get_nodes_by_x_and_z(nodes_dict, x=length, z=0)
        support_nodes = support_nodes_start + support_nodes_end
        
        # Get load values from params
        load_q = params.step_2.load_q or 0.0  # kPa
        wind_pressure = params.step_2.wind_pressure or 0.0  # kPa
        
        # Create load cases
        load_cases: list[LoadCase] = []
        
        # --- Gravitational loads (Dead Load) ---
        # Get nodes where z=0 (bottom chord nodes in XY plane)
        # Convert kPa to N/mm²: 1 kPa = 0.001 N/mm²
        # Then multiply by tributary area to get point load in N
        if load_q > 0:
            dead_loads: list[NodalLoad] = []
            # Nodes at z=0 (bottom chord)
            gravity_nodes = [
                node_id for node_id, node_data in nodes_dict.items()
                if abs(node_data["z"]) < 1e-6
            ]
            # Calculate tributary length per node (approximate: length / (n_divisions + 1))
            n_divisions = int(params.step_1.n_divisions)
            trib_length = length / (n_divisions + 1)  # mm
            trib_width = width  # mm (full width)
            trib_area = trib_length * trib_width  # mm²
            
            # Convert kPa to N/mm²: 1 kPa = 1 kN/m² = 0.001 N/mm²
            pressure_n_mm2 = load_q * 0.001  # N/mm²
            point_load_n = pressure_n_mm2 * trib_area  # N
            
            for node_id in gravity_nodes:
                dead_loads.append({
                    "node_id": node_id,
                    "fx": 0.0,
                    "fy": 0.0,
                    "fz": -point_load_n,  # Negative z = downward
                    "mx": 0.0,
                    "my": 0.0,
                    "mz": 0.0,
                })
            
            load_cases.append({
                "name": "Dead Load",
                "factor": 1.0,  # Will be updated when combinations are added
                "loads": dead_loads,
            })
        
        # --- Wind loads ---
        # Nodes at y=0, z between 0 and height
        if wind_pressure > 0:
            wind_loads: list[NodalLoad] = []
            wind_nodes = [
                node_id for node_id, node_data in nodes_dict.items()
                if (abs(node_data["y"]) < 1e-6 and
                    node_data["z"] >= -1e-6 and
                    node_data["z"] <= height + 1e-6)
            ]
            # Calculate tributary area for wind (height x tributary length)
            n_divisions = int(params.step_1.n_divisions)
            trib_length = length / (n_divisions + 1)  # mm
            trib_height = height  # mm
            trib_area_wind = trib_length * trib_height  # mm²
            
            # Convert kPa to N/mm²
            pressure_n_mm2 = wind_pressure * 0.001  # N/mm²
            point_load_n = pressure_n_mm2 * trib_area_wind  # N
            
            for node_id in wind_nodes:
                wind_loads.append({
                    "node_id": node_id,
                    "fx": 0.0,
                    "fy": point_load_n,  # Positive y = wind direction
                    "fz": 0.0,
                    "mx": 0.0,
                    "my": 0.0,
                    "mz": 0.0,
                })
            
            load_cases.append({
                "name": "Wind Load",
                "factor": 1.0,  # Will be updated when combinations are added
                "loads": wind_loads,
            })
        
        # Create and run the OpenSees model with all load combinations
        model = Model(
            nodes=nodes_dict,
            lines=lines_dict,
            cross_sections=cross_sections,
            members=members,
            support_nodes=support_nodes,
            load_cases=load_cases,
        )
        
        # Run all SLS combinations and find the critical one
        critical_result, all_results = model.run_all_combinations()
        
        # Calculate max displacements in x, y, z from the critical combination
        disp_dict = critical_result["disp_dict"]
        max_dx = max(abs(d["dx"]) for d in disp_dict.values()) if disp_dict else 0.0
        max_dy = max(abs(d["dy"]) for d in disp_dict.values()) if disp_dict else 0.0
        max_dz = max(abs(d["dz"]) for d in disp_dict.values()) if disp_dict else 0.0
        
        # Create JSON result with max displacements and critical combination info
        result_data = {
            "critical_combination": critical_result["combination_name"],
            "max_displacements_mm": {
                "dx": round(max_dx, 4),
                "dy": round(max_dy, 4),
                "dz": round(max_dz, 4),
            },
            "model_parameters": {
                "truss_length_mm": params.step_1.truss_length,
                "truss_width_mm": params.step_1.truss_width,
                "truss_height_mm": params.step_1.truss_height,
                "n_divisions": int(params.step_1.n_divisions),
                "cross_section": selected_cs_name,
                "load_q_kPa": load_q,
                "wind_pressure_kPa": wind_pressure,
            },
            "all_combinations_results": [
                {
                    "combination_name": r["combination_name"],
                    "max_abs_displacement_mm": round(r["max_abs_displacement"], 4),
                }
                for r in all_results
            ],
        }
        
        # Convert to JSON string
        json_content = json.dumps(result_data, indent=2)
        
        return vkt.DownloadResult(json_content, "opensees_results.json")

    @vkt.PlotlyView("Deformed Shape", duration_guess=5)
    def show_deformation(self, params, **kwargs):
        """Display the deformed mesh using Plotly."""
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for OpenSees)
        beam = RectangularTrussBeam(
            length=params.step_1.truss_length,
            width=params.step_1.truss_width,
            height=params.step_1.truss_height,
            n_diagonals=int(params.step_1.n_divisions),
        )
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Convert nodes to NodesInfoDict format (with id field)
        nodes_dict: NodesInfoDict = {}
        for node_id, node_data in nodes.items():
            nodes_dict[node_id] = {
                "id": node_id,
                "x": node_data["x"],
                "y": node_data["y"],
                "z": node_data["z"],
            }
        
        # Convert lines to LinesInfoDict format (with Ni, Nj, Type)
        lines_dict: LinesInfoDict = {}
        for line_id, line_data in lines.items():
            lines_dict[line_id] = {
                "id": line_id,
                "Ni": line_data["NodeI"],
                "Nj": line_data["NodeJ"],
                "Type": "Truss Chord",
            }
        
        # Create cross-sections dict
        cross_sections: CrossSectionsDict = {
            selected_cs["id"]: selected_cs
        }
        
        # Create members (assign selected cross-section to all lines)
        members: MembersDict = {}
        for line_id in lines_dict.keys():
            members[line_id] = {
                "line_id": line_id,
                "cross_section_id": selected_cs["id"],
                "material_name": "Steel",
            }
        
        # Identify support nodes: z=0 AND (x=0 OR x=length)
        from app.opensees.utils import get_nodes_by_x_and_z
        length = params.step_1.truss_length
        width = params.step_1.truss_width
        height = params.step_1.truss_height
        support_nodes_start = get_nodes_by_x_and_z(nodes_dict, x=0, z=0)
        support_nodes_end = get_nodes_by_x_and_z(nodes_dict, x=length, z=0)
        support_nodes = support_nodes_start + support_nodes_end
        
        # Get load values from params
        load_q = params.step_2.load_q or 0.0  # kPa
        wind_pressure = params.step_2.wind_pressure or 0.0  # kPa
        
        # Create load cases (same logic as run_opensees_model)
        load_cases: list[LoadCase] = []
        
        if load_q > 0:
            dead_loads: list[NodalLoad] = []
            gravity_nodes = [
                node_id for node_id, node_data in nodes_dict.items()
                if abs(node_data["z"]) < 1e-6
            ]
            n_divisions = int(params.step_1.n_divisions)
            trib_length = length / (n_divisions + 1)
            trib_area = trib_length * width
            pressure_n_mm2 = load_q * 0.001
            point_load_n = pressure_n_mm2 * trib_area
            
            for node_id in gravity_nodes:
                dead_loads.append({
                    "node_id": node_id,
                    "fx": 0.0, "fy": 0.0, "fz": -point_load_n,
                    "mx": 0.0, "my": 0.0, "mz": 0.0,
                })
            load_cases.append({"name": "Dead Load", "factor": 1.0, "loads": dead_loads})
        
        if wind_pressure > 0:
            wind_loads: list[NodalLoad] = []
            wind_nodes = [
                node_id for node_id, node_data in nodes_dict.items()
                if (abs(node_data["y"]) < 1e-6 and
                    node_data["z"] >= -1e-6 and
                    node_data["z"] <= height + 1e-6)
            ]
            n_divisions = int(params.step_1.n_divisions)
            trib_length = length / (n_divisions + 1)
            trib_area_wind = trib_length * height
            pressure_n_mm2 = wind_pressure * 0.001
            point_load_n = pressure_n_mm2 * trib_area_wind
            
            for node_id in wind_nodes:
                wind_loads.append({
                    "node_id": node_id,
                    "fx": 0.0, "fy": point_load_n, "fz": 0.0,
                    "mx": 0.0, "my": 0.0, "mz": 0.0,
                })
            load_cases.append({"name": "Wind Load", "factor": 1.0, "loads": wind_loads})
        
        # Create the OpenSees model
        model = Model(
            nodes=nodes_dict,
            lines=lines_dict,
            cross_sections=cross_sections,
            members=members,
            support_nodes=support_nodes,
            load_cases=load_cases,
        )
        
        # Run all SLS combinations and find the critical one
        critical_result, all_results = model.run_all_combinations()
        
        # Get deformation scale from params
        deform_scale = params.step_3.deform_scale or 25
        
        # Create the deformed mesh plot with critical combination results
        fig = plot_deformed_mesh(
            nodes=nodes_dict,
            lines=lines_dict,
            members=members,
            cross_sections=cross_sections,
            disp_dict=critical_result["disp_dict"],
            scale=deform_scale,
            critical_combination_name=critical_result["combination_name"],
        )
        
        return vkt.PlotlyResult(fig)

    @vkt.PlotlyView("Loads Visualization", duration_guess=5)
    def show_loads(self, params, **kwargs):
        """Display the 3D model with gravitational point loads at z=0 nodes."""
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for visualization)
        beam = RectangularTrussBeam(
            length=params.step_1.truss_length,
            width=params.step_1.truss_width,
            height=params.step_1.truss_height,
            n_diagonals=int(params.step_1.n_divisions),
        )
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Convert nodes to NodesInfoDict format (with id field)
        nodes_dict: NodesInfoDict = {}
        for node_id, node_data in nodes.items():
            nodes_dict[node_id] = {
                "id": node_id,
                "x": node_data["x"],
                "y": node_data["y"],
                "z": node_data["z"],
            }
        
        # Convert lines to LinesInfoDict format (with Ni, Nj, Type)
        lines_dict: LinesInfoDict = {}
        for line_id, line_data in lines.items():
            lines_dict[line_id] = {
                "id": line_id,
                "Ni": line_data["NodeI"],
                "Nj": line_data["NodeJ"],
                "Type": "Truss Chord",
            }
        
        # Create cross-sections dict
        cross_sections: CrossSectionsDict = {
            selected_cs["id"]: selected_cs
        }
        
        # Create members (assign selected cross-section to all lines)
        members: MembersDict = {}
        for line_id in lines_dict.keys():
            members[line_id] = {
                "line_id": line_id,
                "cross_section_id": selected_cs["id"],
                "material_name": "Steel",
            }
        
        # Get load value from params
        load_q = params.step_2.load_q or 0.0
        
        # Create the loads visualization plot
        fig = plot_loads_3d(
            nodes=nodes_dict,
            lines=lines_dict,
            members=members,
            cross_sections=cross_sections,
            load=load_q,
        )
        
        return vkt.PlotlyResult(fig)

    @vkt.PlotlyView("Wind Loads Visualization", duration_guess=5)
    def show_wind_loads(self, params, **kwargs):
        """Display the 3D model with wind load arrows at nodes on the windward face."""
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for visualization)
        beam = RectangularTrussBeam(
            length=params.step_1.truss_length,
            width=params.step_1.truss_width,
            height=params.step_1.truss_height,
            n_diagonals=int(params.step_1.n_divisions),
        )
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Convert nodes to NodesInfoDict format (with id field)
        nodes_dict: NodesInfoDict = {}
        for node_id, node_data in nodes.items():
            nodes_dict[node_id] = {
                "id": node_id,
                "x": node_data["x"],
                "y": node_data["y"],
                "z": node_data["z"],
            }
        
        # Convert lines to LinesInfoDict format (with Ni, Nj, Type)
        lines_dict: LinesInfoDict = {}
        for line_id, line_data in lines.items():
            lines_dict[line_id] = {
                "id": line_id,
                "Ni": line_data["NodeI"],
                "Nj": line_data["NodeJ"],
                "Type": "Truss Chord",
            }
        
        # Create cross-sections dict
        cross_sections: CrossSectionsDict = {
            selected_cs["id"]: selected_cs
        }
        
        # Create members (assign selected cross-section to all lines)
        members: MembersDict = {}
        for line_id in lines_dict.keys():
            members[line_id] = {
                "line_id": line_id,
                "cross_section_id": selected_cs["id"],
                "material_name": "Steel",
            }
        
        # Get wind pressure from params
        wind_pressure = params.step_2.wind_pressure or 0.0
        truss_width = params.step_1.truss_width
        truss_height = params.step_1.truss_height
        
        # Create the wind loads visualization plot
        fig = plot_wind_loads_3d(
            nodes=nodes_dict,
            lines=lines_dict,
            members=members,
            cross_sections=cross_sections,
            truss_width=truss_width,
            truss_height=truss_height,
            wind_pressure=wind_pressure,
        )
        
        return vkt.PlotlyResult(fig)

    @vkt.PlotlyView("Sensitivity Analysis", duration_guess=30)
    def show_sensitivity_analysis(self, params, **kwargs):
        """Run sensitivity analysis varying truss height and plot max Z deformation."""
        import plotly.graph_objects as go
        from app.opensees.utils import get_nodes_by_x_and_z
        
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Get sensitivity analysis parameters
        min_height = params.step_4.min_height
        max_height = params.step_4.max_height
        n_steps = int(params.step_4.n_steps)
        
        # Fixed parameters from Step 1
        length = params.step_1.truss_length
        width = params.step_1.truss_width
        n_divisions = int(params.step_1.n_divisions)
        
        # Load parameters from Step 2
        load_q = params.step_2.load_q or 0.0
        wind_pressure = params.step_2.wind_pressure or 0.0
        
        # Generate height values to test
        height_values = []
        for i in range(n_steps):
            h = min_height + (max_height - min_height) * i / (n_steps - 1)
            height_values.append(h)
        
        # Store results
        results_heights = []
        results_max_dz = []
        
        # Run analysis for each height
        for height in height_values:
            # Build the truss beam geometry
            beam = RectangularTrussBeam(
                length=length,
                width=width,
                height=height,
                n_diagonals=n_divisions,
            )
            nodes, lines = beam.build()
            nodes, lines = beam.clean_model()
            
            # Convert nodes to NodesInfoDict format
            nodes_dict: NodesInfoDict = {}
            for node_id, node_data in nodes.items():
                nodes_dict[node_id] = {
                    "id": node_id,
                    "x": node_data["x"],
                    "y": node_data["y"],
                    "z": node_data["z"],
                }
            
            # Convert lines to LinesInfoDict format
            lines_dict: LinesInfoDict = {}
            for line_id, line_data in lines.items():
                lines_dict[line_id] = {
                    "id": line_id,
                    "Ni": line_data["NodeI"],
                    "Nj": line_data["NodeJ"],
                    "Type": "Truss Chord",
                }
            
            # Create cross-sections dict
            cross_sections: CrossSectionsDict = {
                selected_cs["id"]: selected_cs
            }
            
            # Create members
            members: MembersDict = {}
            for line_id in lines_dict.keys():
                members[line_id] = {
                    "line_id": line_id,
                    "cross_section_id": selected_cs["id"],
                    "material_name": "Steel",
                }
            
            # Identify support nodes
            support_nodes_start = get_nodes_by_x_and_z(nodes_dict, x=0, z=0)
            support_nodes_end = get_nodes_by_x_and_z(nodes_dict, x=length, z=0)
            support_nodes = support_nodes_start + support_nodes_end
            
            # Create load cases
            load_cases: list[LoadCase] = []
            
            if load_q > 0:
                dead_loads: list[NodalLoad] = []
                gravity_nodes = [
                    node_id for node_id, node_data in nodes_dict.items()
                    if abs(node_data["z"]) < 1e-6
                ]
                trib_length = length / (n_divisions + 1)
                trib_area = trib_length * width
                pressure_n_mm2 = load_q * 0.001
                point_load_n = pressure_n_mm2 * trib_area
                
                for node_id in gravity_nodes:
                    dead_loads.append({
                        "node_id": node_id,
                        "fx": 0.0, "fy": 0.0, "fz": -point_load_n,
                        "mx": 0.0, "my": 0.0, "mz": 0.0,
                    })
                load_cases.append({"name": "Dead Load", "factor": 1.0, "loads": dead_loads})
            
            if wind_pressure > 0:
                wind_loads: list[NodalLoad] = []
                wind_nodes = [
                    node_id for node_id, node_data in nodes_dict.items()
                    if (abs(node_data["y"]) < 1e-6 and
                        node_data["z"] >= -1e-6 and
                        node_data["z"] <= height + 1e-6)
                ]
                trib_length = length / (n_divisions + 1)
                trib_area_wind = trib_length * height
                pressure_n_mm2 = wind_pressure * 0.001
                point_load_n = pressure_n_mm2 * trib_area_wind
                
                for node_id in wind_nodes:
                    wind_loads.append({
                        "node_id": node_id,
                        "fx": 0.0, "fy": point_load_n, "fz": 0.0,
                        "mx": 0.0, "my": 0.0, "mz": 0.0,
                    })
                load_cases.append({"name": "Wind Load", "factor": 1.0, "loads": wind_loads})
            
            # Create and run the OpenSees model
            model = Model(
                nodes=nodes_dict,
                lines=lines_dict,
                cross_sections=cross_sections,
                members=members,
                support_nodes=support_nodes,
                load_cases=load_cases,
            )
            
            # Run all combinations and get critical result
            critical_result, _ = model.run_all_combinations()
            
            # Get max Z displacement
            disp_dict = critical_result["disp_dict"]
            max_dz = max(abs(d["dz"]) for d in disp_dict.values()) if disp_dict else 0.0
            
            # Store results
            results_heights.append(height)
            results_max_dz.append(max_dz)
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=results_heights,
            y=results_max_dz,
            mode='lines+markers',
            name='Max Z Deformation',
            line=dict(color='blue', width=2),
            marker=dict(size=8, color='blue'),
        ))
        
        fig.update_layout(
            title=dict(
                text=f'Sensitivity Analysis: Truss Height vs Max Z Deformation<br><sub>L={length}mm, W={width}mm, Q={load_q}kPa, Wind={wind_pressure}kPa</sub>',
                x=0.5,
                xanchor='center',
            ),
            xaxis_title='Truss Height (mm)',
            yaxis_title='Max Z Deformation (mm)',
            template='plotly_white',
            showlegend=True,
            legend=dict(x=0.95, y=0.95, xanchor='right', yanchor='top'),
        )
        
        return vkt.PlotlyResult(fig)
