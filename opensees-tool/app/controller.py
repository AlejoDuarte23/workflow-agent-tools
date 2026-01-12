import json
from pathlib import Path

import viktor as vkt

from viktor.geometry import Point, Line, RectangularExtrusion
from app.truss_beam import RectangularTrussBeam
from app.opensees.model import Model
from app.types import CrossSectionInfo, NodesInfoDict, LinesInfoDict, MembersDict, CrossSectionsDict, LoadCase, NodalLoad
from app.plots.deformation import plot_deformed_mesh
from app.plots.loads import plot_loads_3d, plot_wind_loads_3d


class Parametrization(vkt.Parametrization):
    # Step 1: Geometry
    step_1 = vkt.Step("Step 1 - Geometry", views=["create_render"])
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
    step_3.run_btn = vkt.ActionButton("Run OpenSees Model", method="run_opensees_model")
    step_3.br1 = vkt.LineBreak()
    step_3.deform_scale = vkt.NumberField("Deformation Scale", min=1, max=500, default=25, step=1)


class Controller(vkt.Controller):
    parametrization = Parametrization
    
    @vkt.GeometryView("3D Model", x_axis_to_right=True)
    def create_render(self, params, **kwargs):
        """Create 3D visualization of the truss beam."""
        # Create the truss beam with parameters (convert from mm to m)
        beam = RectangularTrussBeam(
            length=params.step_1.truss_length / 1000,
            width=params.step_1.truss_width / 1000,
            height=params.step_1.truss_height / 1000,
            n_diagonals=int(params.step_1.n_divisions),
        )
        
        # Build and clean the model
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Create 3D geometry
        sections_group = []
        
        # Parse cross-section size (e.g., "SHS50x4" -> 0.05 meters)
        cs_size = float(params.step_1.cross_section.replace("SHS", "").split("x")[0]) / 1000
        
        for line_id, line_data in lines.items():
            node_i = nodes[line_data["NodeI"]]
            node_j = nodes[line_data["NodeJ"]]
            
            # Map coordinates: x -> x, y (height) -> y (VIKTOR vertical), z -> z
            point_i = Point(node_i["x"], node_i["y"], node_i["z"])
            point_j = Point(node_j["x"], node_j["y"], node_j["z"])
            
            line_k = Line(point_i, point_j)
            section_k = RectangularExtrusion(cs_size, cs_size, line_k, identifier=str(line_id))
            sections_group.append(section_k)
        
        return vkt.GeometryResult(geometry=sections_group)
    
    def run_opensees_model(self, params, **kwargs):
        """Run the OpenSees analysis model."""
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
        
        print(f"\n{'='*60}")
        print("OpenSees Analysis Results - All Load Combinations")
        print(f"Cross-Section: {selected_cs_name}")
        print(f"Load Cases Defined: {[lc['name'] for lc in load_cases]}")
        print(f"{'='*60}")
        print("\nAll Combinations Results:")
        print(f"{'-'*60}")
        for result in all_results:
            max_abs = result['max_abs_displacement']
            print(f"  {result['combination_name']:<25}: Max |ΔZ| = {max_abs:.4f} mm")
        print(f"{'-'*60}")
        print(f"\n*** CRITICAL COMBINATION: {critical_result['combination_name']} ***")
        print(f"    Max Absolute Displacement: {critical_result['max_abs_displacement']:.4f} mm")
        print("\nMax Displacement by Element Type (Critical Combination):")
        for elem_type, disp in critical_result['max_disp_by_type'].items():
            print(f"  {elem_type}: {disp:.4f} mm")
        print(f"{'='*60}\n")
        
        return None

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
