import json
from pathlib import Path

import viktor as vkt

from viktor.geometry import Point, Line, RectangularExtrusion
from app.truss_beam import RectangularTrussBeam
from app.opensees.model import Model, calculate_displacements
from app.types import CrossSectionInfo, NodesInfoDict, LinesInfoDict, MembersDict, CrossSectionsDict
from app.plots.deformation import plot_deformed_mesh
from app.plots.loads import plot_loads_3d


class Parametrization(vkt.Parametrization):
    # Step 1: Geometry
    step_1 = vkt.Step("Step 1 - Geometry", views=["create_render"])
    step_1.geometry = vkt.Section("Geometry Definition")
    step_1.geometry.intro = vkt.Text("""# Rectangular Truss Beam - OpenSees Analysis

Define the truss beam geometry and cross-section parameters below.""")
    
    step_1.geometry.truss_length = vkt.NumberField("Truss Length", min=100, default=10000, suffix="mm")
    step_1.geometry.truss_width = vkt.NumberField("Truss Width", min=100, default=1000, suffix="mm")
    step_1.geometry.truss_height = vkt.NumberField("Truss Height", min=100, default=1500, suffix="mm")
    step_1.geometry.n_divisions = vkt.NumberField("Number of Divisions", min=1, default=6)
    
    step_1.geometry.line_break = vkt.LineBreak()
    
    step_1.geometry.section_title = vkt.Text("""## Cross-Section
Select a cross section size for the truss members:""")
    step_1.geometry.cross_section = vkt.OptionField(
        "Cross-Section Size", 
        options=["SHS50x4", "SHS75x4", "SHS100x4", "SHS150x4"], 
        default="SHS50x4"
    )
    
    step_1.geometry.line_break_2 = vkt.LineBreak()
    
    # Step 2: Run Model
    step_2 = vkt.Step("Step 2 - Run Model", views=["create_render", "show_deformation"])
    step_2.analysis = vkt.Section("OpenSees Analysis")
    step_2.analysis.intro = vkt.Text("""# Run OpenSees Analysis

Run the structural analysis using OpenSees. The model will be generated based on the geometry defined in Step 1.""")
    
    step_2.analysis.run_btn = vkt.ActionButton("Run OpenSees Model", method="run_opensees_model")
    
    step_2.results = vkt.Section("Deformation Settings")
    step_2.results.deform_scale = vkt.NumberField("Deformation Scale", min=1, max=500, default=25, step=1)

    # Step 3: Gravitational Loads
    step_3 = vkt.Step("Step 3 - Gravitational Loads", views=["create_render", "show_loads"])
    step_3.loads = vkt.Section("Gravitational Loads")
    step_3.loads.intro = vkt.Text("""# Gravitational Loads

Define the gravitational load to apply to the truss beam. The load will be visualized as point loads at all nodes in the XY plane (z=0).""")
    
    step_3.loads.load_q = vkt.NumberField("Load Q", min=0, default=5, suffix="kPa")


class Controller(vkt.Controller):
    parametrization = Parametrization
    
    @vkt.GeometryView("3D Model", x_axis_to_right=True)
    def create_render(self, params, **kwargs):
        """Create 3D visualization of the truss beam."""
        # Create the truss beam with parameters (convert from mm to m)
        beam = RectangularTrussBeam(
            length=params.step_1.geometry.truss_length / 1000,
            width=params.step_1.geometry.truss_width / 1000,
            height=params.step_1.geometry.truss_height / 1000,
            n_diagonals=int(params.step_1.geometry.n_divisions),
        )
        
        # Build and clean the model
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Create 3D geometry
        sections_group = []
        
        # Parse cross-section size (e.g., "SHS50x4" -> 0.05 meters)
        cs_size = float(params.step_1.geometry.cross_section.replace("SHS", "").split("x")[0]) / 1000
        
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
        selected_cs_name = params.step_1.geometry.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for OpenSees)
        beam = RectangularTrussBeam(
            length=params.step_1.geometry.truss_length,
            width=params.step_1.geometry.truss_width,
            height=params.step_1.geometry.truss_height,
            n_diagonals=int(params.step_1.geometry.n_divisions),
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
        length = params.step_1.geometry.truss_length
        support_nodes_start = get_nodes_by_x_and_z(nodes_dict, x=0, z=0)
        support_nodes_end = get_nodes_by_x_and_z(nodes_dict, x=length, z=0)
        support_nodes = support_nodes_start + support_nodes_end
        
        # Create and run the OpenSees model
        model = Model(
            nodes=nodes_dict,
            lines=lines_dict,
            cross_sections=cross_sections,
            members=members,
            support_nodes=support_nodes,
        )
        
        model.create_model()
        model.run_model()
        
        # Calculate and print displacements
        max_disp_by_type, disp_dict = calculate_displacements(lines_dict, nodes_dict)
        
        print(f"\n{'='*50}")
        print("OpenSees Analysis Results")
        print(f"Cross-Section: {selected_cs_name}")
        print(f"{'='*50}")
        print("\nMax Displacement by Element Type:")
        for elem_type, disp in max_disp_by_type.items():
            print(f"  {elem_type}: {disp:.4f} mm")
        print(f"\nMin Displacement (all nodes): {min(disp_dict.values()):.4f} mm")
        print(f"Max Displacement (all nodes): {max(disp_dict.values()):.4f} mm")
        print(f"{'='*50}\n")
        
        return None

    @vkt.PlotlyView("Deformed Shape", duration_guess=5)
    def show_deformation(self, params, **kwargs):
        """Display the deformed mesh using Plotly."""
        # Load cross-section library
        cs_library_path = Path(__file__).parent / "cs_library.json"
        with open(cs_library_path, "r") as f:
            cs_library: list[CrossSectionInfo] = json.load(f)
        
        # Get selected cross-section
        selected_cs_name = params.step_1.geometry.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for OpenSees)
        beam = RectangularTrussBeam(
            length=params.step_1.geometry.truss_length,
            width=params.step_1.geometry.truss_width,
            height=params.step_1.geometry.truss_height,
            n_diagonals=int(params.step_1.geometry.n_divisions),
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
        length = params.step_1.geometry.truss_length
        support_nodes_start = get_nodes_by_x_and_z(nodes_dict, x=0, z=0)
        support_nodes_end = get_nodes_by_x_and_z(nodes_dict, x=length, z=0)
        support_nodes = support_nodes_start + support_nodes_end
        
        # Create and run the OpenSees model
        model = Model(
            nodes=nodes_dict,
            lines=lines_dict,
            cross_sections=cross_sections,
            members=members,
            support_nodes=support_nodes,
        )
        
        model.create_model()
        model.run_model()
        
        # Calculate displacements
        max_disp_by_type, disp_dict = calculate_displacements(lines_dict, nodes_dict)
        
        # Get deformation scale from params
        deform_scale = params.step_2.results.deform_scale or 25
        
        # Create the deformed mesh plot
        fig = plot_deformed_mesh(
            nodes=nodes_dict,
            lines=lines_dict,
            members=members,
            cross_sections=cross_sections,
            disp_dict=disp_dict,
            scale=deform_scale,
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
        selected_cs_name = params.step_1.geometry.cross_section
        selected_cs = next((cs for cs in cs_library if cs["name"] == selected_cs_name), None)
        
        if selected_cs is None:
            raise ValueError(f"Cross-section {selected_cs_name} not found in library")
        
        # Build the truss beam geometry (in mm for visualization)
        beam = RectangularTrussBeam(
            length=params.step_1.geometry.truss_length,
            width=params.step_1.geometry.truss_width,
            height=params.step_1.geometry.truss_height,
            n_diagonals=int(params.step_1.geometry.n_divisions),
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
        load_q = params.step_3.loads.load_q or 0.0
        
        # Create the loads visualization plot
        fig = plot_loads_3d(
            nodes=nodes_dict,
            lines=lines_dict,
            members=members,
            cross_sections=cross_sections,
            load=load_q,
        )
        
        return vkt.PlotlyResult(fig)
