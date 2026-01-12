import viktor as vkt

from viktor.geometry import Point, Line, RectangularExtrusion
from app.truss_beam import RectangularTrussBeam


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
    step_2 = vkt.Step("Step 2 - Run Model", views=["create_render"])
    step_2.analysis = vkt.Section("OpenSees Analysis")
    step_2.analysis.intro = vkt.Text("""# Run OpenSees Analysis

Run the structural analysis using OpenSees. The model will be generated based on the geometry defined in Step 1.""")
    
    step_2.analysis.run_btn = vkt.ActionButton("Run OpenSees Model", method="run_opensees_model")


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
        """Run the OpenSees analysis model.
        
        This method will be implemented to run the structural analysis.
        """
        # TODO: Implement OpenSees model execution logic
        pass
