import viktor as vkt

from viktor.geometry import Point, Line, RectangularExtrusion
from app.truss_beam import RectangularTrussBeam


class Parametrization(vkt.Parametrization):
    intro = vkt.Text("# Rectangular Truss Beam - OpenSees Analysis")
    
    inputs_title = vkt.Text('''## Truss Geometry  
Please fill in the following parameters to create the truss beam:''')
    
    truss_length = vkt.NumberField("Truss Length", min=100, default=10000, suffix="mm")
    truss_width = vkt.NumberField("Truss Width", min=100, default=1000, suffix="mm")
    truss_height = vkt.NumberField("Truss Height", min=100, default=1500, suffix="mm")
    n_divisions = vkt.NumberField("Number of Divisions", min=1, default=6)
    
    line_break = vkt.LineBreak()
    
    section_title = vkt.Text('''## Cross-Section  
Please select a cross section size for the truss members:''')
    cross_section = vkt.OptionField(
        "Cross-Section Size", 
        options=["SHS50x4", "SHS75x4", "SHS100x4", "SHS150x4"], 
        default="SHS50x4"
    )
    
    line_break_2 = vkt.LineBreak()
    
    export_title = vkt.Text("## Export Geometry")
    download_btn = vkt.DownloadButton("Download Truss Geometry (JSON)", method="download_geometry_json")


class Controller(vkt.Controller):
    parametrization = Parametrization
    
    @vkt.GeometryView("3D Model", x_axis_to_right=True)
    def create_render(self, params, **kwargs):
        """Create 3D visualization of the truss beam."""
        # Create the truss beam with parameters (convert from mm to m)
        beam = RectangularTrussBeam(
            length=params.truss_length / 1000,
            width=params.truss_width / 1000,
            height=params.truss_height / 1000,
            n_diagonals=int(params.n_divisions),
        )
        
        # Build and clean the model
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Create 3D geometry
        sections_group = []
        
        # Parse cross-section size (e.g., "SHS50x4" -> 0.05 meters)
        cs_size = float(params.cross_section.replace("SHS", "").split("x")[0]) / 1000
        
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
