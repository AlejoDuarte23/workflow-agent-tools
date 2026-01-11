import json
import viktor as vkt

from viktor.geometry import Point, Line, RectangularExtrusion
from app.truss_beam import RectangularTrussBeam


class Parametrization(vkt.Parametrization):
    intro = vkt.Text("# Rectangular Truss Beam Generator")
    
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

    def download_geometry_json(self, params, **kwargs):
        """Download truss geometry as JSON file."""
        # Convert from mm to m for beam calculation
        beam = RectangularTrussBeam(
            length=params.truss_length / 1000,
            width=params.truss_width / 1000,
            height=params.truss_height / 1000,
            n_diagonals=int(params.n_divisions),
        )
        
        # Build and clean the model
        nodes, lines = beam.build()
        nodes, lines = beam.clean_model()
        
        # Parse cross-section size (e.g., "SHS50x4" -> 50)
        cs_size_mm = float(params.cross_section.replace("SHS", "").split("x")[0])
        
        # Prepare JSON data structure (convert node coordinates to mm)
        json_data = {
            "parameters": {
                "truss_length_mm": params.truss_length,
                "truss_width_mm": params.truss_width,
                "truss_height_mm": params.truss_height,
                "n_divisions": int(params.n_divisions),
                "cross_section_mm": cs_size_mm
            },
            "nodes": {
                str(node_id): {
                    "x": round(node_data["x"] * 1000, 3),
                    "y": round(node_data["y"] * 1000, 3),
                    "z": round(node_data["z"] * 1000, 3)
                }
                for node_id, node_data in nodes.items()
            },
            "lines": {
                str(line_id): {
                    "NodeI": line_data["NodeI"],
                    "NodeJ": line_data["NodeJ"]
                }
                for line_id, line_data in lines.items()
            },
            "metadata": {
                "total_nodes": len(nodes),
                "total_lines": len(lines),
                "units": {
                    "length": "millimeters",
                    "cross_section": "millimeters"
                }
            }
        }
        
        # Convert to JSON string
        json_string = json.dumps(json_data, indent=2)
        
        # Create file and return download result with descriptive name
        filename = f"truss_{int(params.truss_length)}x{int(params.truss_width)}x{int(params.truss_height)}_{params.cross_section}.json"
        json_file = vkt.File.from_data(json_string)
        return vkt.DownloadResult(json_file, filename)
    
    @vkt.GeometryView("3D Model", x_axis_to_right=True)
    def create_render(self, params, **kwargs):
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
