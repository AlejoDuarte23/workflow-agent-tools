import json
from math import pi
from typing import Optional, Tuple

import viktor as vkt

from viktor.geometry import Point, Line, Polygon, Material, Group, RectangularExtrusion, SquareBeam
from app.truss_beam import RectangularTrussBeam


def notched_profile(width=2000.0, height=2400.0, notch=400.0) -> list[vkt.Point]:
    """
    Profile is defined in the local XY plane (z=0).
    This outline is a width x height rectangle with a notch (notch x notch)
    removed at the top-right corner.

    Clockwise + closed (first point repeated at the end) as required by Extrusion.
    """
    w = float(width)
    h = float(height)
    n = float(notch)

    # Center the profile around the local origin (0, 0) so the extrusion line passes through the center
    xL, xR = -w / 2, w / 2
    yB, yT = -h / 2, h / 2

    xN = xR - n      # notch inner x
    yN = yT - n      # notch bottom y

    # Clockwise loop
    return [
        vkt.Point(xL, yB),
        vkt.Point(xL, yT),
        vkt.Point(xN, yT),
        vkt.Point(xN, yN),
        vkt.Point(xR, yN),
        vkt.Point(xR, yB),
        vkt.Point(xL, yB),
    ]



class Parametrization(vkt.Parametrization):
    intro = vkt.Text("# Rectangular Truss Beam Generator")
    
    inputs_title = vkt.Text('''## Truss Geometry  
Please fill in the following parameters to create the truss beam:''')
    
    truss_length = vkt.NumberField("Truss Length", min=100, default=20000, suffix="mm")
    truss_width = vkt.NumberField("Truss Width", min=100, default=3000, suffix="mm")
    truss_height = vkt.NumberField("Truss Height", min=100, default=4000, suffix="mm")
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
        nodes, lines, chord_tl_ids, chord_tr_ids = beam.build()
        nodes, lines = beam.clean_model()
        nodes, lines = beam.remove_top_edge_nodes(chord_tl_ids, chord_tr_ids)
        
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
        beam = RectangularTrussBeam(
            length=params.truss_length / 1000,
            width=params.truss_width / 1000,
            height=params.truss_height / 1000,
            n_diagonals=int(params.n_divisions),
        )

        nodes, lines, chord_tl_ids, chord_tr_ids = beam.build()
        nodes, lines = beam.clean_model()
        nodes, lines = beam.remove_top_edge_nodes(chord_tl_ids, chord_tr_ids)

        # Cross-section (meters)
        cs_size = float(params.cross_section.replace("SHS", "").split("x")[0]) / 1000

        # --- Detect which node axis is width / height in the beam output ---
        target_w = params.truss_width / 1000
        target_h = params.truss_height / 1000

        ys = [n["y"] for n in nodes.values()]
        zs = [n["z"] for n in nodes.values()]
        span_y = max(ys) - min(ys)
        span_z = max(zs) - min(zs)

        # Option A: y=width, z=height
        cost_a = abs(span_y - target_w) + abs(span_z - target_h)
        # Option B: y=height, z=width
        cost_b = abs(span_y - target_h) + abs(span_z - target_w)

        y_is_width = cost_a <= cost_b

        def to_vkt_point(n):
            # Ensure: VIKTOR uses x=length, y=width, z=height for both truss + embankment placement
            if y_is_width:
                return Point(n["x"], n["y"], n["z"])  # (x, width, height)
            return Point(n["x"], n["z"], n["y"])      # swap (y<->z) => (x, width, height)

        # --- Build truss geometry ---
        sections_group = []
        for line_id, line_data in lines.items():
            ni = nodes[line_data["NodeI"]]
            nj = nodes[line_data["NodeJ"]]
            pi = to_vkt_point(ni)
            pj = to_vkt_point(nj)
            sections_group.append(
                RectangularExtrusion(cs_size, cs_size, Line(pi, pj), identifier=str(line_id))
            )


        sections_group = []
        truss_material = Material(color=vkt.Color.from_hex("#C41E3A"), roughness=0.8, metalness=0.3)
        for line_id, line_data in lines.items():
            ni = nodes[line_data["NodeI"]]
            nj = nodes[line_data["NodeJ"]]
            pi = to_vkt_point(ni)
            pj = to_vkt_point(nj)
            sections_group.append(
                RectangularExtrusion(cs_size, cs_size, Line(pi, pj), identifier=str(line_id), material=truss_material)
            )

        height =  params.truss_height/1000
        concrete_material = Material(color=vkt.Color(80, 80, 80), roughness=0.9, metalness=0.1)
        node2 = vkt.Point(-0.4, -params.truss_width/1000, -height/2 + cs_size)
        node1 = vkt.Point(-0.400, 2*params.truss_width/1000, -height/2 + cs_size)
        center_line = vkt.Line(node1, node2)
        profile = notched_profile(width=1.000,height=height , notch=cs_size)
        solid = vkt.Extrusion(profile, center_line, profile_rotation=0, material=concrete_material)

        sections_group.append(vkt.Group([solid, center_line]))
        
        node1 = vkt.Point(params.truss_length/1000+ 0.4, -params.truss_width/1000, -height/2 + cs_size) 
        node2 = vkt.Point(params.truss_length/1000+ 0.400, 2*params.truss_width/1000, -height/2 + cs_size)
        center_line = vkt.Line(node1, node2)

        profile = notched_profile(width=1.000, height=height, notch=cs_size)
        solid = vkt.Extrusion(profile, center_line, profile_rotation=180, material=concrete_material)

        sections_group.append(vkt.Group([solid, center_line]))

        # --- Add bridge deck ---
        deck_thickness = 0.2  # meters
        deck_length = params.truss_length / 1000
        deck_width = params.truss_width / 1000
        
        deck_material = Material(color=vkt.Color(100, 100, 100), roughness=0.9, metalness=0.1)
        deck = SquareBeam(deck_length, deck_width, deck_thickness, material=deck_material)
        deck.translate((deck_length / 2, deck_width / 2, deck_thickness / 2))  # Position at z=0
        sections_group.append(deck)
        
        # --- Add asphalt layer on top ---
        asphalt_thickness = 0.05  # meters
        asphalt_material = Material(color=vkt.Color(40, 40, 40), roughness=1.0, metalness=0.0)
        asphalt = SquareBeam(deck_length, deck_width, asphalt_thickness, material=asphalt_material)
        asphalt.translate((deck_length / 2, deck_width / 2, deck_thickness / 2 + asphalt_thickness / 2))
        sections_group.append(asphalt)

        return vkt.GeometryResult(sections_group)

    @vkt.DataView("Truss Information")
    def visualize_data(self, params, **kwargs):
        """Display truss geometry information and statistics."""
        # Convert from mm to m for beam calculation
        beam = RectangularTrussBeam(
            length=params.truss_length / 1000,
            width=params.truss_width / 1000,
            height=params.truss_height / 1000,
            n_diagonals=int(params.n_divisions),
        )
        
        # Build and clean the model
        nodes, lines, chord_tl_ids, chord_tr_ids = beam.build()
        nodes, lines = beam.clean_model()
        nodes, lines = beam.remove_top_edge_nodes(chord_tl_ids, chord_tr_ids)
        
        # Parse cross-section size
        cs_size_mm = float(params.cross_section.replace("SHS", "").split("x")[0])
        
        # Calculate total member length
        total_length = 0
        for line_id, line_data in lines.items():
            node_i = nodes[line_data["NodeI"]]
            node_j = nodes[line_data["NodeJ"]]
            length = ((node_j["x"] - node_i["x"])**2 + 
                     (node_j["y"] - node_i["y"])**2 + 
                     (node_j["z"] - node_i["z"])**2)**0.5
            total_length += length
        
        # Create data structure
        data = vkt.DataGroup(
            geometry_params=vkt.DataItem('Geometry Parameters', '', subgroup=vkt.DataGroup(
                length=vkt.DataItem('Length', params.truss_length, suffix='mm'),
                width=vkt.DataItem('Width', params.truss_width, suffix='mm'),
                height=vkt.DataItem('Height', params.truss_height, suffix='mm'),
                divisions=vkt.DataItem('Number of Divisions', int(params.n_divisions))
            )),
            model_stats=vkt.DataItem('Model Statistics', '', subgroup=vkt.DataGroup(
                nodes=vkt.DataItem('Total Nodes', len(nodes)),
                members=vkt.DataItem('Total Members', len(lines)),
                total_length=vkt.DataItem('Total Member Length', round(total_length * 1000, 2), suffix='mm')
            )),
            cross_section=vkt.DataItem('Cross-Section', '', subgroup=vkt.DataGroup(
                type=vkt.DataItem('Type', params.cross_section),
                size=vkt.DataItem('Size', cs_size_mm, suffix='mm')
            ))
        )
        
        return vkt.DataResult(data)