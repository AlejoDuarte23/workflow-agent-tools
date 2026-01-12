
import openseespy.opensees as ops
from app.types import (
    Vec3,
    NodesDict,
    LinesDict,
    MembersDict,
    MaterialDictType,
    MassDict,
    CrossSectionsDict,
    material_dict,
    LoadCase,
    LoadCombination,
    CombinationResult,
    SLS_COMBINATIONS,
)
from app.opensees.utils import v_cross, v_sub, v_norm, find_critical_combination
from collections import defaultdict
from typing import DefaultDict, Annotated




class Model:
    def __init__(
        self,
        nodes: NodesDict,
        lines: LinesDict,
        cross_sections: CrossSectionsDict,
        members: MembersDict,
        support_nodes: Annotated[list[int] | None, "Node IDs to fix as supports"] = None,
        load_cases: Annotated[list[LoadCase] | None, "List of load cases to apply"] = None,
    ) -> None:
        self.nodes = nodes
        self.lines = lines
        self.cross_sections = cross_sections
        self.members = members
        self.support_nodes = support_nodes if support_nodes is not None else []
        self.load_cases = load_cases if load_cases is not None else []
        self.materials: MaterialDictType = material_dict
        self.mass: MassDict = {}
        self.g = 10000  # 9800
        self.loadsDict: DefaultDict[int, dict[str, float]] = defaultdict(
            lambda: {"fx": 0.0, "fy": 0.0, "fz": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0}
        )

    def create_nodes(self) -> None:
        for n in self.nodes.values():
            ops.node(n["id"], n["x"], n["y"], n["z"])

    def _define_elastic_section(
        self,
        section_id: int,
        rotation: float,
        E: float,
        G_mod: float,
        N: int,
    ) -> None:
        """
        Define an elastic section and a Lobatto integration rule.
        Axis swap when rotation ≠ 0.
        """
        cs = self.cross_sections[section_id]
        if rotation != 0.0:
            Iz, Iy = cs["Iy"], cs["Iz"]
        else:
            Iz, Iy = cs["Iz"], cs["Iy"]
        ops.section(
            "Elastic",
            section_id,
            E,
            cs["A"],
            Iz,
            Iy,
            G_mod,
            cs["Jxx"],
        )
        ops.beamIntegration("Lobatto", section_id, section_id, N)


    def get_support_nodes(self, nodes_dict: NodesDict) -> list[int]:
        """
        Get node IDs where z-coordinate equals 0.
        """
        support_nodes = []
        for node_id, node_data in nodes_dict.items():
            if node_data["z"] == 0:
                support_nodes.append(node_data["id"])
        return support_nodes

    def assign_support(self, support_nodes: list[int] | None = None) -> None:
        """Assign fixed supports to specified nodes.
        
        Args:
            support_nodes: List of node IDs to fix. If None, uses self.support_nodes.
        """
        nodes_to_fix = support_nodes if support_nodes is not None else self.support_nodes
        for node_tag in nodes_to_fix:
            ops.fix(node_tag, 1, 1, 1, 1, 1, 1)

    def create_beam_elements(
        self,
        z_global: Vec3 = (0,0,1),
        N: int = 10,
        verbose: bool = False,
    ) -> None:
        """
        Create geometric transformations, sections, forceBeamColumn
        elements, and update the lumped nodal mass dictionary.
        """

        section_set: set[int] = set()
        for member in self.members.values():
            line_id = member["line_id"]
            section_id = member["cross_section_id"]
            material_name = member["material_name"]
            
            # Geometry
            line = self.lines[line_id]
            node_i = self.nodes[line["Ni"]]
            node_j = self.nodes[line["Nj"]]


            xi: Vec3 = (node_i["x"], node_i["y"], node_i["z"])
            xj: Vec3 = (node_j["x"], node_j["y"], node_j["z"])
            x_axis: Vec3 = v_sub(xi, xj)
            vec_xz: Vec3 = v_cross(x_axis, z_global)

            # Material
            material = self.materials[material_name]
            E,G,gamma = material.E, material.G, material.gamma
                # Elastic member -> rotation = 0
            if section_id not in section_set:
                self._define_elastic_section(section_id, 0, E, G, N)
                section_set.add(section_id)

            # Since I'm not using numpy this make opensees happy.
            vc2 = (1e-29, -1.0, 0)

            # Geometric transformation with original conditions
            if v_norm(vec_xz) == 0.0:
                ops.geomTransf("Linear", line_id, *vc2)
            else:
                # non-zero cross-product, apply vec_xz
                # the nested check for purely horizontal Z stays the same
                if node_i["z"] - node_j["z"] == 0.0:

                    ops.geomTransf("Linear", line_id, *vec_xz)
                else:
                    # We can implement the logic for truss element (diagonal elements)
                    ops.geomTransf("Linear", line_id, *vec_xz)
                    
                    # Element
            ops.element(
                "forceBeamColumn",
                line_id,
                node_i["id"],
                node_j["id"],
                line_id, # geom tranformation
                section_id,
            )

            # Lumped mass
            L = v_norm(x_axis)
            area = self.cross_sections[section_id]["A"]
            m_node = area * L * gamma / (2.0 * self.g)
            for tag in (node_i["id"], node_j["id"]):
                nm = self.mass.setdefault(
                    tag, {"mass_x": 0.0, "mass_y": 0.0, "mass_z": 0.0}
                )
                nm["mass_x"] += m_node
                nm["mass_y"] += m_node
                nm["mass_z"] += m_node

            if verbose:
                print(
                    f"Line {line_id}: section {section_id}, "
                    f"nodes {node_i['id']}–{node_j['id']}",
                    f"Cross Section name: {self.cross_sections[section_id]["name"]}"
                )
    
    def create_loads(self, q_factor: float = 1.0, wl_factor: float = 1.0):
        """Create self weight load and apply load cases with specified factors.
        
        Args:
            q_factor: Factor for gravitational loads (Q). Default 1.0.
            wl_factor: Factor for wind loads (WL). Positive = +WL, Negative = -WL.
        """
        # Reset loads dict for new combination
        self.loadsDict = defaultdict(
            lambda: {"fx": 0.0, "fy": 0.0, "fz": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0}
        )
        
        # Add self-weight loads (SLS - always applied)
        for nodetag, mass_values in self.mass.items():
            # Mass in opensees is N/g * g to loads (negative z = downward)
            self.loadsDict[nodetag]["fz"] -= mass_values["mass_z"] * self.g

        # Apply load cases with appropriate factors based on their name
        for load_case in self.load_cases:
            case_name = load_case.get("name", "").lower()
            
            # Determine the factor to apply based on load case type
            if "dead" in case_name or "gravity" in case_name or case_name == "q":
                factor = q_factor
            elif "wind" in case_name or case_name == "wl":
                factor = wl_factor
            else:
                # For other load cases, use their original factor
                factor = load_case.get("factor", 1.0)
            
            for load in load_case["loads"]:
                node_id = load["node_id"]
                self.loadsDict[node_id]["fx"] += load["fx"] * factor
                self.loadsDict[node_id]["fy"] += load["fy"] * factor
                self.loadsDict[node_id]["fz"] += load["fz"] * factor
                self.loadsDict[node_id]["mx"] += load["mx"] * factor
                self.loadsDict[node_id]["my"] += load["my"] * factor
                self.loadsDict[node_id]["mz"] += load["mz"] * factor

        # Apply loads to OpenSees model
        for nodetag, load_components in self.loadsDict.items():
            ops.load(
                nodetag,
                load_components["fx"],
                load_components["fy"],
                load_components["fz"],
                load_components["mx"],
                load_components["my"],
                load_components["mz"],
            )


    def create_model(self):
        ops.wipe()
        # Creates Opensees Model
        ops.model('basic','-ndm',3,'-ndf',6)
        # Create nodes
        self.create_nodes()
        # ops.getNodeTags()
        # Create beam elements
        self.create_beam_elements(verbose=False)
        # Create Support Nodes
        self.assign_support()
        # vfo.plot_model()

    def run_model(self, q_factor: float = 1.0, wl_factor: float = 1.0):
        """Run the OpenSees analysis with specified load factors.
        
        Args:
            q_factor: Factor for gravitational loads (Q). Default 1.0.
            wl_factor: Factor for wind loads (WL). Positive = +WL, Negative = -WL.
        """
        ops.timeSeries("Linear", 1)
        ops.pattern("Plain", 1, 1)
        self.create_loads(q_factor=q_factor, wl_factor=wl_factor)

        ops.system('BandGen')
        ops.constraints('Plain')
        ops.numberer('Plain')
        ops.algorithm('Linear')
        ops.integrator('LoadControl',1)
        ops.analysis('Static')
        ops.analyze(1)
        ops.reactions()
        return ops

    def run_combination(self, combination: LoadCombination) -> CombinationResult:
        """Run a single load combination and return results.
        
        Args:
            combination: LoadCombination with name, Q_factor, and WL_factor.
            
        Returns:
            CombinationResult with displacements and max absolute displacement.
        """
        # Recreate the model (wipe and rebuild)
        self.create_model()
        
        # Run with the combination factors
        self.run_model(
            q_factor=combination["Q_factor"],
            wl_factor=combination["WL_factor"]
        )
        
        # Calculate displacements
        max_disp_by_type, disp_dict = calculate_displacements(self.lines, self.nodes)
        
        # Calculate max absolute displacement (using dz component)
        max_abs_disp = max(abs(d["dz"]) for d in disp_dict.values()) if disp_dict else 0.0
        
        return {
            "combination_name": combination["name"],
            "max_disp_by_type": max_disp_by_type,
            "disp_dict": disp_dict,
            "max_abs_displacement": max_abs_disp,
        }

    def run_all_combinations(
        self, 
        combinations: list[LoadCombination] | None = None
    ) -> tuple[CombinationResult, list[CombinationResult]]:
        """Run all SLS load combinations and find the critical one.
        
        Args:
            combinations: List of LoadCombination to run. If None, uses SLS_COMBINATIONS.
            
        Returns:
            Tuple of (critical_result, all_results) where critical_result is the 
            combination with the highest absolute displacement.
        """
        if combinations is None:
            combinations = SLS_COMBINATIONS
        
        all_results: list[CombinationResult] = []
        
        for combo in combinations:
            result = self.run_combination(combo)
            all_results.append(result)
        
        critical_result = find_critical_combination(all_results)
        
        return critical_result, all_results


    def __repr__(self) -> str:
        unique_cc = {cs["name"] for cs in self.cross_sections.values()}
        format_names = ", ".join(unique_cc)
        return (
            f"<Model(NoNodes={len(self.nodes)}, "
            f"NoLines={len(self.lines)}, "
            f"NoMembers={len(self.members)}, "
            f"CrossSections={format_names})>"
        )
    

def calculate_displacements(lines:LinesDict, nodes:NodesDict):
    """Calculate displacements for all nodes in x, y, z directions.
    
    Returns:
        max_disp_by_type: Dict mapping element type to max z-displacement.
        disp_dict: Dict mapping node ID to displacement dict with dx, dy, dz.
    """
    from app.types import DispDict
    
    disp_by_type: DefaultDict[str, list[float]] = defaultdict(list)
    disp_dict: DispDict = {}
    
    for lineargs in lines.values():
        for node in (lineargs["Ni"], lineargs["Nj"]):
            disp = ops.nodeDisp(node)
            disp_z = disp[2]
            disp_by_type[lineargs["Type"]].append(disp_z)

            if node not in disp_dict:
                disp_dict[node] = {
                    "dx": disp[0],
                    "dy": disp[1],
                    "dz": disp[2],
                }
                        
    max_disp_by_type = {eletype: min(disp_list) for eletype, disp_list in disp_by_type.items()}   

    for node in nodes:
        disp = ops.nodeDisp(node)
        disp_dict[node] = {
            "dx": disp[0],
            "dy": disp[1],
            "dz": disp[2],
        }

    return max_disp_by_type, disp_dict

