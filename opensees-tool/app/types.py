from typing import TypedDict, Annotated, Literal, Union


class NodeDict(TypedDict):
    """Node coordinates in 3D space."""
    x: float
    y: float
    z: float


class LineDict(TypedDict):
    """Line connectivity between two nodes."""
    NodeI: int
    NodeJ: int


class NodeInfo(TypedDict):
    id: int
    x: float
    y: float
    z: float


class LineInfo(TypedDict):
    id: int
    Ni: int
    Nj: int
    Type: str


MemberType = Literal["Truss Diagonal", "Joist", "Beam", "Column", "Truss Chord"]

class CrossSectionInfo(TypedDict):
    name: str
    id: int
    A: Annotated[float, "Area"]
    Iz: Annotated[float, "Inertia around z, Strong Axis"]
    Iy: Annotated[float, "Inertia around y, Weak Axis"]
    Jxx: Annotated[float, "Torsional Inertia"]
    b: Annotated[float, "Section width"]
    h: Annotated[float, "Section height"]


class MemberInfo(TypedDict):
    line_id: int
    cross_section_id: int
    material_name: Literal["Concrete", "Steel"]


class NodeMass(TypedDict):
    mass_x: float
    mass_y: float
    mass_z: float


class NodalLoad(TypedDict):
    """Nodal load with forces and moments."""
    node_id: int
    fx: Annotated[float, "Force in x direction (N)"]
    fy: Annotated[float, "Force in y direction (N)"]
    fz: Annotated[float, "Force in z direction (N)"]
    mx: Annotated[float, "Moment about x axis (N-mm)"]
    my: Annotated[float, "Moment about y axis (N-mm)"]
    mz: Annotated[float, "Moment about z axis (N-mm)"]


class LoadCase(TypedDict):
    """Load case with name, factor, and list of nodal loads."""
    name: str
    factor: Annotated[float, "Load factor for combinations"]
    loads: list[NodalLoad]


# Aliases
NodesDict = dict[int, NodeDict]
LinesDict = dict[int, LineDict]
NodesInfoDict = dict[int, NodeInfo]
LinesInfoDict = dict[int, LineInfo]
CrossSectionsDict = dict[int, CrossSectionInfo]
MembersDict = dict[int, MemberInfo]
MassDict = dict[int, NodeMass]

Vec3 = tuple[float, float, float]


# Materials
class Steel:
    name: Literal["Steel"] = "Steel"
    G: float = 0.25 * 10**3
    gamma: float = 7.85 * 10**-5
    E: float = 200 * 10**3
    units: Literal["N,mm"] = "N,mm"


class Concrete:
    name: Literal["Concrete"] = "Concrete"
    gamma: float = 2.5 * 10**-5
    E: float = 26_700.0
    units: Literal["N,mm"] = "N,mm"
    G: float     = E / (2 * 1.2)


MaterialName = Literal["Steel", "Concrete"]
MaterialType = Union[Steel, Concrete]
MaterialDictType = dict[MaterialName, MaterialType]

material_dict: MaterialDictType = {
    "Steel": Steel(),
    "Concrete": Concrete(),
}

steel_cost: float = 3.81 # Euros