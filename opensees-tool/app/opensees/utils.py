from app.types import Vec3, NodesDict
import math


def v_sub(a: Vec3, b: Vec3) -> Vec3:
    """Subtract vector b from vector a."""
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_cross(a: Vec3, b: Vec3) -> Vec3:
    """Calculate the cross product of vectors a and b."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(v: Vec3) -> float:
    """Calculate the Euclidean norm (magnitude) of a vector."""
    return math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)

def get_nodes_by_z(nodes: NodesDict, z: float) -> list[int]:
    """
    Get node IDs where z-coordinate equals the specified value.
    """
    return [node_data["id"] for node_data in nodes.values() if node_data["z"] == z]