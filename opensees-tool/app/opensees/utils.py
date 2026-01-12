from app.types import Vec3, NodesDict, LoadCombination, CombinationResult
import math
from typing import Callable


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

def get_nodes_by_z(nodes: NodesDict, z: float, tolerance: float = 1e-6) -> list[int]:
    """Get node IDs where z-coordinate equals the specified value."""
    return [node_data["id"] for node_data in nodes.values() if abs(node_data["z"] - z) < tolerance]

def get_nodes_by_x(nodes: NodesDict, x: float, tolerance: float = 1e-6) -> list[int]:
    """Get node IDs where x-coordinate equals the specified value."""
    return [node_data["id"] for node_data in nodes.values() if abs(node_data["x"] - x) < tolerance]

def get_nodes_by_x_and_z(nodes: NodesDict, x: float, z: float, tolerance: float = 1e-6) -> list[int]:
    """Get node IDs where both x and z coordinates match the specified values."""
    return [
        node_data["id"] 
        for node_data in nodes.values() 
        if abs(node_data["x"] - x) < tolerance and abs(node_data["z"] - z) < tolerance
    ]


def find_critical_combination(results: list[CombinationResult]) -> CombinationResult:
    """Find the load combination with the highest absolute displacement.
    
    Args:
        results: List of CombinationResult from running all combinations.
        
    Returns:
        The CombinationResult with the maximum absolute displacement.
    """
    if not results:
        raise ValueError("No combination results provided")
    
    return max(results, key=lambda r: r["max_abs_displacement"])