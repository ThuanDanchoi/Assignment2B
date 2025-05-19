"""
Graph data structure 
"""

class Graph:
    def __init__(self, nodes, edges, origin, destinations):
        """
        Initialize a graph for the route finding problem
        
        Args:
            nodes (dict): Dictionary mapping node IDs to (x,y) coordinates
            edges (dict): Dictionary mapping (from_node, to_node) to edge cost
            origin (int): Starting node ID
            destinations (list): List of destination node IDs
        """
        self.nodes = nodes
        self.edges = edges
        self.origin = origin
        self.destinations = destinations

    def get_neighbors(self, node_id):
        """
        Get all neighboring nodes that can be reached from the given node
        Returns list of (neighbor_id, cost) tuples, sorted by node ID
        """
        neighbors = []
        for (from_node, to_node), cost in self.edges.items(): 
            if from_node == node_id:
                neighbors.append((to_node, cost))

        # Sort neighbors by node ID (tie-breaking rule)
        return sorted(neighbors, key=lambda x: x[0])

    def is_destination(self, node_id):
        """Check if a node is a destination node"""
        return node_id in self.destinations

    def get_coordinates(self, node_id):
        """Get the coordinates of a node"""
        return self.nodes.get(node_id, (0, 0))  # fallback if node_id not found

    def heuristic(self, node_id, destination_id=None):
        """
        Calculate heuristic value (straight-line distance) from node to destination
        If no specific destination given, finds minimum distance to any destination
        """
        x1, y1 = self.get_coordinates(node_id)

        if destination_id is not None:
            x2, y2 = self.get_coordinates(destination_id)
            return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5  # Euclidean distance
        else:
            if not self.destinations:
                print(f"[ERROR] heuristic() called with no destinations! node_id={node_id}")
                return float('inf')  # hoặc return 0 nếu bạn muốn an toàn
            return min(
                ((self.get_coordinates(dest)[0] - x1) ** 2 + (self.get_coordinates(dest)[1] - y1) ** 2) ** 0.5
                for dest in self.destinations
            )
