import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
from matplotlib.animation import FuncAnimation
import math
from matplotlib.patches import Circle

class Kruskal:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
    
    def add_edge(self, u, v, w):
        # Store edges with 1-indexed vertices
        self.graph.append([u, v, w])
    
    def find(self, parent, i):
        if parent[i] != i:
            parent[i] = self.find(parent, parent[i])
        return parent[i]
    
    def union(self, parent, rank, x, y):
        if rank[x] < rank[y]:
            parent[x] = y
        elif rank[x] > rank[y]:
            parent[y] = x
        else:
            parent[y] = x
            rank[x] += 1
    
    def kruskal_mst(self):
        result = []  # This will store the resultant MST
        steps = []   # This will store the intermediate steps for visualization
        
        # Sort all the edges in non-decreasing order of their weight
        self.graph = sorted(self.graph, key=lambda item: item[2])
        
        # Create arrays for tracking parent and rank
        parent = []
        rank = []
        
        # Initialize parent array (1-indexed)
        for node in range(self.V + 1):
            parent.append(node)
            rank.append(0)
        
        # Track indexes
        i = 0  # Index for sorted edges
        e = 0  # Index for result[]
        
        # Store all edges for visualization
        all_edges = self.graph.copy()
        steps.append({
            "mst_edges": [],
            "considered_edge": None,
            "status": "Initial graph"
        })
        
        # Process all edges in increasing order of weight
        while e < self.V - 1 and i < len(self.graph):
            u, v, w = self.graph[i]
            i += 1
            x = self.find(parent, u)
            y = self.find(parent, v)
            
            # Store the current step for visualization
            if x != y:
                steps.append({
                    "mst_edges": result.copy(),
                    "considered_edge": [u, v, w],
                    "status": f"Adding edge {u}-{v} with weight {w}"
                })
            else:
                steps.append({
                    "mst_edges": result.copy(),
                    "considered_edge": [u, v, w],
                    "status": f"Skipping edge {u}-{v} (would create a cycle)"
                })
            
            # If including this edge doesn't cause cycle, include it in result
            if x != y:
                e += 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
        
        # Add final step
        steps.append({
            "mst_edges": result.copy(),
            "considered_edge": None,
            "status": "Final MST"
        })
        
        # Print the contents of the MST
        minimum_cost = 0
        print("\nEdges in the Minimum Spanning Tree:")
        for u, v, weight in result:
            minimum_cost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree Cost:", minimum_cost)
        
        return result, minimum_cost, steps, all_edges

class ReverseDelete:
    def __init__(self, vertices):
        self.V = vertices
        self.graph = []
        self.adj_list = [[] for _ in range(vertices + 1)]  # +1 for 1-indexing
    
    def add_edge(self, u, v, w):
        self.graph.append([u, v, w])
        # Add to adjacency list for connectivity checking
        self.adj_list[u].append((v, w))
        self.adj_list[v].append((u, w))
    
    def is_connected(self, edges_to_skip=None):
        """Check if the graph is connected using BFS."""
        if edges_to_skip is None:
            edges_to_skip = []
        
        # Create a temporary adjacency list excluding skipped edges
        temp_adj = [[] for _ in range(self.V + 1)]
        for u, v, w in self.graph:
            if [u, v, w] not in edges_to_skip:
                temp_adj[u].append(v)
                temp_adj[v].append(u)
        
        # Run BFS to check connectivity
        visited = [False] * (self.V + 1)
        queue = [1]  # Start from vertex 1
        visited[1] = True
        
        while queue:
            vertex = queue.pop(0)
            for neighbor in temp_adj[vertex]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        
        # Check if all vertices (except 0) are visited
        return all(visited[1:])
    
    def reverse_delete_mst(self):
        result = []  # This will store the resultant MST edges
        removed = []  # Store removed edges
        steps = []   # Store steps for visualization
        
        # Sort all edges in non-increasing order of weight
        self.graph = sorted(self.graph, key=lambda item: item[2], reverse=True)
        
        # All edges initially form the graph
        all_edges = self.graph.copy()
        remaining_edges = self.graph.copy()
        
        # Initial step
        steps.append({
            "mst_edges": remaining_edges.copy(),
            "considered_edge": None,
            "removed_edge": None,
            "status": "Initial graph with all edges"
        })
        
        # Process all edges in decreasing order of weight
        for i, edge in enumerate(self.graph):
            u, v, w = edge
            
            # Check if removing this edge disconnects the graph
            remaining_edges.remove(edge)
            
            if not self.is_connected(removed + [edge]):
                # If removing the edge disconnects the graph, add it back
                remaining_edges.append(edge)
                steps.append({
                    "mst_edges": remaining_edges.copy(),
                    "considered_edge": [u, v, w],
                    "removed_edge": None,
                    "status": f"Keeping edge {u}-{v} with weight {w} (removal would disconnect graph)"
                })
            else:
                # Edge can be removed
                removed.append(edge)
                steps.append({
                    "mst_edges": remaining_edges.copy(),
                    "considered_edge": [u, v, w],
                    "removed_edge": [u, v, w],
                    "status": f"Removing edge {u}-{v} with weight {w}"
                })
        
        # Final step
        steps.append({
            "mst_edges": remaining_edges.copy(),
            "considered_edge": None,
            "removed_edge": None,
            "status": "Final MST"
        })
        
        # Print the contents of the MST
        minimum_cost = sum(edge[2] for edge in remaining_edges)
        print("\nEdges in the Minimum Spanning Tree:")
        for u, v, weight in remaining_edges:
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree Cost:", minimum_cost)
        
        return remaining_edges, minimum_cost, steps, all_edges

class ClosestPair:
    def __init__(self):
        self.points = []
        self.steps = []
        self.closest_pair = None
        self.min_distance = float('inf')
    
    def add_point(self, x, y):
        """Add a point to the set."""
        self.points.append((x, y))
    
    def distance(self, p1, p2):
        """Calculate Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def brute_force(self, points):
        """Find closest pair of points using brute force."""
        n = len(points)
        min_dist = float('inf')
        closest_pair = None
        
        for i in range(n):
            for j in range(i+1, n):
                dist = self.distance(points[i], points[j])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (points[i], points[j])
        
        return min_dist, closest_pair
    
    def closest_pair_strip(self, strip, d, closest_pair):
        """Find the closest pair in a strip of width 2d."""
        min_d = d
        n = len(strip)
        result_pair = closest_pair
        
        # Sort by y-coordinate
        strip.sort(key=lambda point: point[1])
        
        # Compare each point with at most 7 points ahead (proven optimal)
        for i in range(n):
            j = i + 1
            while j < n and (strip[j][1] - strip[i][1]) < min_d:
                dist = self.distance(strip[i], strip[j])
                if dist < min_d:
                    min_d = dist
                    result_pair = (strip[i], strip[j])
                j += 1
                
                # Add step for visualization
                self.steps.append({
                    "phase": "strip",
                    "points": self.points.copy(),
                    "current_points": [strip[i], strip[j-1]],
                    "current_dist": dist,
                    "min_dist": min_d,
                    "closest_pair": result_pair,
                    "strip": strip.copy(),
                    "strip_mid": self.mid_x,
                    "strip_width": min_d,
                    "status": f"Checking points in strip: Distance between {strip[i]} and {strip[j-1]} is {dist:.2f}"
                })
        
        return min_d, result_pair
    
    def closest_pair_recursive(self, points_x, points_y):
        """Recursive function to find closest pair of points."""
        n = len(points_x)
        
        # Base case: if there are 2 or 3 points, use brute force
        if n <= 3:
            min_dist, closest_pair = self.brute_force(points_x)
            
            # Add step for visualization
            self.steps.append({
                "phase": "base_case",
                "points": self.points.copy(),
                "current_points": points_x,
                "current_dist": min_dist,
                "min_dist": min_dist,
                "closest_pair": closest_pair,
                "strip": [],
                "strip_mid": None,
                "strip_width": None,
                "status": f"Base case with {n} points: Closest pair is {closest_pair} with distance {min_dist:.2f}"
            })
            
            return min_dist, closest_pair
        
        # Divide the points into two halves
        mid = n // 2
        mid_point = points_x[mid]
        self.mid_x = mid_point[0]  # Save for visualization
        
        # Add step for divide phase
        self.steps.append({
            "phase": "divide",
            "points": self.points.copy(),
            "current_points": points_x,
            "current_dist": None,
            "min_dist": None,
            "closest_pair": None,
            "strip": [],
            "strip_mid": self.mid_x,
            "strip_width": None,
            "status": f"Dividing points at x = {self.mid_x:.2f}"
        })
        
        # Points sorted by x
        left_x = points_x[:mid]
        right_x = points_x[mid:]
        
        # Points sorted by y
        left_y = []
        right_y = []
        for point in points_y:
            if point[0] <= mid_point[0]:
                left_y.append(point)
            else:
                right_y.append(point)
        
        # Recursive calls for left and right halves
        dl, pair_l = self.closest_pair_recursive(left_x, left_y)
        dr, pair_r = self.closest_pair_recursive(right_x, right_y)
        
        # Determine minimum of two distances
        if dl < dr:
            d = dl
            closest_pair = pair_l
        else:
            d = dr
            closest_pair = pair_r
        
        # Add step for conquer phase
        self.steps.append({
            "phase": "conquer",
            "points": self.points.copy(),
            "current_points": [],
            "current_dist": d,
            "min_dist": d,
            "closest_pair": closest_pair,
            "strip": [],
            "strip_mid": self.mid_x,
            "strip_width": d,
            "status": f"Combining results: Closest pair so far is {closest_pair} with distance {d:.2f}"
        })
        
        # Create a strip of points around the middle vertical line
        strip = []
        for point in points_y:
            if abs(point[0] - mid_point[0]) < d:
                strip.append(point)
        
        # Add step for strip creation
        self.steps.append({
            "phase": "strip_creation",
            "points": self.points.copy(),
            "current_points": [],
            "current_dist": d,
            "min_dist": d,
            "closest_pair": closest_pair,
            "strip": strip.copy(),
            "strip_mid": self.mid_x,
            "strip_width": d,
            "status": f"Created strip of width 2*{d:.2f} around x = {self.mid_x:.2f} with {len(strip)} points"
        })
        
        # Check if there's a closer pair in the strip
        strip_min, strip_pair = self.closest_pair_strip(strip, d, closest_pair)
        
        if strip_min < d:
            closest_pair = strip_pair
            d = strip_min
            
            # Add step for updated minimum from strip
            self.steps.append({
                "phase": "strip_update",
                "points": self.points.copy(),
                "current_points": list(closest_pair),
                "current_dist": d,
                "min_dist": d,
                "closest_pair": closest_pair,
                "strip": strip.copy(),
                "strip_mid": self.mid_x,
                "strip_width": d,
                "status": f"Found closer pair in strip: {closest_pair} with distance {d:.2f}"
            })
        
        return d, closest_pair
    
    def find_closest_pair(self):
        """Find the closest pair of points."""
        if len(self.points) < 2:
            return float('inf'), None
        
        # Initialize steps
        self.steps = []
        
        # Initial step
        self.steps.append({
            "phase": "initial",
            "points": self.points.copy(),
            "current_points": [],
            "current_dist": None,
            "min_dist": None,
            "closest_pair": None,
            "strip": [],
            "strip_mid": None,
            "strip_width": None,
            "status": f"Starting closest pair algorithm with {len(self.points)} points"
        })
        
        # Sort points by x and y coordinates
        points_x = sorted(self.points, key=lambda point: point[0])
        points_y = sorted(self.points, key=lambda point: point[1])
        
        # Run the divide and conquer algorithm
        min_dist, closest_pair = self.closest_pair_recursive(points_x, points_y)
        
        # Final step
        self.steps.append({
            "phase": "final",
            "points": self.points.copy(),
            "current_points": list(closest_pair),
            "current_dist": min_dist,
            "min_dist": min_dist,
            "closest_pair": closest_pair,
            "strip": [],
            "strip_mid": None,
            "strip_width": None,
            "status": f"Final result: Closest pair is {closest_pair} with distance {min_dist:.2f}"
        })
        
        self.min_distance = min_dist
        self.closest_pair = closest_pair
        
        print(f"\nClosest pair of points: {closest_pair}")
        print(f"Minimum distance: {min_dist:.4f}")
        
        return min_dist, closest_pair, self.steps

def visualize_closest_pair_steps(steps):
    """Create a matplotlib visualization of the Closest Pair algorithm steps."""
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.suptitle("Closest Pair of Points (Divide and Conquer)", fontsize=16)
    
    # Create a text object for the status message
    status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12,
                   bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Define animation update function
    def update(frame):
        ax.clear()
        
        step = steps[frame]
        points = step["points"]
        
        # Set title based on the phase
        phase = step["phase"]
        if phase == "initial":
            ax.set_title("Initial Points")
        elif phase == "divide":
            ax.set_title("Dividing Points")
        elif phase == "base_case":
            ax.set_title("Base Case (Brute Force)")
        elif phase == "conquer":
            ax.set_title("Combining Results")
        elif phase == "strip_creation":
            ax.set_title("Creating Strip")
        elif phase == "strip":
            ax.set_title("Checking Points in Strip")
        elif phase == "strip_update":
            ax.set_title("Updated Closest Pair from Strip")
        elif phase == "final":
            ax.set_title("Final Result")
        
        # Plot all points
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        ax.scatter(x, y, color='blue', s=50)
        
        # Add point labels
        for i, (px, py) in enumerate(points):
            ax.annotate(f"P{i}", (px, py), xytext=(5, 5), textcoords='offset points')
        
        # Highlight the points being considered
        if step["current_points"]:
            curr_x = [p[0] for p in step["current_points"]]
            curr_y = [p[1] for p in step["current_points"]]
            ax.scatter(curr_x, curr_y, color='red', s=100, zorder=5)
        
        # Draw the current closest pair
        if step["closest_pair"]:
            p1, p2 = step["closest_pair"]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'g-', linewidth=2)
            
            # Draw a circle with radius = min_distance
            if step["min_dist"]:
                mid_x = (p1[0] + p2[0]) / 2
                mid_y = (p1[1] + p2[1]) / 2
                circle = Circle((mid_x, mid_y), step["min_dist"]/2, fill=False, 
                               edgecolor='green', linestyle='--', alpha=0.7)
                ax.add_patch(circle)
        
        # Draw the dividing line and strip if applicable
        if step["strip_mid"] is not None:
            y_min, y_max = ax.get_ylim()
            ax.plot([step["strip_mid"], step["strip_mid"]], [y_min, y_max], 'k--', alpha=0.5)
            
            # Draw the strip if it exists
            if step["strip_width"] and step["phase"] in ["strip_creation", "strip", "strip_update"]:
                strip_left = step["strip_mid"] - step["strip_width"]
                strip_right = step["strip_mid"] + step["strip_width"]
                ax.axvspan(strip_left, strip_right, alpha=0.2, color='yellow')
                
                # Plot the points in the strip
                if step["strip"]:
                    strip_x = [p[0] for p in step["strip"]]
                    strip_y = [p[1] for p in step["strip"]]
                    ax.scatter(strip_x, strip_y, color='orange', s=70, zorder=4)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Update status message
        status_text.set_text(step["status"])
        
        return ax, status_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(steps), interval=2000, blit=False, repeat_delay=3000)
    
    # Show the animation
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def validate_vertex(vertex, n_vertices):
    """Validate that vertex is within range (1 to n_vertices)."""
    try:
        vertex = int(vertex)
        if vertex < 1 or vertex > n_vertices:
            print(f"Error: Vertex must be between 1 and {n_vertices}")
            return None
        return vertex
    except ValueError:
        print("Error: Vertex must be an integer")
        return None

def get_edge_input(n_vertices):
    """Get edge input from user and validate."""
    while True:
        try:
            edge_input = input("Enter edge as 'node1 node2 weight' (or 'done' to finish, 'menu' to return to main menu): ")
            
            if edge_input.lower() == 'done':
                return None
            
            if edge_input.lower() == 'menu':
                return 'MENU'
                
            # Parse the input
            parts = edge_input.split()
            if len(parts) != 3:
                print("Error: Input must be in format 'node1 node2 weight'")
                continue
                
            u = validate_vertex(parts[0], n_vertices)
            v = validate_vertex(parts[1], n_vertices)
            
            try:
                w = float(parts[2])
            except ValueError:
                print("Error: Weight must be a number")
                continue
                
            if u is not None and v is not None:
                return u, v, w
                
        except Exception as e:
            print(f"Error: {str(e)}")

def visualize_mst(n_vertices, mst_edges):
    """Simple ASCII visualization of the MST."""
    # Create an adjacency list representation of the MST
    adj_list = [[] for _ in range(n_vertices + 1)]  # +1 for 1-indexing
    for u, v, w in mst_edges:
        adj_list[u].append((v, w))
        adj_list[v].append((u, w))
    
    print("\nMinimum Spanning Tree Visualization:")
    print("-" * 40)
    
    for i in range(1, n_vertices + 1):  # Start from 1 for 1-indexing
        connections = []
        for neighbor, weight in adj_list[i]:
            connections.append(f"{neighbor}({weight})")
        
        print(f"Node {i}: {' -> '.join(connections) if connections else 'No connections'}")
    
    print("-" * 40)

def visualize_algorithm_steps(n_vertices, steps, all_edges):
    """Create a matplotlib visualization of the Kruskal's algorithm steps."""
    # Create a NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes (1-indexed)
    for i in range(1, n_vertices + 1):
        G.add_node(i)
    
    # Randomly position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Kruskal's Algorithm Visualization", fontsize=16)
    
    # Create a text object for the status message
    status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12, 
                   bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Define animation update function
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        step = steps[frame]
        
        # Left plot: Current state of MST
        ax1.set_title("Current MST")
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=700, node_color='lightblue')
        
        # Draw MST edges in red
        mst_edges_for_plot = [(u, v) for u, v, _ in step["mst_edges"]]
        if mst_edges_for_plot:
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges_for_plot, ax=ax1, 
                                   width=2.0, edge_color='red')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12)
        
        # Draw edge weights for MST edges
        edge_labels = {(u, v): w for u, v, w in step["mst_edges"]}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1)
        
        # Right plot: Full graph with current edge consideration
        ax2.set_title("All Edges (Considering edges in order of weight)")
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=700, node_color='lightblue')
        
        # Draw all edges in grey
        all_edges_for_plot = [(u, v) for u, v, _ in all_edges]
        nx.draw_networkx_edges(G, pos, edgelist=all_edges_for_plot, ax=ax2, 
                              width=1.0, edge_color='grey', alpha=0.3)
        
        # Draw MST edges in red
        if mst_edges_for_plot:
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges_for_plot, ax=ax2, 
                                   width=2.0, edge_color='red')
        
        # Highlight the edge being considered
        if step["considered_edge"]:
            u, v, _ = step["considered_edge"]
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax2, 
                                   width=2.0, edge_color='blue')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12)
        
        # Draw edge weights for all edges
        edge_labels = {(u, v): w for u, v, w in all_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2)
        
        # Update status message (instead of creating a new one each time)
        status_text.set_text(step["status"])
        
        return ax1, ax2, status_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(steps), interval=2000, blit=False, repeat_delay=3000)
    
    # Show the animation
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def visualize_reverse_delete_steps(n_vertices, steps, all_edges):
    """Create a matplotlib visualization of the Reverse Delete algorithm steps."""
    # Create a NetworkX graph for visualization
    G = nx.Graph()
    
    # Add nodes (1-indexed)
    for i in range(1, n_vertices + 1):
        G.add_node(i)
    
    # Randomly position nodes
    pos = nx.spring_layout(G, seed=42)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle("Reverse Delete Algorithm Visualization", fontsize=16)
    
    # Create a text object for the status message
    status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12, 
                   bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Define animation update function
    def update(frame):
        ax1.clear()
        ax2.clear()
        
        step = steps[frame]
        
        # Left plot: Current state of MST
        ax1.set_title("Current MST")
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, ax=ax1, node_size=700, node_color='lightblue')
        
        # Draw MST edges in red
        mst_edges_for_plot = [(u, v) for u, v, _ in step["mst_edges"]]
        if mst_edges_for_plot:
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges_for_plot, ax=ax1, 
                                   width=2.0, edge_color='red')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax1, font_size=12)
        
        # Draw edge weights for MST edges
        edge_labels = {(u, v): w for u, v, w in step["mst_edges"]}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax1)
        
        # Right plot: Full graph with current edge consideration
        ax2.set_title("All Edges (Considering removal in order of decreasing weight)")
        
        # Draw all nodes
        nx.draw_networkx_nodes(G, pos, ax=ax2, node_size=700, node_color='lightblue')
        
        # Draw all edges in grey
        all_edges_for_plot = [(u, v) for u, v, _ in all_edges]
        nx.draw_networkx_edges(G, pos, edgelist=all_edges_for_plot, ax=ax2, 
                              width=1.0, edge_color='grey', alpha=0.3)
        
        # Draw current MST edges in red
        if mst_edges_for_plot:
            nx.draw_networkx_edges(G, pos, edgelist=mst_edges_for_plot, ax=ax2, 
                                   width=2.0, edge_color='red')
        
        # Highlight the edge being considered
        if step["considered_edge"]:
            u, v, _ = step["considered_edge"]
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax2, 
                                   width=2.0, edge_color='blue')
        
        # Highlight the edge being removed (if any)
        if step["removed_edge"]:
            u, v, _ = step["removed_edge"]
            nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax2, 
                                   width=2.0, edge_color='green', style='dashed')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, ax=ax2, font_size=12)
        
        # Draw edge weights for all edges
        edge_labels = {(u, v): w for u, v, w in all_edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax2)
        
        # Update status message
        status_text.set_text(step["status"])
        
        return ax1, ax2, status_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(steps), interval=2000, blit=False, repeat_delay=3000)
    
    # Show the animation
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def run_kruskal():
    """Run Kruskal's algorithm with user input."""
    print("\n=== Kruskal's Algorithm for Minimum Spanning Tree ===")
    
    while True:
        try:
            n_vertices = int(input("Enter the number of vertices (or 0 to return to main menu): "))
            if n_vertices == 0:
                print("Returning to main menu...")
                return
            if n_vertices < 0:
                print("Number of vertices must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid integer")
    
    print(f"\nVertices are numbered from 1 to {n_vertices}")
    print("Now enter the edges in format: node1 node2 weight")
    print(f"You need to enter {n_vertices-1} edges for a complete MST")
    print("Type 'done' when you've finished entering edges or 'menu' to return to main menu")
    
    kruskal = Kruskal(n_vertices)
    edge_count = 0
    max_edges = n_vertices + 1  # Stop at N+1 edges as requested
    
    while edge_count < max_edges:
        edge = get_edge_input(n_vertices)
        if edge is None:
            break
        if edge == 'MENU':
            print("Returning to main menu...")
            return
            
        u, v, w = edge
        kruskal.add_edge(u, v, w)
        edge_count += 1
        print(f"Added edge: {u} -- {v} with weight {w}")
        
        if edge_count == max_edges:
            print(f"\nReached the maximum of {max_edges} edges. No more edges can be added.")
            break
    
    if not kruskal.graph:
        print("No edges were entered. Cannot find MST.")
        return
    
    print("\nCalculating Minimum Spanning Tree...")
    mst_edges, mst_cost, steps, all_edges = kruskal.kruskal_mst()
    
    if len(mst_edges) < n_vertices - 1:
        print("\nWarning: The graph is not connected. The result is a minimum spanning forest.")
    
    # Display MST value prominently
    print("\n" + "="*40)
    print(f"MST TOTAL VALUE: {mst_cost}")
    print("="*40)
    
    # Simple text visualization of the MST
    visualize_mst(n_vertices, mst_edges)
    
    # Ask if user wants to see the graphical visualization
    show_viz = input("\nWould you like to see a graphical visualization of the algorithm steps? (y/n): ")
    if show_viz.lower() == 'y':
        visualize_algorithm_steps(n_vertices, steps, all_edges)

def run_reverse_delete():
    """Run Reverse Delete algorithm with user input."""
    print("\n=== Reverse Delete Algorithm for Minimum Spanning Tree ===")
    
    while True:
        try:
            n_vertices = int(input("Enter the number of vertices (or 0 to return to main menu): "))
            if n_vertices == 0:
                print("Returning to main menu...")
                return
            if n_vertices < 0:
                print("Number of vertices must be positive")
                continue
            break
        except ValueError:
            print("Please enter a valid integer")
    
    print(f"\nVertices are numbered from 1 to {n_vertices}")
    print("Now enter the edges in format: node1 node2 weight")
    print("Type 'done' when you've finished entering edges or 'menu' to return to main menu")
    
    rev_delete = ReverseDelete(n_vertices)
    edge_count = 0
    max_edges = n_vertices * (n_vertices - 1) // 2  # Maximum number of edges in an undirected graph
    
    while edge_count < max_edges:
        edge = get_edge_input(n_vertices)
        if edge is None:
            break
        if edge == 'MENU':
            print("Returning to main menu...")
            return
            
        u, v, w = edge
        rev_delete.add_edge(u, v, w)
        edge_count += 1
        print(f"Added edge: {u} -- {v} with weight {w}")
    
    if not rev_delete.graph:
        print("No edges were entered. Cannot find MST.")
        return
    
    print("\nCalculating Minimum Spanning Tree using Reverse Delete algorithm...")
    mst_edges, mst_cost, steps, all_edges = rev_delete.reverse_delete_mst()
    
    if len(mst_edges) < n_vertices - 1:
        print("\nWarning: The graph is not connected. The result is a minimum spanning forest.")
    
    # Display MST value prominently
    print("\n" + "="*40)
    print(f"MST TOTAL VALUE: {mst_cost}")
    print("="*40)
    
    # Skip the text visualization and directly show the graphical visualization
    print("\nDisplaying graphical visualization of the algorithm steps...")
    visualize_reverse_delete_steps(n_vertices, steps, all_edges)

def run_closest_pair():
    """Run Closest Pair algorithm with user input."""
    print("\n=== Closest Pair of Points (Divide and Conquer) ===")
    
    closest_pair = ClosestPair()
    
    print("\nYou will need to enter coordinates for points in a 2D plane.")
    print("Enter points as 'x y' (both coordinates should be numbers)")
    print("Type 'done' when you've finished entering points or 'menu' to return to main menu")
    
    point_count = 0
    
    while True:
        point_input = input(f"Enter coordinates for point {point_count + 1} (x y): ")
        
        if point_input.lower() == 'done':
            break
        
        if point_input.lower() == 'menu':
            print("Returning to main menu...")
            return
            
        try:
            # Parse the input
            parts = point_input.split()
            if len(parts) != 2:
                print("Error: Input must be in format 'x y'")
                continue
                
            x = float(parts[0])
            y = float(parts[1])
            
            closest_pair.add_point(x, y)
            point_count += 1
            print(f"Added point ({x}, {y})")
            
        except ValueError:
            print("Error: Coordinates must be numbers")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    if point_count < 2:
        print("At least 2 points are needed to find the closest pair. Returning to main menu.")
        return
    
    print("\nCalculating closest pair of points...")
    min_dist, result_pair, steps = closest_pair.find_closest_pair()
    
    # Display results prominently
    print("\n" + "="*50)
    print("CLOSEST PAIR RESULTS:")
    print("="*50)
    print(f"Point 1: ({result_pair[0][0]:.2f}, {result_pair[0][1]:.2f})")
    print(f"Point 2: ({result_pair[1][0]:.2f}, {result_pair[1][1]:.2f})")
    print(f"Distance: {min_dist:.4f}")
    print("="*50)
    
    # Ask if user wants to see the graphical visualization
    print("\nDisplaying graphical visualization of the algorithm steps...")
    visualize_closest_pair_steps(steps)

class MergeSort:
    def __init__(self):
        self.array = []
        self.steps = []
    
    def add_element(self, value):
        """Add an element to the array."""
        self.array.append(value)
    
    def merge_sort(self, arr=None):
        """Main merge sort function that initializes the steps and starts the recursion."""
        if arr is None:
            arr = self.array.copy()
        
        # Initialize steps
        self.steps = []
        
        # Initial step
        self.steps.append({
            "phase": "initial",
            "full_array": self.array.copy(),
            "current_array": arr.copy(),
            "left_array": None,
            "right_array": None,
            "merged_array": None,
            "level": 0,
            "indices": list(range(len(arr))),
            "status": f"Starting merge sort on array of length {len(arr)}"
        })
        
        # Start recursion
        sorted_arr = self.merge_sort_recursive(arr, 0, list(range(len(arr))))
        
        # Final step
        self.steps.append({
            "phase": "final",
            "full_array": self.array.copy(),
            "current_array": sorted_arr.copy(),
            "left_array": None,
            "right_array": None,
            "merged_array": None,
            "level": 0,
            "indices": list(range(len(arr))),
            "status": f"Array sorted: {sorted_arr}"
        })
        
        return sorted_arr, self.steps
    
    def merge_sort_recursive(self, arr, level, indices):
        """Recursive function to sort using merge sort algorithm."""
        n = len(arr)
        
        # Base case: if array has 0 or 1 element, it's already sorted
        if n <= 1:
            self.steps.append({
                "phase": "base_case",
                "full_array": self.array.copy(),
                "current_array": arr.copy(),
                "left_array": None,
                "right_array": None,
                "merged_array": None,
                "level": level,
                "indices": indices,
                "status": f"Base case: array of length {n} is already sorted: {arr}"
            })
            return arr
        
        # Divide phase
        mid = n // 2
        left_indices = indices[:mid]
        right_indices = indices[mid:]
        
        self.steps.append({
            "phase": "divide",
            "full_array": self.array.copy(),
            "current_array": arr.copy(),
            "left_array": arr[:mid],
            "right_array": arr[mid:],
            "merged_array": None,
            "level": level,
            "indices": indices,
            "status": f"Dividing array {arr} into {arr[:mid]} and {arr[mid:]}"
        })
        
        # Recursive calls
        left = self.merge_sort_recursive(arr[:mid], level + 1, left_indices)
        right = self.merge_sort_recursive(arr[mid:], level + 1, right_indices)
        
        # Merge phase
        self.steps.append({
            "phase": "merge_start",
            "full_array": self.array.copy(),
            "current_array": arr.copy(),
            "left_array": left.copy(),
            "right_array": right.copy(),
            "merged_array": None,
            "level": level,
            "indices": indices,
            "status": f"Merging sorted subarrays: {left} and {right}"
        })
        
        # Merge the two sorted subarrays
        merged = self.merge(left, right, level, indices)
        
        return merged
    
    def merge(self, left, right, level, indices):
        """Merge two sorted arrays."""
        result = []
        i = j = 0
        
        # Detailed merge steps for visualization
        while i < len(left) and j < len(right):
            # Compare elements from both arrays
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
            
            # Add a step showing the current state of the merge
            merge_status = f"Comparing {left[i-1] if i > 0 else 'None'} and {right[j-1] if j > 0 else 'None'}, current result: {result}"
            if i < len(left) and j < len(right):
                merge_status = f"Comparing {left[i]} and {right[j]}, current result: {result}"
            
            self.steps.append({
                "phase": "merge_step",
                "full_array": self.array.copy(),
                "current_array": result + left[i:] + right[j:],
                "left_array": left.copy(),
                "right_array": right.copy(),
                "merged_array": result.copy(),
                "left_ptr": i,
                "right_ptr": j,
                "level": level,
                "indices": indices,
                "status": merge_status
            })
        
        # Handle remaining elements
        if i < len(left):
            result.extend(left[i:])
            self.steps.append({
                "phase": "merge_step",
                "full_array": self.array.copy(),
                "current_array": result,
                "left_array": left.copy(),
                "right_array": right.copy(),
                "merged_array": result.copy(),
                "left_ptr": len(left),
                "right_ptr": j,
                "level": level,
                "indices": indices,
                "status": f"Adding remaining left elements: {left[i:]} to result: {result}"
            })
        
        if j < len(right):
            result.extend(right[j:])
            self.steps.append({
                "phase": "merge_step",
                "full_array": self.array.copy(),
                "current_array": result,
                "left_array": left.copy(),
                "right_array": right.copy(),
                "merged_array": result.copy(),
                "left_ptr": i,
                "right_ptr": len(right),
                "level": level,
                "indices": indices,
                "status": f"Adding remaining right elements: {right[j:]} to result: {result}"
            })
        
        # Final merged array
        self.steps.append({
            "phase": "merge_complete",
            "full_array": self.array.copy(),
            "current_array": result.copy(),
            "left_array": left.copy(),
            "right_array": right.copy(),
            "merged_array": result.copy(),
            "level": level,
            "indices": indices,
            "status": f"Merged result: {result}"
        })
        
        return result

def visualize_merge_sort_steps(steps):
    """Create a matplotlib visualization of the Merge Sort algorithm steps."""
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle("Merge Sort (Divide and Conquer)", fontsize=16)
    
    # Create a text object for the status message
    status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12,
                   bbox={"facecolor":"white", "alpha":0.5, "pad":5})
    
    # Get the original array length for consistent bar positions
    full_array_len = len(steps[0]["full_array"])
    
    # Color map for visualization
    colors = plt.cm.viridis(np.linspace(0, 1, full_array_len))
    
    # Define animation update function
    def update(frame):
        ax.clear()
        
        step = steps[frame]
        full_array = step["full_array"]
        
        # Set title based on the phase
        phase = step["phase"]
        if phase == "initial":
            ax.set_title("Initial Array")
        elif phase == "divide":
            ax.set_title(f"Dividing Array (Level {step['level']})")
        elif phase == "base_case":
            ax.set_title(f"Base Case (Level {step['level']})")
        elif phase == "merge_start":
            ax.set_title(f"Starting Merge (Level {step['level']})")
        elif phase == "merge_step":
            ax.set_title(f"Merging Step (Level {step['level']})")
        elif phase == "merge_complete":
            ax.set_title(f"Merge Complete (Level {step['level']})")
        elif phase == "final":
            ax.set_title("Final Sorted Array")
        
        # Draw the full array as background reference (dimmed)
        positions = np.arange(len(full_array))
        ax.bar(positions, full_array, alpha=0.2, color='gray')
        
        # Determine current positions based on indices
        if step["indices"] is not None:
            indices = step["indices"]
            
            if step["current_array"] is not None:
                current_array = step["current_array"]
                
                # Only plot bars for the current subarray being processed
                for i, pos in enumerate(indices):
                    if i < len(current_array):
                        if phase in ["merge_step", "merge_complete"]:
                            # Use different colors for merged result
                            ax.bar(pos, current_array[i], alpha=0.8, color='green')
                        else:
                            # Use original color mapping
                            ax.bar(pos, current_array[i], alpha=0.8, color=colors[pos])
                
                # Add text labels for values
                for i, pos in enumerate(indices):
                    if i < len(current_array):
                        ax.text(pos, current_array[i] + 0.1, str(current_array[i]), 
                                ha='center', va='bottom', fontsize=10)
            
            # Highlight left and right subarrays during merge
            if phase in ["merge_start", "merge_step"]:
                if step["left_array"] is not None and step["right_array"] is not None:
                    left_indices = indices[:len(step["left_array"])]
                    right_indices = indices[len(step["left_array"]):]
                    
                    # Add a dividing line between left and right
                    if left_indices and right_indices:
                        divider_x = (left_indices[-1] + right_indices[0]) / 2
                        ax.axvline(x=divider_x, color='red', linestyle='--', alpha=0.5)
                    
                    # Highlight merging process
                    if "left_ptr" in step and "right_ptr" in step:
                        left_ptr = step["left_ptr"]
                        right_ptr = step["right_ptr"]
                        
                        # Highlight current elements being compared
                        if left_ptr < len(step["left_array"]) and right_indices:
                            ax.bar(left_indices[left_ptr], step["left_array"][left_ptr], color='orange', alpha=0.8)
                        
                        if right_ptr < len(step["right_array"]) and right_indices:
                            ax.bar(right_indices[right_ptr], step["right_array"][right_ptr], color='orange', alpha=0.8)
        
        # Set y axis limit to accommodate labels
        max_value = max(full_array) if full_array else 1
        ax.set_ylim(0, max_value * 1.2)
        
        # Add grid lines
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add x-axis labels
        ax.set_xticks(positions)
        ax.set_xticklabels([str(i) for i in range(len(full_array))])
        ax.set_xlabel("Array Index")
        ax.set_ylabel("Value")
        
        # Set reasonable x-axis limits
        ax.set_xlim(-0.5, len(full_array) - 0.5)
        
        # Update status message
        status_text.set_text(step["status"])
        
        return ax, status_text
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=len(steps), interval=1000, blit=False, repeat_delay=3000)
    
    # Show the animation
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.show()

def run_merge_sort():
    """Run Merge Sort algorithm with user input."""
    print("\n=== Merge Sort (Divide and Conquer) ===")
    
    merge_sort = MergeSort()
    
    print("\nYou will need to enter elements for the array to be sorted.")
    print("Enter one number at a time. Use integers for best visualization.")
    print("Type 'done' when you've finished entering numbers or 'menu' to return to main menu")
    
    element_count = 0
    
    while True:
        element_input = input(f"Enter value for element {element_count + 1}: ")
        
        if element_input.lower() == 'done':
            break
        
        if element_input.lower() == 'menu':
            print("Returning to main menu...")
            return
            
        try:
            # Parse the input
            value = float(element_input)
            
            # For better visualization, convert to int if possible
            if value.is_integer():
                value = int(value)
            
            merge_sort.add_element(value)
            element_count += 1
            print(f"Added element: {value}")
            
        except ValueError:
            print("Error: Input must be a number")
        except Exception as e:
            print(f"Error: {str(e)}")
    
    if element_count < 2:
        print("At least 2 elements are needed for sorting. Returning to main menu.")
        return
    
    print("\nSorting array using Merge Sort algorithm...")
    print(f"Initial array: {merge_sort.array}")
    
    sorted_array, steps = merge_sort.merge_sort()
    
    # Display results prominently
    print("\n" + "="*50)
    print("MERGE SORT RESULTS:")
    print("="*50)
    print(f"Initial array: {merge_sort.array}")
    print(f"Sorted array: {sorted_array}")
    print("="*50)
    
    # Display graphical visualization
    print("\nDisplaying graphical visualization of the algorithm steps...")
    visualize_merge_sort_steps(steps)

def main():
    print("ALGORITHM SIMULATOR")
    while True:
        try:
            print("\nChoose an algorithm to simulate:")
            print("1. Kruskal's Algorithm")
            print("2. Reverse Delete Algorithm")
            print("3. Divide and Conquer Algorithm")
            print("4. Brute Force Algorithm")
            print("5. Dynamic Programming Algorithm")
            print("0. Exit")
            
            choice = input("Enter your choice (0-5): ")
            
            if choice == '1':
                run_kruskal()
            elif choice == '2':
                run_reverse_delete()
            elif choice == '3':
                run_closest_pair()
            elif choice == '4':
                run_merge_sort()
            elif choice == '5':
                print("Dynamic Programming Algorithm not implemented yet")
            elif choice == '0':
                print("Exiting the simulator.")
                break
            else:
                print("Invalid choice. Please choose a number between 0 and 5.")
        except KeyboardInterrupt:
            print("\nProgram interrupted. Exiting.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()