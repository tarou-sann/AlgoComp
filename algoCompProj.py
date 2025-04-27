import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import time
from matplotlib.animation import FuncAnimation

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

def main():
    print("ALGORITHM SIMULATOR")
    while True:
        try:
            print("\nChoose an algorithm to simulate:")
            print("1. Kruskal's Algorithm")
            print("2. Reverse Delete Algorithm")
            print("3. Divide and Conquer Algorithm")
            print("4. String Algorithm")
            print("5. Dynamic Programming Algorithm")
            print("0. Exit")
            
            choice = input("Enter your choice (0-5): ")
            
            if choice == '1':
                run_kruskal()
            elif choice == '2':
                print("Reverse Delete Algorithm not implemented yet")
            elif choice == '3':
                print("Divide and Conquer Algorithm not implemented yet")
            elif choice == '4':
                print("String Algorithm not implemented yet")
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