import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import networkx as nx
from matplotlib.animation import FuncAnimation
import sys
import io
from contextlib import redirect_stdout
import algoCompProj as algo  # Import the original algorithm module

class AlgorithmSimulatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Simulator")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Set theme and style
        self.style = ttk.Style()
        self.style.configure("TButton", font=("Arial", 10))
        self.style.configure("TLabel", font=("Arial", 11))
        self.style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="20")
        self.main_frame.pack(fill="both", expand=True)
        
        # Title
        ttk.Label(self.main_frame, text="Algorithm Simulator", style="Header.TLabel").pack(pady=10)
        
        # Algorithm selection frame
        self.algo_frame = ttk.LabelFrame(self.main_frame, text="Select Algorithm", padding="10")
        self.algo_frame.pack(fill="x", padx=5, pady=5)
        
        # Radio buttons for algorithm selection
        self.selected_algo = tk.StringVar(value="1")
        algos = [
            ("Kruskal's Algorithm", "1"),
            ("Reverse Delete Algorithm", "2"),
            ("Merge Sort (Divide and Conquer)", "3"),
            ("Brute Force Algorithm", "4"),
            ("Dynamic Programming Algorithm", "5")
        ]
        
        for text, value in algos:
            ttk.Radiobutton(self.algo_frame, text=text, value=value, 
                           variable=self.selected_algo).pack(anchor="w", pady=2)
        
        # Input frame
        self.input_frame = ttk.LabelFrame(self.main_frame, text="Input", padding="10")
        self.input_frame.pack(fill="x", padx=5, pady=5)
        
        # Number of vertices
        ttk.Label(self.input_frame, text="Number of vertices:").grid(row=0, column=0, sticky="w", pady=5)
        self.n_vertices = tk.StringVar()
        ttk.Entry(self.input_frame, textvariable=self.n_vertices, width=10).grid(row=0, column=1, sticky="w", pady=5)
        
        # Edges input - MAKE THIS SMALLER
        ttk.Label(self.input_frame, text="Enter edges (format: node1 node2 weight, one per line):").grid(
            row=1, column=0, columnspan=2, sticky="w", pady=5)
        
        # Reduced height from 10 to 6
        self.edges_text = scrolledtext.ScrolledText(self.input_frame, width=40, height=6)
        self.edges_text.grid(row=2, column=0, columnspan=2, sticky="ew", pady=5)
        
        # Add sample data button
        ttk.Button(self.input_frame, text="Load Sample Data", command=self.load_sample_data).grid(
            row=3, column=0, sticky="w", pady=5)
        
        # Buttons frame
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill="x", padx=5, pady=10)
        
        ttk.Button(self.button_frame, text="Run Algorithm", command=self.run_algorithm).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Clear", command=self.clear_form).pack(side="left", padx=5)
        ttk.Button(self.button_frame, text="Exit", command=root.destroy).pack(side="right", padx=5)
        
        # Output frame - Make it proportionally larger
        self.output_frame = ttk.LabelFrame(self.main_frame, text="Output", padding="10")
        self.output_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Use a more reasonable height value (20-25 is typically good for visibility)
        # while still allowing expansion
        self.output_text = scrolledtext.ScrolledText(self.output_frame, width=80, height=25)
        self.output_text.pack(fill="both", expand=True, pady=5)
        
        # Configure text tags for formatting
        self.output_text.tag_configure("bold", font=("Arial", 11, "bold"))
        self.output_text.tag_configure("blue", foreground="blue")
        self.output_text.tag_configure("green", foreground="green")
        self.output_text.tag_configure("red", foreground="red")
        self.output_text.tag_configure("purple", foreground="purple")
        self.output_text.tag_configure("brown", foreground="#8B4513")  # Saddle Brown
        
        # Visualization frame - will be created when needed
        self.viz_window = None
    
    def load_sample_data(self):
        """Load sample graph data based on selected algorithm"""
        algo_type = self.selected_algo.get()
        
        # Clear current data
        self.n_vertices.set("")
        self.edges_text.delete(1.0, tk.END)
        
        # Sample data based on algorithm
        if algo_type in ["1", "2"]:  # MST algorithms
            self.n_vertices.set("5")
            sample_edges = "1 2 2\n1 3 3\n2 3 1\n2 4 3\n3 4 2\n3 5 4\n4 5 5"
            self.edges_text.insert(tk.END, sample_edges)
        elif algo_type == "3":  # Merge Sort
            # We don't need n_vertices for merge sort
            self.n_vertices.set("")
            sample_array = "38\n27\n43\n3\n9\n82\n10"
            self.edges_text.insert(tk.END, sample_array)
        elif algo_type == "4":  # Brute Force
            self.n_vertices.set("")
            sample_array = "8\n2\n5\n1\n9\n3"
            self.edges_text.insert(tk.END, sample_array)
        elif algo_type == "5":  # Dynamic Programming
            self.n_vertices.set("")
            sample_text = "ABCBDAB\nBDCABA"  # Two sequences for LCS
            self.edges_text.insert(tk.END, sample_text)

    def select_dc_algorithm(self):
        """Show dialog to select specific Divide and Conquer algorithm"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Divide and Conquer Algorithm")
        dialog.geometry("400x200")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()  # Make dialog modal
        
        result = [None]  # Use a list to store the result
        
        ttk.Label(dialog, text="Choose Algorithm:", font=("Arial", 12)).pack(pady=10)
        
        # Create frame for radio buttons
        rb_frame = ttk.Frame(dialog, padding=10)
        rb_frame.pack(fill="x")
        
        # Radio buttons
        selected_algo = tk.StringVar(value="closest_pair")
        ttk.Radiobutton(rb_frame, text="Closest Pair of Points", value="closest_pair", 
                       variable=selected_algo).pack(anchor="w", pady=5)
        ttk.Radiobutton(rb_frame, text="Merge Sort", value="merge_sort", 
                       variable=selected_algo).pack(anchor="w", pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(dialog, padding=10)
        btn_frame.pack(fill="x")
        
        def on_ok():
            result[0] = selected_algo.get()
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ttk.Button(btn_frame, text="OK", command=on_ok).pack(side="right", padx=5)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="right", padx=5)
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        return result[0]

    def clear_form(self):
        """Clear all input fields"""
        self.n_vertices.set("")
        self.edges_text.delete(1.0, tk.END)
        self.output_text.delete(1.0, tk.END)
    
    def run_algorithm(self):
        """Run the selected algorithm with the provided inputs"""
        # Clear previous output
        self.output_text.delete(1.0, tk.END)
        
        try:
            # Get input values
            algo_type = self.selected_algo.get()
            
            # For algorithms that use elements instead of graph
            if algo_type in ["3", "4", "5"]:  # Merge Sort, Brute Force, or Dynamic Programming
                # Parse inputs
                elements_text = self.edges_text.get(1.0, tk.END).strip()
                if not elements_text:
                    messagebox.showerror("Input Error", "Please enter input data")
                    return
                
                # Run the appropriate algorithm
                if algo_type == "3":
                    self.run_merge_sort(elements_text)
                elif algo_type == "4":
                    self.run_brute_force(elements_text)
                elif algo_type == "5":
                    self.run_dynamic_programming(elements_text)
                return
            
            # For other algorithms, continue with existing logic
            if not self.n_vertices.get().strip():
                messagebox.showerror("Input Error", "Please specify the number of vertices")
                return
            
            # Parse inputs
            edges_text = self.edges_text.get(1.0, tk.END).strip()
            if not edges_text:
                messagebox.showerror("Input Error", "Please enter at least one edge/point")
                return
            
            # Capture console output
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                if algo_type == "1":
                    n_vertices = int(self.n_vertices.get())
                    self.run_kruskal(n_vertices, edges_text)
                elif algo_type == "2":
                    n_vertices = int(self.n_vertices.get())
                    self.run_reverse_delete(n_vertices, edges_text)
            
            # Display captured output
            self.output_text.insert(tk.END, buffer.getvalue())
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def run_merge_sort(self, elements_text):
        """Run Merge Sort algorithm with the provided inputs and show visualization in output text field"""
        # Create a MergeSort instance
        merge_sort = algo.MergeSort()
        
        # Parse and add elements
        element_lines = elements_text.strip().split('\n')
        elements_added = []
        for line in element_lines:
            line = line.strip()
            if not line:
                continue
            try:
                value = float(line)
                # Convert to int if possible for better visualization
                if value.is_integer():
                    value = int(value)
                merge_sort.add_element(value)
                elements_added.append(value)
            except ValueError:
                continue
        
        if len(merge_sort.array) < 2:
            self.output_text.insert(tk.END, "At least 2 elements are needed for sorting.\n")
            return
        
        # Sort the array and get steps
        self.output_text.insert(tk.END, f"Initial array: {merge_sort.array}\n\n")
        sorted_array, steps = merge_sort.merge_sort()
        
        # Create a simplified text visualization of merge sort steps in the output field
        self.output_text.insert(tk.END, "MERGE SORT STEPS:\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Track levels for indentation
        current_level = 0
        last_level = 0
        level_colors = {0: "black", 1: "blue", 2: "purple", 3: "green", 4: "brown"}
        
        # Process steps excluding initial and final
        for i, step in enumerate(steps):
            phase = step["phase"]
            level = step.get("level", 0)
            current_level = level
            
            # Indent based on recursion level
            indent = "  " * level
            
            # Skip initial step
            if phase == "initial":
                continue
                
            # Add extra line when going up in the recursion tree
            if current_level < last_level:
                self.output_text.insert(tk.END, "\n")
            
            # Get current color for this level
            color = level_colors.get(level, "black")
            
            # Format step based on phase
            if phase == "divide":
                self.output_text.insert(tk.END, f"{indent}Dividing: ", color)
                self.output_text.insert(tk.END, f"{step['current_array']} into {step['left_array']} and {step['right_array']}\n")
            elif phase == "base_case":
                self.output_text.insert(tk.END, f"{indent}Base case: ", color)
                self.output_text.insert(tk.END, f"{step['current_array']} (already sorted)\n")
            elif phase == "merge_start":
                self.output_text.insert(tk.END, f"{indent}Merging: ", color)
                self.output_text.insert(tk.END, f"{step['left_array']} and {step['right_array']}\n")
            elif phase == "merge_complete":
                self.output_text.insert(tk.END, f"{indent}Merged result: ", color)
                self.output_text.insert(tk.END, f"{step['current_array']}\n")
            elif phase == "final":
                self.output_text.insert(tk.END, "\nFINAL SORTED ARRAY: ", "bold")
                self.output_text.insert(tk.END, f"{step['current_array']}\n")
            
            last_level = current_level
        
        # Display final results prominently
        self.output_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.output_text.insert(tk.END, "MERGE SORT RESULTS:\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")
        self.output_text.insert(tk.END, f"Initial array: {merge_sort.array}\n")
        self.output_text.insert(tk.END, f"Sorted array: {sorted_array}\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")

    def create_merge_sort_visualization_window(self, steps):
        """Create a new window with visualization for merge sort algorithm"""
        # Create a new window for visualization
        if self.viz_window is not None and self.viz_window.winfo_exists():
            self.viz_window.destroy()
        
        self.viz_window = tk.Toplevel(self.root)
        self.viz_window.title("Merge Sort Visualization")
        self.viz_window.geometry("1000x700")
        self.viz_window.minsize(800, 600)  # Set minimum size
        
        # Create visualization
        self.create_merge_sort_visualization(self.viz_window, steps)

    def ask_for_merge_sort_visualization(self, steps):
        """Ask user if they want to see merge sort visualization"""
        if messagebox.askyesno("Visualization", "Would you like to see a graphical visualization of the algorithm steps?"):
            # Create a new window for visualization
            if self.viz_window is not None and self.viz_window.winfo_exists():
                self.viz_window.destroy()
            
            self.viz_window = tk.Toplevel(self.root)
            self.viz_window.title("Merge Sort Visualization")
            self.viz_window.geometry("1000x700")
            self.viz_window.minsize(800, 600)  # Set minimum size
            
            # Create visualization
            self.create_merge_sort_visualization(self.viz_window, steps)

    def create_merge_sort_visualization(self, window, steps):
        """Create visualization for merge sort algorithm"""
        # Create main container frame
        main_container = ttk.Frame(window)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for visualization
        viz_frame = ttk.Frame(main_container)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.suptitle("Merge Sort (Divide and Conquer)", fontsize=16)
        
        # Current step index
        self.current_step = 0
        
        # Create a text object for the status message
        status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12,
                       bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Create a canvas to display the plot
        canvas = FigureCanvasTkAgg(fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add results summary frame
        summary_frame = ttk.LabelFrame(main_container, text="Merge Sort Results", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Add results details
        initial_array = steps[0]["full_array"]
        final_array = steps[-1]["current_array"]
        
        result_text = scrolledtext.ScrolledText(summary_frame, height=3, width=80)
        result_text.pack(fill=tk.X, expand=True)
        result_text.insert(tk.END, f"Initial Array: {initial_array}\n")
        result_text.insert(tk.END, f"Sorted Array: {final_array}")
        result_text.config(state=tk.DISABLED)  # Make read-only
        
        # Create controls frame
        controls_frame = ttk.Frame(main_container, padding="10")
        controls_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Step navigation buttons
        controls_inner = ttk.Frame(controls_frame)
        controls_inner.pack(fill=tk.X)
        
        # Navigation buttons with fixed width
        ttk.Button(controls_inner, text="<<", width=3, 
                   command=lambda: self.go_to_step(0, update_plot)).grid(row=0, column=0, padx=5)
        ttk.Button(controls_inner, text="<", width=3,
                   command=lambda: self.go_to_step(max(0, self.current_step-1), update_plot)).grid(row=0, column=1, padx=5)
        
        # Step display label
        step_label = ttk.Label(controls_inner, text=f"Step 1 of {len(steps)}", width=15)
        step_label.grid(row=0, column=2, padx=20)
        
        ttk.Button(controls_inner, text=">", width=3,
                   command=lambda: self.go_to_step(min(len(steps)-1, self.current_step+1), update_plot)).grid(row=0, column=3, padx=5)
        ttk.Button(controls_inner, text=">>", width=3,
                   command=lambda: self.go_to_step(len(steps)-1, update_plot)).grid(row=0, column=4, padx=5)
        
        # Close button
        ttk.Button(controls_inner, text="Close", width=8,
                   command=window.destroy).grid(row=0, column=5, padx=10, sticky=tk.E)
        
        # Center the controls
        controls_inner.grid_columnconfigure(2, weight=1)
        controls_inner.grid_columnconfigure(5, weight=1)
        
        # Get the original array length for consistent bar positions
        full_array_len = len(steps[0]["full_array"])
        
        # Color map for visualization
        colors = plt.cm.viridis(np.linspace(0, 1, full_array_len))
        
        # Update function for visualization
        def update_plot(frame):
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
                            if left_ptr < len(step["left_array"]) and left_indices:
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
            
            # Update step counter label
            step_label.config(text=f"Step {frame+1} of {len(steps)}")
            
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for status
            canvas.draw()
        
        # Initialize with the first step
        update_plot(0)
    
    def go_to_step(self, step_idx, update_func):
        """Navigate to a specific step in the visualization"""
        self.current_step = step_idx
        update_func(step_idx)

    def run_kruskal(self, n_vertices, edges_text):
        """Run Kruskal's algorithm with the provided inputs"""
        # Create Kruskal instance
        kruskal = algo.Kruskal(n_vertices)
        
        # Parse and add edges
        edge_lines = edges_text.strip().split('\n')
        for line in edge_lines:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                w = float(parts[2])
                if 1 <= u <= n_vertices and 1 <= v <= n_vertices:
                    kruskal.add_edge(u, v, w)
            except ValueError:
                continue
        
        if not kruskal.graph:
            print("No valid edges were entered. Cannot find MST.")
            return
        
        # Calculate MST
        print("\nCalculating Minimum Spanning Tree using Kruskal's algorithm...")
        mst_edges, mst_cost, steps, all_edges = kruskal.kruskal_mst()
        
        if len(mst_edges) < n_vertices - 1:
            print("\nWarning: The graph is not connected. The result is a minimum spanning forest.")
        
        # Display MST calculation details
        print("\n" + "="*40)
        print("MINIMUM SPANNING TREE DETAILS:")
        print("="*40)
        print("Edges in the MST:")
        for u, v, w in mst_edges:
            print(f"  {u} -- {v} with weight {w}")
        print("-"*40)
        print(f"MST TOTAL WEIGHT: {mst_cost}")
        print("="*40)
        
        # Ask if user wants to see visualization
        self.ask_for_visualization("Kruskal's", n_vertices, steps, all_edges)
    
    def run_reverse_delete(self, n_vertices, edges_text):
        """Run Reverse Delete algorithm with the provided inputs"""
        # Create Reverse Delete instance
        rev_delete = algo.ReverseDelete(n_vertices)
        
        # Parse and add edges
        edge_lines = edges_text.strip().split('\n')
        for line in edge_lines:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                w = float(parts[2])
                if 1 <= u <= n_vertices and 1 <= v <= n_vertices:
                    rev_delete.add_edge(u, v, w)
            except ValueError:
                continue
        
        if not rev_delete.graph:
            print("No valid edges were entered. Cannot find MST.")
            return
        
        # Calculate MST
        print("\nCalculating Minimum Spanning Tree using Reverse Delete algorithm...")
        mst_edges, mst_cost, steps, all_edges = rev_delete.reverse_delete_mst()
        
        if len(mst_edges) < n_vertices - 1:
            print("\nWarning: The graph is not connected. The result is a minimum spanning forest.")
        
        # Display MST calculation details
        print("\n" + "="*40)
        print("MINIMUM SPANNING TREE DETAILS:")
        print("="*40)
        print("Edges in the MST:")
        for u, v, w in mst_edges:
            print(f"  {u} -- {v} with weight {w}")
        print("-"*40)
        print(f"MST TOTAL WEIGHT: {mst_cost}")
        print("="*40)
        
        # Ask if user wants to see visualization
        self.ask_for_visualization("Reverse Delete", n_vertices, steps, all_edges)
    
    def ask_for_visualization(self, algo_name, n_vertices, steps, all_edges):
        """Ask user if they want to see visualization"""
        if messagebox.askyesno("Visualization", "Would you like to see a graphical visualization of the algorithm steps?"):
            # Create a new window for visualization
            if self.viz_window is not None and self.viz_window.winfo_exists():
                self.viz_window.destroy()
            
            self.viz_window = tk.Toplevel(self.root)
            self.viz_window.title(f"{algo_name} Algorithm Visualization")
            # Larger default size to ensure controls are visible
            self.viz_window.geometry("1000x700")
            self.viz_window.minsize(800, 600)  # Set minimum size
            
            # Create visualization based on algorithm type
            if algo_name == "Kruskal's":
                self.create_visualization(self.viz_window, algo_name, n_vertices, steps, all_edges, 
                                         algo.visualize_algorithm_steps)
            elif algo_name == "Reverse Delete":
                self.create_visualization(self.viz_window, algo_name, n_vertices, steps, all_edges, 
                                         algo.visualize_reverse_delete_steps)
    
    def create_visualization(self, window, algo_name, n_vertices, steps, all_edges, viz_func):
        """Create visualization in a new window"""
        # Create main container frame
        main_container = ttk.Frame(window)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create a frame for visualization
        viz_frame = ttk.Frame(main_container)
        viz_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create a figure with two subplots - reducing figure height to leave room for controls
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"{algo_name} Algorithm Visualization", fontsize=16)
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes (1-indexed)
        for i in range(1, n_vertices + 1):
            G.add_node(i)
        
        # Randomly position nodes
        pos = nx.spring_layout(G, seed=42)
        
        # Current step index
        self.current_step = 0
        
        # Create a text object for the status message
        status_text = fig.text(0.5, 0.01, "", ha="center", fontsize=12, 
                       bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        # Calculate MST cost for summary
        mst_cost = sum(w for _, _, w in steps[-1]["mst_edges"])
        
        # Create a canvas to display the plot
        canvas = FigureCanvasTkAgg(fig, master=viz_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add MST summary frame
        summary_frame = ttk.LabelFrame(main_container, text="MST Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Add MST details
        mst_text = scrolledtext.ScrolledText(summary_frame, height=4, width=80)
        mst_text.pack(fill=tk.X, expand=True)
        
        # Insert MST details
        mst_text.insert(tk.END, "Edges in the MST:\n")
        for u, v, w in steps[-1]["mst_edges"]:
            mst_text.insert(tk.END, f"  {u} -- {v} with weight {w}\n")
        mst_text.insert(tk.END, f"\nMST TOTAL WEIGHT: {mst_cost}")
        mst_text.config(state=tk.DISABLED)  # Make read-only
        
        # Create controls frame with fixed height
        controls_frame = ttk.Frame(main_container, padding="10")
        controls_frame.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=10)
        
        # Step navigation buttons - using grid layout for more control
        controls_inner = ttk.Frame(controls_frame)
        controls_inner.pack(fill=tk.X)
        
        # Navigation buttons with fixed width to ensure visibility
        ttk.Button(controls_inner, text="<<", width=3, 
                   command=lambda: self.go_to_step(0, update_plot)).grid(row=0, column=0, padx=5)
        ttk.Button(controls_inner, text="<", width=3,
                   command=lambda: self.go_to_step(max(0, self.current_step-1), update_plot)).grid(row=0, column=1, padx=5)
        
        # Step display label with fixed width
        step_label = ttk.Label(controls_inner, text=f"Step 1 of {len(steps)}", width=15)
        step_label.grid(row=0, column=2, padx=20)
        
        ttk.Button(controls_inner, text=">", width=3,
                   command=lambda: self.go_to_step(min(len(steps)-1, self.current_step+1), update_plot)).grid(row=0, column=3, padx=5)
        ttk.Button(controls_inner, text=">>", width=3,
                   command=lambda: self.go_to_step(len(steps)-1, update_plot)).grid(row=0, column=4, padx=5)
        
        # Close button on the right side
        ttk.Button(controls_inner, text="Close", width=8,
                   command=window.destroy).grid(row=0, column=5, padx=10, sticky=tk.E)
        
        # Center the controls
        controls_inner.grid_columnconfigure(2, weight=1)
        controls_inner.grid_columnconfigure(5, weight=1)
        
        # Update function - modified to work with GUI
        def update_plot(step_idx):
            ax1.clear()
            ax2.clear()
            
            step = steps[step_idx]
            
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
            if algo_name == "Kruskal's":
                ax2.set_title("All Edges (Considering edges in order of weight)")
            else:
                ax2.set_title("All Edges (Considering removal in order of decreasing weight)")
            
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
            
            # For Reverse Delete, highlight the edge being removed (if any)
            if algo_name == "Reverse Delete" and "removed_edge" in step and step["removed_edge"]:
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
            
            # Update step counter label
            step_label.config(text=f"Step {step_idx+1} of {len(steps)}")
            
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # Adjust layout to make room for status
            canvas.draw()
        
        # Initialize with the first step
        update_plot(0)
    
    def go_to_step(self, step_idx, update_func):
        """Navigate to a specific step in the visualization"""
        self.current_step = step_idx
        update_func(step_idx)

    def run_brute_force(self, elements_text):
        """Run Brute Force sorting algorithm with the provided inputs"""
        # Parse and prepare array
        array = []
        element_lines = elements_text.strip().split('\n')
        
        for line in element_lines:
            line = line.strip()
            if not line:
                continue
            try:
                value = float(line)
                # Convert to int if possible for better visualization
                if value.is_integer():
                    value = int(value)
                array.append(value)
            except ValueError:
                continue
        
        if len(array) < 2:
            self.output_text.insert(tk.END, "At least 2 elements are needed for sorting.\n")
            return
        
        # Display initial array
        self.output_text.insert(tk.END, f"Initial array: {array}\n\n")
        self.output_text.insert(tk.END, "BRUTE FORCE SORTING ALGORITHM:\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n\n")
        
        # Display algorithm process
        self.output_text.insert(tk.END, "This algorithm compares each element with every other element\n")
        self.output_text.insert(tk.END, "and swaps them if they are in the wrong order.\n\n")
        
        # Perform bubble sort (as an example of brute force sorting)
        steps = []
        sorted_array = array.copy()
        n = len(sorted_array)
        
        # Track the number of comparisons and swaps
        comparisons = 0
        swaps = 0
        
        # Record the initial state
        steps.append({
            "array": sorted_array.copy(),
            "i": -1,
            "j": -1,
            "comparison": None,
            "swap": None,
            "comparisons": comparisons,
            "swaps": swaps,
            "phase": "initial"
        })
        
        # Perform bubble sort
        for i in range(n):
            # Flag to optimize if no swaps occur in a pass
            swapped = False
            
            for j in range(0, n - i - 1):
                # Record the state before comparison
                steps.append({
                    "array": sorted_array.copy(),
                    "i": i,
                    "j": j,
                    "comparison": [j, j+1],
                    "swap": None,
                    "comparisons": comparisons,
                    "swaps": swaps,
                    "phase": "comparison"
                })
                
                # Increment comparison counter
                comparisons += 1
                
                # Compare adjacent elements
                if sorted_array[j] > sorted_array[j + 1]:
                    # Record the state before swap
                    steps.append({
                        "array": sorted_array.copy(),
                        "i": i,
                        "j": j,
                        "comparison": [j, j+1],
                        "swap": [j, j+1],
                        "comparisons": comparisons,
                        "swaps": swaps,
                        "phase": "swap_needed"
                    })
                    
                    # Swap elements
                    sorted_array[j], sorted_array[j + 1] = sorted_array[j + 1], sorted_array[j]
                    swapped = True
                    swaps += 1
                    
                    # Record the state after swap
                    steps.append({
                        "array": sorted_array.copy(),
                        "i": i,
                        "j": j,
                        "comparison": [j, j+1],
                        "swap": [j, j+1],
                        "comparisons": comparisons,
                        "swaps": swaps,
                        "phase": "swap_complete"
                    })
                else:
                    # Record the state when no swap is needed
                    steps.append({
                        "array": sorted_array.copy(),
                        "i": i,
                        "j": j,
                        "comparison": [j, j+1],
                        "swap": None,
                        "comparisons": comparisons,
                        "swaps": swaps,
                        "phase": "no_swap"
                    })
            
            # If no swaps occurred in this pass, the array is already sorted
            if not swapped:
                break
        
        # Record the final state
        steps.append({
            "array": sorted_array.copy(),
            "i": -1,
            "j": -1,
            "comparison": None,
            "swap": None,
            "comparisons": comparisons,
            "swaps": swaps,
            "phase": "final"
        })
        
        # Display the steps in the output text area
        for i, step in enumerate(steps):
            phase = step["phase"]
            
            if phase == "initial":
                self.output_text.insert(tk.END, "Starting the sorting process with bubble sort algorithm\n\n")
                self.output_text.insert(tk.END, f"Array: {step['array']}\n\n")
                
            elif phase == "comparison":
                j = step["j"]
                self.output_text.insert(tk.END, f"Pass {step['i']+1}, Step {j+1}: ", "bold")
                self.output_text.insert(tk.END, f"Comparing elements at positions {j} and {j+1}\n")
                self.output_text.insert(tk.END, f"   {step['array'][j]} <=> {step['array'][j+1]}\n")
                
            elif phase == "swap_needed":
                j = step["j"]
                self.output_text.insert(tk.END, f"   {step['array'][j]} > {step['array'][j+1]}, need to swap\n", "red")
                
            elif phase == "swap_complete":
                j = step["j"]
                prev_array = steps[i-1]["array"]
                self.output_text.insert(tk.END, f"   Swapped: {prev_array[j]} and {prev_array[j+1]}\n", "green")
                self.output_text.insert(tk.END, f"   Array after swap: {step['array']}\n\n")
                
            elif phase == "no_swap":
                j = step["j"]
                self.output_text.insert(tk.END, f"   {step['array'][j]} <= {step['array'][j+1]}, no swap needed\n", "blue")
                self.output_text.insert(tk.END, f"   Array remains: {step['array']}\n\n")
                
            elif phase == "final":
                self.output_text.insert(tk.END, "\nSorting complete!\n", "bold")
                self.output_text.insert(tk.END, f"Final array: {step['array']}\n")
                self.output_text.insert(tk.END, f"Total comparisons: {step['comparisons']}\n")
                self.output_text.insert(tk.END, f"Total swaps: {step['swaps']}\n")
        
        # Display final results prominently
        self.output_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.output_text.insert(tk.END, "BRUTE FORCE SORTING RESULTS:\n", "bold")
        self.output_text.insert(tk.END, "=" * 50 + "\n")
        self.output_text.insert(tk.END, f"Initial array: {array}\n")
        self.output_text.insert(tk.END, f"Sorted array: {sorted_array}\n")
        self.output_text.insert(tk.END, f"Comparisons made: {comparisons}\n")
        self.output_text.insert(tk.END, f"Swaps performed: {swaps}\n")
        self.output_text.insert(tk.END, "Time Complexity: O(n²)\n")
        self.output_text.insert(tk.END, "Space Complexity: O(1)\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")

    def run_dynamic_programming(self, input_text):
        """Run the Longest Common Subsequence (LCS) dynamic programming algorithm"""
        # Split the input into two sequences
        sequences = input_text.strip().split('\n')
        if len(sequences) < 2:
            self.output_text.insert(tk.END, "Error: Two sequences are required for LCS algorithm.\n")
            self.output_text.insert(tk.END, "Enter one sequence per line.\n")
            return
        
        # Get the two sequences
        X = sequences[0].strip()
        Y = sequences[1].strip()
        
        if not X or not Y:
            self.output_text.insert(tk.END, "Error: Both sequences must not be empty.\n")
            return
        
        # Display initial information
        self.output_text.insert(tk.END, "LONGEST COMMON SUBSEQUENCE (LCS) ALGORITHM\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n\n")
        self.output_text.insert(tk.END, f"Sequence X: {X}\n")
        self.output_text.insert(tk.END, f"Sequence Y: {Y}\n\n")
        
        # Initialize LCS table and steps
        m = len(X)
        n = len(Y)
        lcs_table = [[0 for _ in range(n+1)] for _ in range(m+1)]
        steps = []
        
        # Record initial state
        steps.append({
            "phase": "initial",
            "table": [row[:] for row in lcs_table],
            "i": -1,
            "j": -1,
            "X": X,
            "Y": Y,
            "current_chars": None,
            "match": None,
            "lcs_length": 0,
            "status": "Initializing LCS table with zeros"
        })
        
        # Fill LCS table and record steps
        for i in range(1, m+1):
            for j in range(1, n+1):
                # Record state before comparison
                steps.append({
                    "phase": "comparison",
                    "table": [row[:] for row in lcs_table],
                    "i": i,
                    "j": j,
                    "X": X,
                    "Y": Y,
                    "current_chars": [X[i-1], Y[j-1]],
                    "match": None,
                    "lcs_length": lcs_table[i-1][j-1],
                    "status": f"Comparing characters: '{X[i-1]}' and '{Y[j-1]}'"
                })
                
                if X[i-1] == Y[j-1]:
                    # Characters match
                    lcs_table[i][j] = lcs_table[i-1][j-1] + 1
                    steps.append({
                        "phase": "match",
                        "table": [row[:] for row in lcs_table],
                        "i": i,
                        "j": j,
                        "X": X,
                        "Y": Y,
                        "current_chars": [X[i-1], Y[j-1]],
                        "match": True,
                        "lcs_length": lcs_table[i][j],
                        "status": f"Match found: '{X[i-1]}' == '{Y[j-1]}', LCS length incremented to {lcs_table[i][j]}"
                    })
                else:
                    # Characters don't match, take max from left or top
                    lcs_table[i][j] = max(lcs_table[i-1][j], lcs_table[i][j-1])
                    steps.append({
                        "phase": "no_match",
                        "table": [row[:] for row in lcs_table],
                        "i": i,
                        "j": j,
                        "X": X,
                        "Y": Y,
                        "current_chars": [X[i-1], Y[j-1]],
                        "match": False,
                        "lcs_length": lcs_table[i][j],
                        "status": f"No match: '{X[i-1]}' != '{Y[j-1]}', taking max({lcs_table[i-1][j]}, {lcs_table[i][j-1]}) = {lcs_table[i][j]}"
                    })
        
        # Record final state
        steps.append({
            "phase": "final",
            "table": [row[:] for row in lcs_table],
            "i": m,
            "j": n,
            "X": X,
            "Y": Y,
            "current_chars": None,
            "match": None,
            "lcs_length": lcs_table[m][n],
            "status": f"Completed LCS table. Length of LCS is {lcs_table[m][n]}"
        })
        
        # Find one LCS (there could be multiple)
        lcs = self.traceback_lcs(lcs_table, X, Y, m, n)
        
        # Display calculation process in the output text
        self.output_text.insert(tk.END, "DYNAMIC PROGRAMMING CALCULATION PROCESS:\n")
        self.output_text.insert(tk.END, "-" * 50 + "\n\n")
        
        # Show table headers
        self.output_text.insert(tk.END, "LCS Table Construction:\n\n")
        self.print_lcs_table_header(Y)
        
        for step in steps:
            phase = step["phase"]
            
            if phase == "initial":
                self.output_text.insert(tk.END, "Initial table filled with zeros:\n")
                self.print_lcs_table(step["table"], X, Y)
                self.output_text.insert(tk.END, "\n")
                
            elif phase == "comparison":
                i, j = step["i"], step["j"]
                x_char, y_char = step["current_chars"]
                self.output_text.insert(tk.END, f"Step [{i},{j}]: Comparing '{x_char}' and '{y_char}'\n")
                
            elif phase == "match":
                i, j = step["i"], step["j"]
                x_char, y_char = step["current_chars"]
                self.output_text.insert(tk.END, f"  Match found! Setting table[{i}][{j}] = table[{i-1}][{j-1}] + 1 = {step['lcs_length']}\n", "green")
                
            elif phase == "no_match":
                i, j = step["i"], step["j"]
                x_char, y_char = step["current_chars"]
                self.output_text.insert(tk.END, f"  No match. Setting table[{i}][{j}] = max(table[{i-1}][{j}], table[{i}][{j-1}]) = {step['lcs_length']}\n", "red")
                
            elif phase == "final":
                self.output_text.insert(tk.END, "\nFinal LCS table:\n")
                self.print_lcs_table(step["table"], X, Y)
                self.output_text.insert(tk.END, "\n")
        
        # Display tracing back the LCS
        self.output_text.insert(tk.END, "TRACEBACK PROCESS TO FIND THE LCS:\n")
        self.output_text.insert(tk.END, "-" * 50 + "\n\n")
        
        # Traceback animation
        current_lcs = []
        i, j = m, n
        
        while i > 0 and j > 0:
            if X[i-1] == Y[j-1]:
                # If current characters match, they're part of the LCS
                current_lcs.insert(0, X[i-1])
                self.output_text.insert(tk.END, f"At position ({i},{j}): Characters match ('{X[i-1]}'). Add to LCS.\n", "blue")
                self.output_text.insert(tk.END, f"  Current LCS: {''.join(current_lcs)}\n")
                self.output_text.insert(tk.END, f"  Moving diagonally to ({i-1},{j-1})\n\n")
                i -= 1
                j -= 1
            elif lcs_table[i-1][j] >= lcs_table[i][j-1]:
                # Move up
                self.output_text.insert(tk.END, f"At position ({i},{j}): No match. table[{i-1}][{j}] >= table[{i}][{j-1}]\n")
                self.output_text.insert(tk.END, f"  Moving up to ({i-1},{j})\n\n")
                i -= 1
            else:
                # Move left
                self.output_text.insert(tk.END, f"At position ({i},{j}): No match. table[{i-1}][{j}] < table[{i}][{j-1}]\n")
                self.output_text.insert(tk.END, f"  Moving left to ({i},{j-1})\n\n")
                j -= 1
        
        # Display final results prominently
        self.output_text.insert(tk.END, "\n" + "=" * 50 + "\n")
        self.output_text.insert(tk.END, "DYNAMIC PROGRAMMING RESULTS:\n", "bold")
        self.output_text.insert(tk.END, "=" * 50 + "\n")
        self.output_text.insert(tk.END, f"Sequence X: {X}\n")
        self.output_text.insert(tk.END, f"Sequence Y: {Y}\n")
        self.output_text.insert(tk.END, f"Longest Common Subsequence: {lcs}\n")
        self.output_text.insert(tk.END, f"Length of LCS: {lcs_table[m][n]}\n")
        self.output_text.insert(tk.END, "Time Complexity: O(m×n)\n")
        self.output_text.insert(tk.END, "Space Complexity: O(m×n)\n")
        self.output_text.insert(tk.END, "=" * 50 + "\n")

    def traceback_lcs(self, dp_table, X, Y, i, j):
        """Trace back through the DP table to find one LCS"""
        if i == 0 or j == 0:
            return ""
        
        if X[i-1] == Y[j-1]:
            return self.traceback_lcs(dp_table, X, Y, i-1, j-1) + X[i-1]
        
        if dp_table[i-1][j] > dp_table[i][j-1]:
            return self.traceback_lcs(dp_table, X, Y, i-1, j)
        else:
            return self.traceback_lcs(dp_table, X, Y, i, j-1)

    def print_lcs_table_header(self, Y):
        """Print the header row for the LCS table"""
        header = "    |   | "
        for char in Y:
            header += f"{char} | "
        self.output_text.insert(tk.END, header + "\n")
        self.output_text.insert(tk.END, "-" * (len(header)) + "\n")

    def print_lcs_table(self, table, X, Y):
        """Print the LCS table in a formatted way"""
        # Print header row with Y characters
        header = "    |   | "
        for char in Y:
            header += f"{char} | "
        self.output_text.insert(tk.END, header + "\n")
        self.output_text.insert(tk.END, "-" * (len(header)) + "\n")
        
        # Print table rows
        for i in range(len(table)):
            if i == 0:
                row = "    | "
            else:
                row = f" {X[i-1]} | "
            
            for j in range(len(table[0])):
                row += f" {table[i][j]} | "
            
            self.output_text.insert(tk.END, row + "\n")
            self.output_text.insert(tk.END, "-" * (len(row)) + "\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmSimulatorGUI(root)
    root.mainloop()