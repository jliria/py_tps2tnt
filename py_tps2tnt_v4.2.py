import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
import morphops as mops
import numpy as np
from collections import defaultdict
from scipy.stats import sem, bootstrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D

class py_tps2tntApp:
    def __init__(self, root):
        self.root = root
        self.root.title("py_tps2tnt  J. Liria, 2024")
        self.species_count = {}
        self.reference_species = None
        self.use_first_as_ref = tk.BooleanVar()
        self.interval_type = tk.StringVar(value="CI")
        self.confidence_level = tk.DoubleVar(value=0.95)
        self.multi_specimen = False
        self.selected_landmarks = []
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Load TPS File
        self.load_button = ttk.Button(main_frame, text="Load TPS File", command=self.load_tps_file)
        self.load_button.grid(row=0, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Load a TPS file containing landmarks.").grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        # Select Reference Species
        self.select_ref_button = ttk.Button(main_frame, text="Select Reference Species", command=self.select_reference_species, state='disabled')
        self.select_ref_button.grid(row=1, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Select a species to use as reference.").grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        # Use First Configuration as Reference
        self.first_as_ref_check = ttk.Checkbutton(main_frame, text="Use First Configuration as Reference", variable=self.use_first_as_ref)
        self.first_as_ref_check.grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)

        # Interval Type Option
        ttk.Label(main_frame, text="Interval Type:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.interval_options = ttk.Combobox(main_frame, textvariable=self.interval_type, values=["CI", "Mean ± SE"], state='disabled')
        self.interval_options.grid(row=3, column=1, padx=5, pady=5, sticky=tk.W)

        # Confidence Level
        ttk.Label(main_frame, text="Confidence Level:").grid(row=4, column=0, padx=5, pady=5, sticky=tk.W)
        self.confidence_entry = ttk.Entry(main_frame, textvariable=self.confidence_level, state='disabled')
        self.confidence_entry.grid(row=4, column=1, padx=5, pady=5, sticky=tk.W)

        # Load Specimen Counts Button (initially disabled)
        self.load_counts_button = ttk.Button(main_frame, text="Load Specimen Counts", command=self.load_specimen_counts, state='disabled')
        self.load_counts_button.grid(row=5, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Load a file with species and their specimen counts.").grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

        # Run GPA
        self.run_gpa_button = ttk.Button(main_frame, text="Run GPA", command=self.run_gpa, state='disabled')
        self.run_gpa_button.grid(row=6, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Perform Generalized Procrustes Analysis.").grid(row=6, column=1, padx=5, pady=5, sticky=tk.W)

        # Show Results
        self.show_results_button = ttk.Button(main_frame, text="Show Results", command=self.show_results, state='disabled')
        self.show_results_button.grid(row=7, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Show the aligned landmarks.").grid(row=7, column=1, padx=5, pady=5, sticky=tk.W)

        # Save Results
        self.save_results_button = ttk.Button(main_frame, text="Save Results", command=self.save_results, state='disabled')
        self.save_results_button.grid(row=8, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Save the results to files.").grid(row=8, column=1, padx=5, pady=5, sticky=tk.W)

        # Select Landmarks for EDMA
        self.select_landmarks_button = ttk.Button(main_frame, text="Select Landmarks for EDMA", command=self.select_landmarks_for_edma, state='disabled')
        self.select_landmarks_button.grid(row=9, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Select specific landmarks for EDMA.").grid(row=9, column=1, padx=5, pady=5, sticky=tk.W)

        # Export TNT File
        self.export_tnt_button = ttk.Button(main_frame, text="Export TNT File", command=self.export_tnt_file, state='disabled')
        self.export_tnt_button.grid(row=10, column=0, padx=5, pady=5)
        ttk.Label(main_frame, text="Export the results to a TNT file.").grid(row=10, column=1, padx=5, pady=5, sticky=tk.W)

        # Status Message
        self.status_message = ttk.Label(main_frame, text="Please load a TPS file to start.")
        self.status_message.grid(row=11, column=0, columnspan=2, padx=5, pady=5, sticky=(tk.W, tk.E))

    def load_tps_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("TPS Files", "*.tps"), ("All Files", "*.*")])
        if not self.file_path:
            return
        self.landmarks, self.specimen_ids = self.read_tps_file(self.file_path)
        self.original_centroid_sizes = [self.calculate_centroid_size(landmarks) for landmarks in self.landmarks]

        self.multi_specimen = messagebox.askyesno("Multiple Specimens", "Does this file contain multiple specimens per species?")
        if self.multi_specimen:
            # Enable the Load Specimen Counts button
            self.load_counts_button.state(['!disabled'])
            self.load_specimen_counts()  # Cargar conteos directamente
            self.calculate_means_by_species()
            self.save_original_centroid_sizes()
            self.save_mean_configurations()
            self.interval_options.state(['!disabled'])
            self.confidence_entry.state(['!disabled'])
        else:
            self.save_centroid_sizes_for_single()

        self.select_ref_button.state(['!disabled'])
        self.run_gpa_button.state(['!disabled'])
        self.select_landmarks_button.state(['!disabled'])
        self.status_message.config(text="TPS file loaded successfully.")

    def read_tps_file(self, file_path):
        landmarks = []
        specimen_ids = []
        scale_factor = 1.0  # Default scale value (if not SCALE)

        with open(file_path, 'r') as file:
            lines = file.readlines()
            current_landmarks = []
            current_id = None
            for line in lines:
                if line.startswith("LM="):
                    if current_landmarks and current_id:
                        # Apply if is necessary
                        if scale_factor != 1.0:
                            current_landmarks = [[coord[0] * scale_factor, coord[1] * scale_factor] for coord in current_landmarks]
                        landmarks.append(current_landmarks)
                        specimen_ids.append(current_id)
                        current_landmarks = []
                    current_id = None
                elif line.startswith("ID="):
                    current_id = line.strip().split('=')[1]
                elif line.startswith("SCALE="):
                    scale_factor = float(line.strip().split('=')[1])
                else:
                    try:
                        coords = list(map(float, line.strip().split()))
                        if coords:
                            current_landmarks.append(coords)
                    except ValueError:
                        continue
            if current_landmarks and current_id:
                # Apply scaling if necessary
                if scale_factor != 1.0:
                    current_landmarks = [[coord[0] * scale_factor, coord[1] * scale_factor] for coord in current_landmarks]
                landmarks.append(current_landmarks)
                specimen_ids.append(current_id)
        return landmarks, specimen_ids

    def load_specimen_counts(self):
        counts_file_path = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not counts_file_path:
            return

        try:
            with open(counts_file_path, 'r') as file:
                for line in file:
                    species, count = line.strip().split()
                    self.species_count[species] = int(count)
            messagebox.showinfo("Success", "Specimen counts loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load specimen counts: {e}")

    def calculate_means_by_species(self):
        species_groups = defaultdict(list)
        for specimen_id, landmark in zip(self.specimen_ids, self.landmarks):
            species_groups[specimen_id].append(landmark)

        species_means = {}
        self.mean_centroid_sizes = {}
        for species, count in self.species_count.items():
            all_landmarks = np.array(species_groups[species])
            species_mean = np.mean(all_landmarks[:count], axis=0)
            species_means[species] = species_mean
            self.mean_centroid_sizes[species] = self.calculate_centroid_size(species_mean)

        self.landmarks = list(species_means.values())
        self.specimen_ids = list(species_means.keys())

    def select_reference_species(self):
        species_list = list(set(self.specimen_ids))
        self.reference_species = simpledialog.askstring("Select Reference Species", "Enter the species to use as reference:", initialvalue=species_list[0])
        if self.reference_species not in species_list:
            messagebox.showerror("Error", "Selected species not found in the data.")
            self.reference_species = None
        else:
            self.status_message.config(text=f"Reference species selected: {self.reference_species}")

    def run_gpa(self):
        if not self.landmarks:
            messagebox.showerror("Error", "No landmarks loaded")
            return

        if self.reference_species is not None:
            ref_index = self.specimen_ids.index(self.reference_species)
            ref_landmarks = self.landmarks[ref_index]
            # Reorder landmarks to put the reference species first
            self.landmarks.insert(0, self.landmarks.pop(ref_index))
            self.specimen_ids.insert(0, self.specimen_ids.pop(ref_index))

        if self.use_first_as_ref.get():
            aligned_data = mops.gpa(self.landmarks)
        else:
            aligned_data = mops.gpa(self.landmarks)

        self.aligned_landmarks = aligned_data['aligned']
        self.mean_shape = aligned_data['mean']

        self.centroid_sizes = [self.calculate_centroid_size(landmarks) for landmarks in self.aligned_landmarks]

        self.show_results_button.state(['!disabled'])
        self.save_results_button.state(['!disabled'])
        self.export_tnt_button.state(['!disabled'])
        self.status_message.config(text="GPA completed successfully.")

    def calculate_centroid_size(self, landmarks):
        centroid = np.mean(landmarks, axis=0)
        size = np.sqrt(np.sum(np.square(np.linalg.norm(landmarks - centroid, axis=1))))
        return size

    def show_results(self):
        if self.aligned_landmarks is None:
            messagebox.showerror("Error", "No GPA results available")
            return

        plt.figure()
        for i, landmarks in enumerate(self.aligned_landmarks):
            landmarks = np.array(landmarks)
            plt.scatter(landmarks[:, 0], landmarks[:, 1], label=self.specimen_ids[i])
        plt.legend()
        plt.title("Aligned Landmarks")
        plt.show()

        mean_centroid_size = np.mean(self.centroid_sizes)
        messagebox.showinfo("Centroid Sizes", f"Mean Centroid Size: {mean_centroid_size}")

    def save_results(self):
        if self.aligned_landmarks is None:
            messagebox.showerror("Error", "No results to save")
            return

        messagebox.showinfo("Save TPS", "Save aligned GPA configurations...")
        save_tps_path = filedialog.asksaveasfilename(defaultextension=".tps", filetypes=[("TPS Files", "*.tps"), ("All Files", "*.*")])
        if save_tps_path:
            self.save_tps(save_tps_path, self.aligned_landmarks, self.specimen_ids)

        messagebox.showinfo("Save Completed", "Results saved successfully.")

    def save_original_centroid_sizes(self):
        messagebox.showinfo("Save TXT", "Save mean species centroid size...")
        save_txt_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if save_txt_path:
            with open(save_txt_path, 'w') as file:
                for species, size in self.mean_centroid_sizes.items():
                    file.write(f"{species}: {size:.6f}\n")

    def save_centroid_sizes_for_single(self):
        messagebox.showinfo("Save TXT", "Save centroid sizes for individual specimens...")
        save_txt_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if save_txt_path:
            with open(save_txt_path, 'w') as file:
                for specimen_id, size in zip(self.specimen_ids, self.original_centroid_sizes):
                    file.write(f"{specimen_id}: {size:.6f}\n")

    def save_mean_configurations(self):
        messagebox.showinfo("Save TPS", "Save mean species coordinates...")
        save_mean_tps_path = filedialog.asksaveasfilename(defaultextension=".tps", filetypes=[("TPS Files", "*.tps"), ("All Files", "*.*")], title="Save Mean Configurations TPS")
        if save_mean_tps_path:
            self.save_tps(save_mean_tps_path, self.landmarks, self.specimen_ids)

    def save_tps(self, file_path, landmarks, specimen_ids):
        with open(file_path, 'w') as file:
            for i, lmk_set in enumerate(landmarks):
                file.write(f"LM={len(lmk_set)}\n")
                for lmk in lmk_set:
                    file.write(f"{lmk[0]:.6f} {lmk[1]:.6f}\n")
                file.write(f"ID={specimen_ids[i]}\n")
                file.write("\n")

    def calculate_edma_distances(self, landmarks, selected_landmarks=None):
        if selected_landmarks is None:
            selected_landmarks = range(len(landmarks))
        distances = []
        for i in selected_landmarks:
            for j in selected_landmarks:
                if i < j:
                    distance = np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))
                    distances.append(distance)
        return distances

    def calculate_intervals(self, landmarks_list, interval_type, confidence_level, selected_landmarks=None):
        distances_matrix = [self.calculate_edma_distances(landmarks, selected_landmarks) for landmarks in landmarks_list]
        distances_matrix = np.array(distances_matrix)

        if len(distances_matrix) < 2:
            return [f"{dist:.6f}" for dist in distances_matrix[0]]

        if interval_type == "CI":
            ci_low, ci_high = [], []
            for i in range(distances_matrix.shape[1]):
                res = bootstrap((distances_matrix[:, i],), np.mean, confidence_level=confidence_level)
                ci_low.append(res.confidence_interval.low)
                ci_high.append(res.confidence_interval.high)
            return [f"{low:.6f}-{high:.6f}" for low, high in zip(ci_low, ci_high)]

        elif interval_type == "Mean ± SE":
            mean_values = np.mean(distances_matrix, axis=0)
            min_values = mean_values - sem(distances_matrix, axis=0)
            max_values = mean_values + sem(distances_matrix, axis=0)
            return [f"{min_val:.6f}-{max_val:.6f}" for min_val, max_val in zip(min_values, max_values)]

    def select_landmarks_for_edma(self):
        if not self.landmarks:
            messagebox.showerror("Error", "No landmarks loaded")
            return

        landmarks_count = len(self.landmarks[0])
        landmark_list = [f"Landmark {i+1}" for i in range(landmarks_count)]

        self.selection_window = tk.Toplevel(self.root)
        self.selection_window.title("Select Landmarks for EDMA")

        tk.Label(self.selection_window, text="Select Landmarks:").pack(padx=10, pady=5)
        self.landmark_listbox = tk.Listbox(self.selection_window, selectmode=tk.MULTIPLE, exportselection=False)
        self.landmark_listbox.bind('<<ListboxSelect>>', self.on_landmark_select)  # Bind the selection event
        for landmark in landmark_list:
            self.landmark_listbox.insert(tk.END, landmark)
        self.landmark_listbox.pack(padx=10, pady=5)

        ttk.Button(self.selection_window, text="OK", command=self.confirm_landmark_selection).pack(pady=10)

        self.figure, self.ax = plt.subplots()
        self.ax.set_title('Selected Landmarks and Connections')

        # Embed the matplotlib figure into the Tkinter window
        canvas = FigureCanvasTkAgg(self.figure, master=self.selection_window)
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.draw_landmarks()

    def on_landmark_select(self, event):
        selected_indices = self.landmark_listbox.curselection()
        self.selected_landmarks = [index for index in selected_indices]
        self.draw_landmarks()  # Update the figure with the selected landmarks

    def draw_landmarks(self):
        self.ax.clear()
        first_landmarks = np.array(self.landmarks[0])
        self.ax.scatter(first_landmarks[:, 0], first_landmarks[:, 1], c='blue')

        for i, txt in enumerate(range(len(first_landmarks))):
            self.ax.annotate(f'{txt+1}', (first_landmarks[i, 0], first_landmarks[i, 1]))

        for i in range(len(first_landmarks)):
            for j in range(i + 1, len(first_landmarks)):
                if i in self.selected_landmarks and j in self.selected_landmarks:
                    line = Line2D([first_landmarks[i, 0], first_landmarks[j, 0]],
                                  [first_landmarks[i, 1], first_landmarks[j, 1]],
                                  lw=2, color='red')
                    self.ax.add_line(line)
        self.figure.canvas.draw()

    def confirm_landmark_selection(self):
        self.status_message.config(text=f"{len(self.selected_landmarks)} landmarks selected for EDMA.")
        self.selection_window.destroy()

    def export_tnt_file(self):
        options = ExportOptionsDialog(self.root, self.interval_type.get(), self.confidence_level.get())
        self.root.wait_window(options.top)
        if not options.ok:
            return

        title = options.title
        include_centroid_size = options.include_centroid_size
        include_mean_coords = options.include_mean_coords
        include_gpa_coords = options.include_gpa_coords
        include_edma_distances = options.include_edma_distances

        num_species = len(self.specimen_ids)
        num_characters = 0

        if include_centroid_size:
            num_characters += 1

        if include_mean_coords or include_gpa_coords:
            num_characters += 1  # Configurations will be treated as one character

        if include_edma_distances:
            if self.selected_landmarks:
                num_distances = len(self.calculate_edma_distances(self.landmarks[0], self.selected_landmarks))
            else:
                num_landmarks = len(self.landmarks[0])
                num_distances = (num_landmarks * (num_landmarks - 1)) // 2
            num_characters += num_distances

        save_tnt_path = filedialog.asksaveasfilename(defaultextension=".tnt", filetypes=[("TNT Files", "*.tnt"), ("All Files", "*.*")])
        if not save_tnt_path:
            return

        with open(save_tnt_path, 'w') as file:
            file.write("nstates cont;\n")
            file.write("nstates 32;\n")
            file.write(f"xread '{title}'\n")
            file.write(f"{num_characters} {num_species}\n")

            # 1. Export CSize only if it was selected
            if include_centroid_size:
                file.write("&[cont]\n")
                for specimen_id, landmarks in zip(self.specimen_ids, self.landmarks):
                    # CSize mean calculation
                    size = self.calculate_centroid_size(landmarks)
                    
                    if self.multi_specimen:
                        # If there are multiple specimens, calculate the range (mean ± SE or CI)
                        centroid_sizes = [self.calculate_centroid_size(lmk) for lmk in self.landmarks]
                        if self.interval_type.get() == "CI":
                            # Calculate CI
                            res = bootstrap((centroid_sizes,), np.mean, confidence_level=self.confidence_level.get())
                            size_interval = f"{res.confidence_interval.low:.6f}-{res.confidence_interval.high:.6f}"
                        else:
                            # Calculate mean ± SE
                            mean_value = np.mean(centroid_sizes)
                            se_value = sem(centroid_sizes)
                            size_interval = f"{mean_value-se_value:.6f}-{mean_value+se_value:.6f}"
                        file.write(f"{specimen_id.replace(' ', '_')} {size_interval}\n")
                    else:
                        # If there is only one specimen, save the centroid size directly
                        file.write(f"{specimen_id.replace(' ', '_')} {size:.6f}\n")
                file.write("\n")

            # 2. Export EDMA distances if selected
            if include_edma_distances:
                file.write("&[cont]\n")
                for specimen_id in self.specimen_ids:
                    specimen_id = specimen_id.replace(" ", "_")
                    distances = self.calculate_edma_distances(self.landmarks[0], self.selected_landmarks)
                    if self.multi_specimen:
                        # Calculate intervals if there are multiple specimens
                        distance_intervals = self.calculate_intervals(self.landmarks, self.interval_type.get(), self.confidence_level.get(), self.selected_landmarks)
                        file.write(f"{specimen_id} {' '.join(distance_intervals)}\n")
                    else:
                        # Output distances directly if there is only one specimen per species
                        distances_str = " ".join([f"{dist:.6f}" for dist in distances])
                        file.write(f"{specimen_id} {distances_str}\n")
                file.write("\n")

            # 3. Export settings (average or GPA) if selected
            if include_mean_coords or include_gpa_coords:
                file.write("&[landmark 2D]\n")
                if include_mean_coords:
                    for specimen_id, landmarks in zip(self.specimen_ids, self.landmarks):
                        specimen_id = specimen_id.replace(" ", "_")
                        coords = " ".join([f"{x:.6f},{y:.6f}" for x, y in landmarks])
                        file.write(f"{specimen_id} {coords}\n")
                elif include_gpa_coords:
                    for specimen_id, landmarks in zip(self.specimen_ids, self.aligned_landmarks):
                        specimen_id = specimen_id.replace(" ", "_")
                        coords = " ".join([f"{x:.6f},{y:.6f}" for x, y in landmarks])
                        file.write(f"{specimen_id} {coords}\n")
                file.write("\n")

            file.write(";\n")
            file.write("proc/;\n")

        messagebox.showinfo("Export Completed", "TNT file exported successfully.")

class ExportOptionsDialog:
    def __init__(self, parent, interval_type, confidence_level):
        top = self.top = tk.Toplevel(parent)
        top.title("Export Options")

        ttk.Label(top, text="Title:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.title_entry = ttk.Entry(top)
        self.title_entry.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)

        self.include_centroid_size = tk.BooleanVar()
        ttk.Checkbutton(top, text="Include Centroid Size", variable=self.include_centroid_size).grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        self.include_mean_coords = tk.BooleanVar()
        ttk.Checkbutton(top, text="Include Mean Coordinates", variable=self.include_mean_coords).grid(row=2, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        self.include_gpa_coords = tk.BooleanVar()
        ttk.Checkbutton(top, text="Include GPA Coordinates", variable=self.include_gpa_coords).grid(row=3, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        self.include_edma_distances = tk.BooleanVar()
        ttk.Checkbutton(top, text="Include EDMA Distances", variable=self.include_edma_distances).grid(row=4, column=0, columnspan=2, padx=5, pady=5, sticky=tk.W)

        self.interval_type = interval_type
        self.confidence_level = confidence_level

        self.ok = False

        ttk.Button(top, text="OK", command=self.on_ok).grid(row=5, column=0, padx=5, pady=5)
        ttk.Button(top, text="Cancel", command=self.on_cancel).grid(row=5, column=1, padx=5, pady=5)

    def on_ok(self):
        self.ok = True
        self.title = self.title_entry.get()
        self.include_centroid_size = self.include_centroid_size.get()
        self.include_mean_coords = self.include_mean_coords.get()
        self.include_gpa_coords = self.include_gpa_coords.get()
        self.include_edma_distances = self.include_edma_distances.get()
        self.top.destroy()

    def on_cancel(self):
        self.top.destroy()

if __name__ == "__main__":    
    try:
        root = tk.Tk()
        app = py_tps2tntApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Error: {e}")
        input("Press Enter to exit...")
