import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
import numpy as np
import random
import datetime
import io

# --- Configuration ---
# If automatic delimiter detection is inaccurate, or you want to force a specific delimiter for all input files, set it here.
# Set to None to rely entirely on automatic detection.
FORCE_INPUT_DELIMITER = None 

# --- Helper Functions ---
def get_script_directory():
    """Returns the absolute path to the directory where the current script is located."""
    return os.path.dirname(os.path.abspath(__file__))

def find_all_csv_tsv_files(directory):
    """
    Finds all .csv and .tsv files in the specified directory.
    """
    file_paths = []
    if not os.path.exists(directory):
        return []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(('.csv', '.tsv')):
            file_paths.append(file_path)
    return file_paths

def determine_delimiter(file_path, first_line_content):
    """
    Intelligently determines the delimiter of a file based on its first line content.
    If FORCE_INPUT_DELIMITER is set in configuration, it takes precedence.
    """
    if FORCE_INPUT_DELIMITER:
        return FORCE_INPUT_DELIMITER

    num_commas = first_line_content.count(',')
    num_tabs = first_line_content.count('\t')

    # Heuristic: If tabs are more frequent and present, prioritize tab.
    if num_tabs > num_commas and num_tabs > 0:
        return '\t'
    # Heuristic: If commas are more frequent and present, prioritize comma.
    elif num_commas > num_tabs and num_commas > 0:
        return ','
    # If counts are similar or zero, guess based on file extension.
    elif file_path.lower().endswith('.tsv'):
        return '\t'
    elif file_path.lower().endswith('.csv'):
        return ','
    
    # Fallback to comma, as it's a common default for CSV.
    return ','

# --- Core Logic for Analysis ---
def load_and_categorize_data(file_paths):
    """
    Loads data from multiple CSV/TSV files and categorizes it as connection, coordinate,
    frame cross-section definition, and plate cross-section definition data,
    grouped by (filename, Dataset_ID).
    
    Returns:
        tuple: (all_conn_data, all_coord_data, all_frame_def_data, all_plate_def_data)
        Each dictionary maps (filename, dataset_id) to a pandas.DataFrame.
    """
    all_conn_data = {}
    all_coord_data = {}
    all_frame_def_data = {} 
    all_plate_def_data = {} 
    
    print("\n" + "="*70)
    print("Loading and categorizing data from files...")
    print("="*70)

    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        print(f"Loading '{file_name}'...")
        try:
            with open(file_path, 'r', newline='', encoding='utf-8') as f:
                raw_content = f.read()
            
            if not raw_content.strip():
                print(f"  Warning: File '{file_name}' is empty. Skipping.")
                continue
            
            first_line = raw_content.splitlines()[0]
            delimiter = determine_delimiter(file_path, first_line)
            
            # Use pandas to read CSV/TSV data
            df = pd.read_csv(io.StringIO(raw_content), delimiter=delimiter)
            
            # Clean up leading/trailing spaces in column names
            df.columns = df.columns.str.strip()
            
            if 'Dataset_ID' not in df.columns:
                print(f"  Warning: File '{file_name}' does not have a 'Dataset_ID' column. Skipping grouping.")
                continue

            # Determine file type based on key column names
            is_connections = all(col in df.columns for col in ['ElemID', 'StartNodeID', 'EndNodeID'])
            is_coordinates = all(col in df.columns for col in ['NodeID', 'X', 'Y', 'Z'])
            is_frame_def = all(col in df.columns for col in ['family', 'height_cm', 'Flange_width_cm', 'Area_cm2'])
            is_plate_def = all(col in df.columns for col in ['family', 'face1_height_cm']) # face2_height_cm might not exist or be empty

            # Group by Dataset_ID within the file
            for dataset_id, group_df in df.groupby('Dataset_ID'):
                key = (file_name, dataset_id) # Unique identifier for each structure/group
                
                if is_connections:
                    all_conn_data[key] = group_df
                    print(f"  -> Identified as Connection Data (Connections) for {key}, with {len(group_df)} rows.")
                    print("  Data Preview:")
                    print(group_df.head())
                    print("-" * 20)
                elif is_coordinates:
                    all_coord_data[key] = group_df
                    print(f"  -> Identified as Coordinate Data (Coordinates) for {key}, with {len(group_df)} rows.")
                    print("  Data Preview:")
                    print(group_df.head())
                    print("-" * 20)
                elif is_frame_def:
                    # Convert relevant columns to numeric, coercing errors to NaN
                    for col in ['height_cm', 'Flange_width_cm', 'Area_cm2']:
                        if col in group_df.columns:
                            group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
                    all_frame_def_data[key] = group_df
                    print(f"  -> Identified as FrameCrossection Definitions for {key}, with {len(group_df)} rows.")
                    print("  Data Preview:")
                    print(group_df.head())
                    print("-" * 20)
                elif is_plate_def:
                     # Convert relevant columns to numeric, coercing errors to NaN
                    for col in ['face1_height_cm', 'face2_height_cm']:
                        if col in group_df.columns:
                            group_df[col] = pd.to_numeric(group_df[col], errors='coerce')
                    all_plate_def_data[key] = group_df
                    print(f"  -> Identified as PlateCrossection Definitions for {key}, with {len(group_df)} rows.")
                    print("  Data Preview:")
                    print(group_df.head())
                    print("-" * 20)
                else:
                    print(f"  Warning: Unable to clearly categorize data for Dataset_ID={dataset_id} in file '{file_name}'. Skipping.")
                    
        except Exception as e:
            print(f"  Error: Failed to load or parse file '{file_name}': {e}. Skipping.")
            
    return all_conn_data, all_coord_data, all_frame_def_data, all_plate_def_data

def analyze_overall_structural_height(all_coord_data):
    """
    Collects the maximum Z-value (structural height) for each Dataset_ID from all coordinate data,
    and plots a histogram of the overall height distribution across all structures.
    """
    print("\n" + "="*70)
    print("Analyzing overall structural height distribution...")
    print("="*70)

    all_structural_heights = [] 
    
    if not all_coord_data:
        print("No coordinate data found for overall structural height analysis.")
        return

    # Iterate through all loaded coordinate data, regardless of file
    for key, coord_df in all_coord_data.items():
        file_name, dataset_id = key
        structural_height = np.nan
        if 'Z' in coord_df.columns and not coord_df['Z'].empty:
            # Ensure Z column is numeric
            z_values = pd.to_numeric(coord_df['Z'], errors='coerce').dropna()
            if not z_values.empty:
                structural_height = z_values.max()
                all_structural_heights.append(structural_height)

    if all_structural_heights:
        plt.figure(figsize=(8, 6))
        # Adjust bins for better visualization: use at least 5 bins or half the number of unique height values
        bins = max(5, len(set(all_structural_heights)) // 2) if len(set(all_structural_heights)) > 1 else 1
        plt.hist(all_structural_heights, bins=bins, edgecolor='black') 
        plt.title('Overall Structural Height Distribution')
        plt.xlabel('Structural Height (Max Z Value)')
        plt.ylabel('Number of Structures')
        plt.grid(axis='y', alpha=0.75)
        plt.show()
    else:
        print("\n  No valid structural height data found across all structures to plot overall distribution.")


def analyze_crossection_definitions(all_frame_def_data, all_plate_def_data):
    """
    Analyzes FrameCrossection definition data for height distribution and outliers.
    """
    print("\n" + "="*70)
    print("Analyzing FrameCrossection and PlateCrossection Definition Data...")
    print("="*70)

    # --- 1. Collect FrameCrossection data for distribution analysis ---
    all_col_max_heights = []
    all_col_mean_heights = []
    all_beam_max_heights = []
    all_beam_mean_heights = []
    
    if not all_frame_def_data:
        print("No FrameCrossection definition data found for analysis.")
    else:
        print("\n--- Collecting FrameCrossection statistics per group ---")
        for key, df in all_frame_def_data.items():
            file_name, dataset_id = key
            
            # ColumnSet analysis
            col_df = df[df['family'] == 'ColumnSet']
            if not col_df.empty and 'height_cm' in col_df.columns:
                col_heights = col_df['height_cm'].dropna()
                if not col_heights.empty:
                    all_col_max_heights.append(col_heights.max())
                    all_col_mean_heights.append(col_heights.mean())

            # BeamSet analysis
            beam_df = df[df['family'] == 'BeamSet']
            if not beam_df.empty and 'height_cm' in beam_df.columns:
                beam_heights = beam_df['height_cm'].dropna()
                if not beam_heights.empty:
                    all_beam_max_heights.append(beam_heights.max())
                    all_beam_mean_heights.append(beam_heights.mean())

    # --- 2. Plot FrameCrossection distributions (4 plots) ---
    if all_col_max_heights or all_col_mean_heights or all_beam_max_heights or all_beam_mean_heights:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('FrameCrossection Definitions: Height Distribution Across Structures')
        
        # ColumnSet Max Height Distribution
        if all_col_max_heights:
            bins = max(5, len(set(all_col_max_heights)) // 2) if len(set(all_col_max_heights)) > 1 else 1
            axes[0, 0].hist(all_col_max_heights, bins=bins, edgecolor='black')
            axes[0, 0].set_title('ColumnSet Max Height Distribution')
            axes[0, 0].set_xlabel('Max Height (cm)')
            axes[0, 0].set_ylabel('Number of Structures')
        else:
            axes[0, 0].text(0.5, 0.5, 'No ColumnSet Max Height Data', horizontalalignment='center', verticalalignment='center', transform=axes[0, 0].transAxes)
            axes[0, 0].set_title('ColumnSet Max Height Distribution')

        # ColumnSet Mean Height Distribution
        if all_col_mean_heights:
            bins = max(5, len(set(all_col_mean_heights)) // 2) if len(set(all_col_mean_heights)) > 1 else 1
            axes[0, 1].hist(all_col_mean_heights, bins=bins, edgecolor='black')
            axes[0, 1].set_title('ColumnSet Mean Height Distribution')
            axes[0, 1].set_xlabel('Mean Height (cm)')
            axes[0, 1].set_ylabel('Number of Structures')
        else:
            axes[0, 1].text(0.5, 0.5, 'No ColumnSet Mean Height Data', horizontalalignment='center', verticalalignment='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('ColumnSet Mean Height Distribution')

        # BeamSet Max Height Distribution
        if all_beam_max_heights:
            bins = max(5, len(set(all_beam_max_heights)) // 2) if len(set(all_beam_max_heights)) > 1 else 1
            axes[1, 0].hist(all_beam_max_heights, bins=bins, edgecolor='black')
            axes[1, 0].set_title('BeamSet Max Height Distribution')
            axes[1, 0].set_xlabel('Max Height (cm)')
            axes[1, 0].set_ylabel('Number of Structures')
        else:
            axes[1, 0].text(0.5, 0.5, 'No BeamSet Max Height Data', horizontalalignment='center', verticalalignment='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('BeamSet Max Height Distribution')

        # BeamSet Mean Height Distribution
        if all_beam_mean_heights:
            bins = max(5, len(set(all_beam_mean_heights)) // 2) if len(set(all_beam_mean_heights)) > 1 else 1
            axes[1, 1].hist(all_beam_mean_heights, bins=bins, edgecolor='black')
            axes[1, 1].set_title('BeamSet Mean Height Distribution')
            axes[1, 1].set_xlabel('Mean Height (cm)')
            axes[1, 1].set_ylabel('Number of Structures')
        else:
            axes[1, 1].text(0.5, 0.5, 'No BeamSet Mean Height Data', horizontalalignment='center', verticalalignment='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('BeamSet Mean Height Distribution')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
    else:
        print("No sufficient FrameCrossection data collected to plot distributions.")

    # --- 3. PlateCrossection Outlier Identification ---
    print("\n--- PlateCrossection Outlier Identification (based on face1_height_cm) ---")
    if not all_plate_def_data:
        print("No PlateCrossection definition data found for outlier analysis.")
        return
    
    found_plate_outliers = False
    for key, df in all_plate_def_data.items():
        file_name, dataset_id = key
        
        if 'face1_height_cm' not in df.columns:
            print(f"  {key}: 'face1_height_cm' column not found for outlier analysis.")
            continue

        valid_heights = df[['family', 'face1_height_cm']].dropna(subset=['face1_height_cm'])
        if valid_heights.empty:
            print(f"  {key}: No valid 'face1_height_cm' values found for outlier analysis.")
            continue

        value_counts = valid_heights['face1_height_cm'].value_counts()
        if value_counts.empty:
            print(f"  {key}: Unable to determine majority/outlier (no data).")
            continue

        # The most frequent 'face1_height_cm' value is considered the majority
        majority_height = value_counts.index[0]
        
        # All entries where 'face1_height_cm' is not equal to the majority height are outliers
        outliers_df = valid_heights[valid_heights['face1_height_cm'] != majority_height]

        print(f"\n  For Structure: {key}")
        print(f"    Majority 'face1_height_cm' value: {majority_height} cm (occurred {value_counts.iloc[0]} times)")
        
        if not outliers_df.empty:
            found_plate_outliers = True
            print("    Outlier Data Points ('face1_height_cm' not equal to majority value):")
            for _, row in outliers_df.iterrows():
                print(f"      - Family: {row['family']}, Height: {row['face1_height_cm']} cm")
        else:
            print("    No anomalous 'face1_height_cm' values found (all 'face1_height_cm' values are the same as the majority).")

    if not found_plate_outliers:
        print("\nNo 'face1_height_cm' outliers detected in any PlateCrossection data (based on majority value principle).")

def analyze_connection_anomalies(all_conn_data, all_coord_data):
    """
    Identifies and reports potential anomalous connections in the data
    (e.g., missing start/end node coordinates, or excessively short/long connection lengths).
    """
    print("\n" + "="*70)
    print("Analyzing connection anomalies...")
    print("="*70)

    if not all_conn_data or not all_coord_data:
        print("No connection or coordinate data found, unable to analyze connection anomalies.")
        return

    # Group connection and coordinate data by Dataset_ID to match across files
    conn_by_dataset_id = {}
    for key_tuple, df in all_conn_data.items():
        conn_by_dataset_id.setdefault(key_tuple[1], []).append((key_tuple[0], df)) # Store (filename, df)

    coord_by_dataset_id = {}
    for key_tuple, df in all_coord_data.items():
        coord_by_dataset_id.setdefault(key_tuple[1], []).append((key_tuple[0], df)) # Store (filename, df)

    # Find Dataset_ID's that have both connection and coordinate data
    common_dataset_ids = sorted(list(set(conn_by_dataset_id.keys()) & set(coord_by_dataset_id.keys())))

    if not common_dataset_ids:
        print("No complete structures found (Dataset_ID matched across connections and coordinates) to analyze connection anomalies.")
        return

    # These thresholds are examples; you may need to adjust them based on your actual data
    min_length_threshold = 0.01  # Considered zero-length or extremely short connection (e.g., less than 1 cm)
    max_length_threshold = 1000.0 # Considered excessively long connection (e.g., over 1000 meters)

    total_anomalies_found = False

    for ds_id in common_dataset_ids:
        # For simplicity, pick the first found connection and coordinate data for this dataset_id
        conn_filename, conn_df = conn_by_dataset_id[ds_id][0]
        coord_filename, coord_df = coord_by_dataset_id[ds_id][0]

        display_info = f"Dataset_ID {ds_id} (Connections from: '{conn_filename}', Coordinates from: '{coord_filename}')"

        anomalies_for_this_structure = False
        print(f"\n  Analyzing structure: {display_info}")

        # Map NodeID to coordinates for quick lookup
        # Ensure X, Y, Z columns are numeric
        coord_df_numeric = coord_df.copy()
        for col in ['X', 'Y', 'Z']:
            if col in coord_df_numeric.columns:
                coord_df_numeric[col] = pd.to_numeric(coord_df_numeric[col], errors='coerce')

        node_coords = coord_df_numeric.set_index('NodeID')[['X', 'Y', 'Z']].to_dict('index')

        for idx, row in conn_df.iterrows():
            elem_id = row.get('ElemID', f'Idx_{idx}') # Fallback for ElemID if not present
            start_node_id = row['StartNodeID']
            end_node_id = row['EndNodeID']

            start_coords = node_coords.get(start_node_id)
            end_coords = node_coords.get(end_node_id)

            # 1. Missing or invalid coordinates for start/end node
            if start_coords is None or any(pd.isna(v) for v in start_coords.values()):
                print(f"    Anomaly: Element {elem_id} (Nodes {start_node_id} -> {end_node_id}) has missing or invalid coordinates for start node {start_node_id}.")
                anomalies_for_this_structure = True
            if end_coords is None or any(pd.isna(v) for v in end_coords.values()):
                print(f"    Anomaly: Element {elem_id} (Nodes {start_node_id} -> {end_node_id}) has missing or invalid coordinates for end node {end_node_id}.")
                anomalies_for_this_structure = True
            
            if (start_coords is not None and all(not pd.isna(v) for v in start_coords.values())) and \
               (end_coords is not None and all(not pd.isna(v) for v in end_coords.values())):
                # 2. Extremely short/long connection length
                try:
                    p1 = np.array([start_coords['X'], start_coords['Y'], start_coords['Z']])
                    p2 = np.array([end_coords['X'], end_coords['Y'], end_coords['Z']])
                    length = np.linalg.norm(p2 - p1)

                    if length < min_length_threshold:
                        print(f"    Anomaly: Element {elem_id} (Nodes {start_node_id} -> {end_node_id}) has an extremely short length ({length:.4f}), possibly a zero-length connection.")
                        anomalies_for_this_structure = True
                    elif length > max_length_threshold:
                        print(f"    Anomaly: Element {elem_id} (Nodes {start_node_id} -> {end_node_id}) has an excessively long length ({length:.2f}), possibly data error.")
                        anomalies_for_this_structure = True
                except Exception as e:
                    print(f"    Warning: Error calculating length for element {elem_id} (Nodes {start_node_id} -> {end_node_id}): {e}")

        if not anomalies_for_this_structure:
            print(f"    No obvious connection anomalies found in this structure.")
        else:
            total_anomalies_found = True

    if not total_anomalies_found:
        print("\nNo connection anomalies found across all structures.")


def reconstruct_and_plot_structures(all_conn_data, all_coord_data, num_specific_ids=3):
    """
    Plots 3D structures for Dataset_ID 0, 1, 2 (if they exist).
    """
    print("\n" + "="*70)
    print(f"Reconstructing and plotting 3D structures for Dataset_ID 0, 1, 2 (up to {num_specific_ids} instances)...")
    print("="*70)

    # Group connection and coordinate data by Dataset_ID to match across files
    conn_by_dataset_id = {}
    for key_tuple, df in all_conn_data.items():
        conn_by_dataset_id.setdefault(key_tuple[1], []).append((key_tuple[0], df)) # Store (filename, df)

    coord_by_dataset_id = {}
    for key_tuple, df in all_coord_data.items():
        coord_by_dataset_id.setdefault(key_tuple[1], []).append((key_tuple[0], df)) # Store (filename, df)

    # Find Dataset_ID's that have both connection and coordinate data
    common_dataset_ids = sorted(list(set(conn_by_dataset_id.keys()) & set(coord_by_dataset_id.keys())))
    
    specific_dataset_ids_to_plot = []
    target_dataset_ids = [0, 1, 2] # The Dataset_IDs you explicitly asked to plot

    # Select target Dataset_IDs from the common ones, limiting by num_specific_ids
    for ds_id in common_dataset_ids:
        if ds_id in target_dataset_ids and len(specific_dataset_ids_to_plot) < num_specific_ids:
            specific_dataset_ids_to_plot.append(ds_id)
    
    if not specific_dataset_ids_to_plot:
        print(f"Warning: No complete structures found with Dataset_ID {target_dataset_ids} (having both connections and coordinates) for plotting.")
        return

    print(f"Found {len(specific_dataset_ids_to_plot)} complete structure instances with Dataset_ID {specific_dataset_ids_to_plot} for plotting.")

    for ds_id in specific_dataset_ids_to_plot:
        # For plotting, we typically need one set of connections and one set of coordinates for a given Dataset_ID.
        # If multiple files contain data for the same Dataset_ID, we'll use the first one found.
        conn_filename, conn_df = conn_by_dataset_id[ds_id][0]
        coord_filename, coord_df = coord_by_dataset_id[ds_id][0]

        print(f"  Plotting structure: Dataset_ID {ds_id} (Connections from '{conn_filename}', Coordinates from '{coord_filename}')...")

        # Basic validation for essential columns required for plotting
        if not all(col in coord_df.columns for col in ['X', 'Y', 'Z', 'NodeID']):
            print(f"    Warning: Missing X, Y, Z, or NodeID columns in coordinate data, unable to plot for Dataset_ID {ds_id}. Skipping plot.")
            continue
        if not all(col in conn_df.columns for col in ['StartNodeID', 'EndNodeID']):
            print(f"    Warning: Missing StartNodeID or EndNodeID columns in connection data, unable to plot for Dataset_ID {ds_id}. Skipping plot.")
            continue

        # Map NodeID to coordinates for quick lookup
        # Ensure X, Y, Z columns are numeric
        coord_df_numeric = coord_df.copy()
        for col in ['X', 'Y', 'Z']:
            if col in coord_df_numeric.columns:
                coord_df_numeric[col] = pd.to_numeric(coord_df_numeric[col], errors='coerce')
        node_coords = coord_df_numeric.set_index('NodeID')[['X', 'Y', 'Z']].to_dict('index')

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'3D Structure: Dataset_ID {ds_id} (Conn: {conn_filename}, Coords: {coord_filename})')

        # Plot nodes as scatter points
        nodes_x = []
        nodes_y = []
        nodes_z = []
        for node_id, coords in node_coords.items():
            if all(not pd.isna(v) for v in coords.values()): # Only plot nodes with valid coordinates
                nodes_x.append(coords['X'])
                nodes_y.append(coords['Y'])
                nodes_z.append(coords['Z'])
        if nodes_x: # Only scatter if there are valid nodes
            ax.scatter(nodes_x, nodes_y, nodes_z, c='blue', marker='o', s=20, label='Nodes')

        # Plot elements as lines connecting nodes
        for _, row in conn_df.iterrows():
            start_node_id = row['StartNodeID']
            end_node_id = row['EndNodeID']

            start_coords = node_coords.get(start_node_id)
            end_coords = node_coords.get(end_node_id)
            
            if (start_coords is not None and all(not pd.isna(v) for v in start_coords.values())) and \
               (end_coords is not None and all(not pd.isna(v) for v in end_coords.values())):
                ax.plot([start_coords['X'], end_coords['X']],
                        [start_coords['Y'], end_coords['Y']],
                        [start_coords['Z'], end_coords['Z']], 'k-', linewidth=1) # Black solid line for elements
            # else: # Uncomment for more detailed debugging if elements cannot be plotted
            #     print(f"      Warning: Node {start_node_id} or {end_node_id} not found in coordinates for Dataset_ID {ds_id}. Skipping element.")

        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate')
        ax.legend()
        plt.show()

# --- Main Execution Flow ---
def main():
    script_dir = get_script_directory()
    data_source_dir = script_dir 

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting structural data analysis and visualization...")
    print(f"Script directory: {script_dir}")
    print(f"Data file source directory: '{data_source_dir}' (Expected to be the same as script directory)")
    print("-" * 70)

    # 1. Find all CSV/TSV files in the script directory
    data_file_paths = find_all_csv_tsv_files(data_source_dir)

    if not data_file_paths:
        print(f"Error: No .csv or .tsv files found directly in '{data_source_dir}'.")
        print(f"Please ensure your data files (connections, coordinates, cross-section definitions) are in this folder (same as this script).")
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script finished.")
        return

    print(f"Detected {len(data_file_paths)} files in '{data_source_dir}' for analysis:")
    for p in data_file_paths:
        print(f"  - {os.path.basename(p)}")

    # 2. Load and categorize data from these files
    all_conn_data, all_coord_data, all_frame_def_data, all_plate_def_data = load_and_categorize_data(data_file_paths)
    
    # Check if any data was loaded
    if not (all_conn_data or all_coord_data or all_frame_def_data or all_plate_def_data):
        print("\nError: No usable data found after loading files. Exiting.")
        return

    # 3. Analyze FrameCrossection and PlateCrossection definition data
    analyze_crossection_definitions(all_frame_def_data, all_plate_def_data) 

    # 4. Analyze overall structural height distribution (macro information from coordinate data)
    analyze_overall_structural_height(all_coord_data)

    # 5. Analyze connection anomalies
    analyze_connection_anomalies(all_conn_data, all_coord_data)

    # 6. Reconstruct and plot 3D structures for specific Dataset_IDs (0, 1, 2)
    # Check if there are any complete structures that can be plotted (connections and coordinates for the same Dataset_ID)
    if all_conn_data and all_coord_data:
        # Group connection and coordinate data by Dataset_ID to find common IDs
        conn_by_dataset_id = {}
        for key_tuple, df in all_conn_data.items():
            conn_by_dataset_id.setdefault(key_tuple[1], []).append((key_tuple[0], df))

        coord_by_dataset_id = {}
        for key_tuple, df in all_coord_data.items():
            coord_by_dataset_id.setdefault(key_tuple[1], []).append((key_tuple[0], df))

        common_dataset_ids = sorted(list(set(conn_by_dataset_id.keys()) & set(coord_by_dataset_id.keys())))

        if common_dataset_ids:
            print(f"\nFound {len(common_dataset_ids)} complete structures (identified by Dataset_ID) for 3D plotting.")
            reconstruct_and_plot_structures(all_conn_data, all_coord_data, num_specific_ids=3) 
        else:
            print("\nWarning: No complete structures found (with both connection and coordinate data for the same Dataset_ID) for 3D plotting.")
    else:
        print("\nWarning: Missing connection or coordinate data (or both), unable to perform 3D plotting.")


    print("\n" + "=" * 70)
    print("Analysis and visualization complete.")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script finished.")

if __name__ == "__main__":
    main()
