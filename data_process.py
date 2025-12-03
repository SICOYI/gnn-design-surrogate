import os
import csv
import datetime
import io

# --- Configuration ---
# Output files will be saved in this subfolder within the script's directory.
OUTPUT_SUBFOLDER_NAME = "processed_data" 

# If automatic delimiter detection is inaccurate, or if you want to force a specific delimiter
# for all input files, set it here. Set to None to rely solely on auto-detection.
# Common options: '\t' (tab), ',' (comma)
# Example: FORCE_INPUT_DELIMITER = '\t'
FORCE_INPUT_DELIMITER = None 

# By default, output files will use the same delimiter as detected for the input file.
# If you want to force all output files to use a specific delimiter, set it here.
# Example: OUTPUT_OVERRIDE_DELIMITER = ','
OUTPUT_OVERRIDE_DELIMITER = None 

# --- Helper Functions ---
def get_script_directory():
    """Returns the absolute path of the directory containing the current script."""
    return os.path.dirname(os.path.abspath(__file__))

def find_input_files(directory, exclude_subfolder_name=None):
    """
    Finds all .csv and .tsv files in the specified directory, excluding files
    within a specific subfolder (e.g., the output directory).
    """
    file_paths = []
    if not os.path.exists(directory):
        return []

    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        # Exclude the output subfolder itself and files within it.
        if exclude_subfolder_name and (file_path == os.path.join(directory, exclude_subfolder_name) or \
                                       file_path.startswith(os.path.join(directory, exclude_subfolder_name + os.sep))):
            continue

        if os.path.isfile(file_path):
            if filename.lower().endswith(('.csv', '.tsv')):
                file_paths.append(file_path)
    return file_paths

def determine_delimiter(file_path, first_line_content):
    """
    Attempts to intelligently determine the file's delimiter based on its first line.
    If FORCE_INPUT_DELIMITER is set in configuration, it takes precedence.
    """
    if FORCE_INPUT_DELIMITER:
        return FORCE_INPUT_DELIMITER

    num_commas = first_line_content.count(',')
    num_tabs = first_line_content.count('\t')

    # Heuristic: Prioritize tab if it's more frequent and present.
    if num_tabs > num_commas and num_tabs > 0:
        return '\t'
    # Heuristic: Prioritize comma if it's more frequent and present.
    elif num_commas > num_tabs and num_commas > 0:
        return ','
    # If counts are similar or zero, guess based on file extension.
    elif file_path.lower().endswith('.tsv'):
        return '\t'
    elif file_path.lower().endswith('.csv'):
        return ','
    
    # Fallback to comma as it's a common default for CSVs.
    return ','

# --- Core Logic: Independent File Processor Function ---
def process_single_file(file_path, output_dir):
    """
    Processes a single CSV/TSV file, remapping its 'Dataset_ID' column
    to start from 0, local to this file, based on the *order of first appearance*.
    
    Args:
        file_path (str): The full path to the input file.
        output_dir (str): The directory where the processed file should be saved.
        
    Returns:
        tuple: A tuple containing (list of log messages, boolean indicating success).
    """
    logs = []
    file_name = os.path.basename(file_path)
    output_file_path = os.path.join(output_dir, file_name)
    
    # Initialize variables for this file's processing
    # --- IMPORTANT CHANGE START ---
    # This list will store unique Dataset_IDs in the order they are first encountered.
    unique_ids_in_order = [] 
    # This set is used for efficient lookup to check if an ID has already been added to unique_ids_in_order.
    seen_ids_for_lookup = set()
    # --- IMPORTANT CHANGE END ---
    
    local_dataset_id_mapping = None # Will store the final mapping for this file
    processed_rows = [] # Stores rows for the output file
    
    logs.append(f"--- Processing file: '{file_name}' ---") # Mark start of processing for this file
    
    try:
        # 1. Read the entire file content into memory. This allows for multiple passes
        # (e.g., delimiter detection, then actual data processing) without re-opening.
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            raw_file_content_string = f.read()
            if not raw_file_content_string.strip(): # Check if file is empty or only whitespace.
                logs.append(f"WARNING: File '{file_name}' is empty or contains only whitespace. Skipping this file.")
                return logs, False
        
        # Use io.StringIO to simulate a file object from the string content.
        # First pass: Determine delimiter and collect local Dataset_IDs.
        string_buffer_for_detection = io.StringIO(raw_file_content_string)
        first_line_content = string_buffer_for_detection.readline().strip()
        
        current_file_delimiter = determine_delimiter(file_path, first_line_content)
        logs.append(f"INFO: File '{file_name}' detected delimiter '{current_file_delimiter}'.")
        
        # Reset buffer and parse with the determined delimiter to find header and Dataset_IDs.
        string_buffer_for_detection.seek(0)
        reader_for_id_collection = csv.reader(string_buffer_for_detection, delimiter=current_file_delimiter)
        
        header = next(reader_for_id_collection, None)
        
        if not header:
            logs.append(f"ERROR: File '{file_name}' has no header row. Skipping this file.")
            return logs, False

        try:
            dataset_id_col_index = header.index('Dataset_ID')
            logs.append(f"INFO: Header for '{file_name}': {header}")
        except ValueError:
            logs.append(f"ERROR: File '{file_name}' does not contain a 'Dataset_ID' column. Skipping this file.")
            logs.append(f"    Detected Delimiter: '{current_file_delimiter}'")
            logs.append(f"    Raw First Line: '{first_line_content}'")
            logs.append(f"    Parsed Header List: {header}")
            return logs, False
        
        # --- IMPORTANT CHANGE START ---
        # Collect unique Dataset_IDs for THIS file only, in order of their first appearance.
        for row_idx, row in enumerate(reader_for_id_collection):
            if row and len(row) > dataset_id_col_index:
                try:
                    raw_id = int(row[dataset_id_col_index])
                    if raw_id not in seen_ids_for_lookup: # Check if this ID has been seen before
                        unique_ids_in_order.append(raw_id) # Add to ordered list if new
                        seen_ids_for_lookup.add(raw_id) # Add to set for quick lookup
                except ValueError:
                    logs.append(f"WARNING: File '{file_name}', row {row_idx+2}: 'Dataset_ID' value '{row[dataset_id_col_index]}' cannot be parsed as an integer. This row's ID is skipped for mapping collection.")
            elif row: 
                logs.append(f"WARNING: File '{file_name}', row {row_idx+2}: Row is too short and does not contain 'Dataset_ID'. This row's ID is skipped for mapping collection.")
        # --- IMPORTANT CHANGE END ---

        # --- Debugging Logs for ID Collection ---
        if not unique_ids_in_order: # Check the new ordered list
            logs.append(f"WARNING: File '{file_name}' did not contain any valid integer 'Dataset_ID's. Skipping this file.")
            return logs, False
        
        logs.append(f"DEBUG: Collected unique Dataset_IDs in order of appearance for '{file_name}': {unique_ids_in_order}")

        # Create a local mapping for this file's Dataset_IDs.
        # This mapping *always* starts from 0 for *this specific file*,
        # and uses the order from unique_ids_in_order.
        local_dataset_id_mapping = {original_id: new_id for new_id, original_id in enumerate(unique_ids_in_order)}
        
        logs.append(f"DEBUG: Generated ID map for '{file_name}': {local_dataset_id_mapping}")
        logs.append(f"  Local Dataset_ID mapping for '{file_name}': {len(local_dataset_id_mapping)} unique IDs found. First encountered unique ID '{unique_ids_in_order[0]}' will be mapped to '0'.")
        
        # Second pass: Apply mapping and write to new file.
        output_delimiter_for_this_file = OUTPUT_OVERRIDE_DELIMITER if OUTPUT_OVERRIDE_DELIMITER else current_file_delimiter
        
        # Reset buffer and reader to process from the beginning for mapping application.
        string_buffer_for_mapping = io.StringIO(raw_file_content_string)
        reader_for_mapping = csv.reader(string_buffer_for_mapping, delimiter=current_file_delimiter)

        header_for_mapping = next(reader_for_mapping) # Read header again to ensure consistent processing
        processed_rows.append(header_for_mapping) # Add original header to processed data.

        # Re-get index to be safe, though it should be the same as dataset_id_col_index.
        dataset_id_col_index_for_mapping = header_for_mapping.index('Dataset_ID') 
        
        for row_idx, row in enumerate(reader_for_mapping):
            if row and len(row) > dataset_id_col_index_for_mapping:
                try:
                    original_id_val = int(row[dataset_id_col_index_for_mapping])
                    if original_id_val in local_dataset_id_mapping:
                        # Replace original ID with the mapped ID (convert to string for CSV writing).
                        row[dataset_id_col_index_for_mapping] = str(local_dataset_id_mapping[original_id_val])
                        processed_rows.append(row)
                    else:
                        logs.append(f"WARNING: File '{file_name}', row {row_idx+2}: Original 'Dataset_ID' {original_id_val} not found in this file's local mapping (this might happen if it was invalid during collection). This row has been skipped from output.")
                except ValueError:
                    logs.append(f"WARNING: File '{file_name}', row {row_idx+2}: 'Dataset_ID' value '{row[dataset_id_col_index_for_mapping]}' cannot be parsed as an integer. This row has been skipped from output.")
            elif row: # Only log if the row actually exists but is too short
                logs.append(f"WARNING: File '{file_name}', row {row_idx+2}: Row is too short and does not contain 'Dataset_ID'. This row has been skipped from output.")
            # If row is completely empty, it's typically ignored by csv.reader anyway.

        # Write processed data to the new file.
        with open(output_file_path, 'w', newline='', encoding='utf-8') as outfile:
            writer = csv.writer(outfile, delimiter=output_delimiter_for_this_file)
            writer.writerows(processed_rows)
        
        logs.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Successfully processed and saved '{file_name}' to '{output_file_path}' (using delimiter '{output_delimiter_for_this_file}')")
        return logs, True

    except FileNotFoundError:
        logs.append(f"ERROR: File '{file_path}' not found (possibly moved or deleted). Skipping this file.")
        return logs, False
    except Exception as e:
        # Catch any other unexpected errors during processing a single file.
        logs.append(f"FATAL ERROR: An unexpected exception occurred while processing file '{file_name}': {e}")
        # Add more context if available
        if 'first_line_content' in locals():
            logs.append(f"    First line content (raw): '{first_line_content}'")
        return logs, False
    finally:
        # Explicitly clear mapping-related variables at the end of the function.
        # This is primarily for visual clarity and debugging, as Python's local scope handles GC.
        del unique_ids_in_order
        del seen_ids_for_lookup
        del local_dataset_id_mapping
        del processed_rows
        logs.append(f"DEBUG: Explicitly cleared local mapping variables for '{file_name}'.")

# --- Main Script Logic ---
def main():
    script_dir = get_script_directory()
    output_dir = os.path.join(script_dir, OUTPUT_SUBFOLDER_NAME)

    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting CSV/TSV file processing (local Dataset_ID remapping by first appearance)...")
    print(f"Script Directory: {script_dir}")
    print(f"Output Directory: {output_dir}")
    if FORCE_INPUT_DELIMITER:
        print(f"Forced Input Delimiter: '{FORCE_INPUT_DELIMITER}' (Auto-detection disabled)")
    if OUTPUT_OVERRIDE_DELIMITER:
        print(f"Forced Output Delimiter: '{OUTPUT_OVERRIDE_DELIMITER}'")
    print("-" * 70)

    # 1. Find all input files.
    csv_file_paths = find_input_files(script_dir, OUTPUT_SUBFOLDER_NAME)

    if not csv_file_paths:
        print("ERROR: No .csv or .tsv files found in the script directory. Ensure files exist and are not in the output subfolder.")
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script Finished (No files to process).")
        return # Exit if no files found

    print(f"Detected {len(csv_file_paths)} input files:")
    for p in csv_file_paths:
        print(f"  - {os.path.basename(p)}")
    print("-" * 70)

    # 2. Prepare output directory.
    try:
        os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist.
        print(f"Output directory created/verified: {output_dir}")
    except Exception as e:
        print(f"FATAL ERROR: Could not create output directory '{output_dir}': {e}")
        print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script Finished (Output directory error).")
        return # Exit if cannot create output directory

    overall_success = True # Track if any file processing failed
    
    # 3. Iterate through each file, process independently.
    for file_path in csv_file_paths:
        # Call the isolated function to process each file.
        file_logs, file_success = process_single_file(file_path, output_dir)
        
        # Print logs for the current file immediately after it's processed.
        for msg in file_logs:
            print(msg)
        
        if not file_success:
            overall_success = False
            print(f"WARNING: Processing of '{os.path.basename(file_path)}' failed or skipped.")
        print("=" * 50) # Separator for clarity between file processing.

    print("\n" + "=" * 70)
    print("Processing Complete. Summary:")
    if overall_success:
        print("Script execution successful. All files processed independently.")
    else:
        print("Script execution finished with some errors or warnings. Please review the messages above.")
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Script Finished.")


if __name__ == "__main__":
    main()
