import os
import json
from pathlib import Path

def scan_python_files(root_dir="."):
    """
    Recursively scan directory for Python files and create a dictionary with
    file paths as keys and file contents as values.
    
    Args:
        root_dir (str): Root directory to start scanning from
    
    Returns:
        dict: Dictionary with file paths as keys and file contents as values
    """
    # Get the absolute path of the current script
    current_script = os.path.abspath(__file__)
    
    # Initialize dictionary to store results
    codebase = {}
    
    # Walk through directory tree
    for root, _, files in os.walk(root_dir):
        # Skip tests directory
        if "tests" in Path(root).parts:
            continue
            
        for file in files:
            # Get full file path
            file_path = os.path.join(root, file)
            abs_file_path = os.path.abspath(file_path)
            
            # Skip if this is the current script
            if abs_file_path == current_script:
                continue
                
            # Skip if not a Python file
            if not file.endswith('.py'):
                continue
                
            try:
                # Get relative path from root directory
                rel_path = os.path.relpath(file_path, root_dir)
                
                # Read file contents
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Store in dictionary using relative path as key
                codebase[rel_path] = content
                    
            except Exception as e:
                print(f"Error reading file {file_path}: {str(e)}")
                continue
    
    return codebase

def save_codebase_json(codebase, output_file="codebase.json"):
    """
    Save the codebase dictionary to a JSON file.
    
    Args:
        codebase (dict): Dictionary containing the codebase
        output_file (str): Name of the output JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(codebase, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved codebase to {output_file}")
    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")

def main():
    # Get codebase dictionary
    codebase = scan_python_files()
    
    # Save to JSON file
    save_codebase_json(codebase)
    
    # Print summary
    print(f"Scanned {len(codebase)} Python files")

if __name__ == "__main__":
    main()