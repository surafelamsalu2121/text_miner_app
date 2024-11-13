import os
import json  # Import json module here

def save_json_output(content, filename):
    """Save parsed content to a JSON file."""
    output_dir = "parsed_outputs"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{filename}.json")
    with open(file_path, "w") as file:
        json.dump(content, file, indent=4)
    return file_path
