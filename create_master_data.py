import json
import sys
import uuid
from pathlib import Path

import pandas as pd


def infer_domain(filename):
    # This function infers the domain from the filename
    # You can replace this with your own logic
    if filename.startswith("ASR"):
        return "ASR"
    elif filename.startswith("OCR"):
        return "OCR"
    elif filename.startswith("script"):
        return "Script"
    else:
        return "Unknown"


if __name__ == "__main__":
    file_list = Path.cwd() / "gauntlet_docs_files.txt"
    assert file_list.exists(), "gauntlet_docs_files.txt not found in current directory"

    with open(file_list, "r", encoding="utf-8") as f:
        source_files = f.read().splitlines()
    print(f"Found {len(source_files)} source files in gauntlet_docs_files.txt")

    for file in source_files:
        if not isinstance(file, str):
            print(f"Non-string element found in source_files: {file}")
            sys.exit()

    # Create a master data dataframe with unique IDs for each source file
    master_data = pd.DataFrame(source_files, columns=["filename"])

    # Assign a 12-character UUID to each source file
    master_data["id"] = [str(uuid.uuid4())[:12] for _ in range(len(master_data))]

    # Infer the domain from the filename
    master_data["domain"] = master_data["filename"].apply(infer_domain)

    # Convert the dataframe to a dictionary and save it as a JSON file
    output_file = Path.cwd() / "gauntlet_master_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(master_data.to_dict(orient="records"), f, indent=3)
    print(f"Saved master data to {output_file}")
