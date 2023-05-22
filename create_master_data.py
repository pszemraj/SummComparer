"""
create_master_data - Create a master data JSON file containing the source files and their metadata

Usage:
    create_master_data.py <input_file> [--output_file=<output_file>]
"""
import json
import logging
import uuid
from pathlib import Path

import fire
import pandas as pd

logging.basicConfig(
    filename="logfile.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)


def infer_domain(filename):
    """Infer the domain from the filename."""
    if filename.startswith("ASR"):
        return "ASR"
    elif filename.startswith("OCR"):
        return "OCR"
    elif filename.startswith("script"):
        return "Script"
    else:
        return "Unknown"


def create_master_data(input_file: str, output_file: str = "gauntlet_master_data.json"):
    """
    create_master_data - Create a master data JSON file containing the source files and their metadata

    :param str input_file: path to a text file containing the list of source files
    :param str output_file: path to the output JSON file
    :raises FileNotFoundError: if the input file does not exist
    """
    input_file_path = Path(input_file)
    if not input_file_path.exists():
        logging.error(f"{input_file_path} does not exist.")
        raise FileNotFoundError(f"{input_file_path} does not exist.")

    logging.info(f"Processing {input_file_path}.")
    with input_file_path.open("r", encoding="utf-8") as f:
        source_files = f.read().splitlines()
    logging.info(f"Found {len(source_files)} source files in {input_file_path}")

    # Create a master data dataframe with unique IDs for each source file
    master_data = pd.DataFrame(source_files, columns=["filename"])

    # Assign a 12-character UUID to each source file
    master_data["id"] = [str(uuid.uuid4())[:12] for _ in range(len(master_data))]

    # Infer the domain from the filename
    master_data["domain"] = master_data["filename"].apply(infer_domain)

    output_file_path = Path(output_file)
    logging.info(f"Saving master data to {output_file_path}")
    with output_file_path.open("w", encoding="utf-8") as f:
        json.dump(master_data.to_dict(orient="records"), f, indent=3)
    logging.info(f"Saved master data to {output_file_path}")


if __name__ == "__main__":
    fire.Fire(create_master_data)
