import json
import logging
from pathlib import Path

import fire
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm import tqdm


def setup_logging():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_master_data(master_data_file):
    # Load the master data from the JSON file
    with master_data_file.open("r", encoding="utf-8") as f:
        source_file_to_id = json.load(f)
    return source_file_to_id


def get_best_match(summary_file, source_file_to_id):
    # Remove the '_summary.txt' from the summary filename
    clean_summary_file = summary_file.replace("_summary.txt", "")

    try:
        # Use fuzzywuzzy's process.extractOne() function to find the source file that best matches the summary file
        best_match = process.extractOne(
            clean_summary_file, list(source_file_to_id.keys())
        )

        # Get the ID of the best match from the source_file_to_id dictionary
        best_match_id = source_file_to_id[best_match[0]]

        return best_match_id
    except KeyError as e:
        logging.error(f"KeyError - {summary_file}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error - {summary_file}: {e}")
        return None


def main(
    dataframe_file: str,
    master_data_file: str = "gauntlet_master_data.json",
    filename_column: str = "file_name",
    new_col_basename: str = "source_file",
    output_file: str = None,
):
    setup_logging()

    master_data_file = Path(master_data_file)
    dataframe_file = Path(dataframe_file)
    assert master_data_file.exists(), f"{master_data_file} not found"
    assert dataframe_file.exists(), f"{dataframe_file} not found"
    output_file = (
        Path(output_file)
        if output_file
        else dataframe_file.parent / "gauntlet_docs_files_mapped.csv"
    )
    logging.info(f"Output file: {output_file}")
    source_file_to_id = load_master_data(master_data_file)

    # Load the dataframe from the CSV file
    df = pd.read_csv(dataframe_file)
    logging.info(f"Loaded dataframe, info: {df.info()}")
    # Apply the get_best_match function to each summary file in the dataframe
    # Use tqdm's progress_apply to show a progress bar
    tqdm.pandas()
    df[new_col_basename] = df[filename_column].progress_apply(
        lambda x: get_best_match(x, source_file_to_id)
    )

    # Save the dataframe to the output CSV file
    df.to_csv(output_file, index=False)
    logging.info(f"Saved mapped dataframe to:\n\t{str(output_file)}")


if __name__ == "__main__":
    fire.Fire(main)
