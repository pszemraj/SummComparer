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
        master_data = json.load(f)
    return master_data


def get_best_match(summary_file, master_data):
    # Remove the '_summary.txt' from the summary filename
    clean_summary_file = summary_file.replace("_summary.txt", "")

    try:
        # Use fuzzywuzzy's process.extractOne() function to find the source file that best matches the summary file
        best_match = process.extractOne(
            clean_summary_file, [record["source_file"] for record in master_data]
        )

        # Get the record of the best match from the master_data
        best_match_record = next(
            record for record in master_data if record["source_file"] == best_match[0]
        )

        return best_match_record
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
    master_data = load_master_data(master_data_file)

    # Load the dataframe from the CSV file
    df = pd.read_csv(dataframe_file)
    logging.info(f"Loaded dataframe, info: {df.info()}")
    # Apply the get_best_match function to each summary file in the dataframe
    # Use tqdm's progress_apply to show a progress bar
    tqdm.pandas()
    df = df.join(
        df[filename_column].progress_apply(
            lambda x: pd.Series(get_best_match(x, master_data))
        )
    )

    # Save the dataframe to the output CSV file
    df.to_csv(output_file, index=False)
    logging.info(f"Saved mapped dataframe to:\n\t{str(output_file)}")


if __name__ == "__main__":
    fire.Fire(main)
