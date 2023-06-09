"""
map_gauntlet_files - fuzzy-match the summary files to their original input doc via a master data file

Usage:
    map_gauntlet_files.py <summary_file> [--master_data=<master_data>] [--filename_column=<filename_column>] [--src_prefix=<src_prefix>]
"""
import json
import logging
from pathlib import Path

import fire
import pandas as pd
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm


def setup_logging():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_master_data(master_data_file):
    # Load the master data from the JSON file
    with master_data_file.open("r", encoding="utf-8") as f:
        master_data = json.load(f)
    return master_data


def get_best_match(
    summary_file: str or Path,
    master_data: str or Path = "gauntlet_master_data.json",
    filename_key: str = "filename",
    src_prefix: str = "source_doc",
) -> dict:
    """
    get_best_match - match a summary file to a source file in the master data

    :param strorPath summary_file: path to the summary file
    :param strorPath master_data: path to the master data JSON file, default: "gauntlet_master_data.json"
    :param str filename_key: key in the master data JSON file that contains the filename, default: "filename"
    :param str src_prefix: prefix to add to the keys in the returned dict, default: "source_doc"
    :return dict: dict of the best match record from the master data
    """
    # Remove the '_summary.txt' suffix
    clean_summary_file = summary_file.replace("_summary.txt", "").strip()

    try:
        best_match = process.extractOne(
            clean_summary_file, [record[filename_key] for record in master_data]
        )

        best_match_record = next(
            record for record in master_data if record[filename_key] == best_match[0]
        )
        best_match_record = {
            f"{src_prefix}_{k}": v for k, v in best_match_record.items()
        }
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
    src_prefix: str = "source_doc",
    output_file: str = None,
    parquet: bool = False,
    drop_ids: list = None,
):
    """
    main - main function for the map_gauntlet_files script

    :param str dataframe_file: path to the CSV data file containing summary data
    :param str master_data_file: path to the JSON master data file, defaults to "gauntlet_master_data.json"
    :param str filename_column: column name in dataframe_file containing the filename, defaults to "file_name"
    :param str output_file: path to the output CSV file, defaults to None
    :param bool parquet: also save the output as a parquet file, defaults to False
    """
    setup_logging()

    master_data_file = Path(master_data_file)
    dataframe_file = Path(dataframe_file)
    assert master_data_file.exists(), f"{master_data_file} not found"
    assert dataframe_file.exists(), f"{dataframe_file} not found"
    output_file = (
        Path(output_file)
        if output_file
        else dataframe_file.parent / f"{dataframe_file.stem}_mapped_src_docs.csv"
    )
    logging.info(f"Output file: {output_file}")
    master_data = load_master_data(master_data_file)

    # Load the dataframe from the CSV file
    df = pd.read_csv(dataframe_file).convert_dtypes()
    logging.info(f"Loaded dataframe, info: {df.info()}")
    # Apply the get_best_match function to each summary file in the dataframe
    tqdm.pandas(desc="Mapping files")
    df = df.join(
        df[filename_column].progress_apply(
            lambda x: pd.Series(get_best_match(x, master_data, src_prefix=src_prefix))
        )
    )
    if drop_ids:
        search_col = f"{src_prefix}_id"
        if isinstance(drop_ids, str):
            drop_ids = [drop_ids]
        logging.info(f"Dropping rows with values in {search_col} matching: {drop_ids}")
        # check if any ids are not actually in master data and warn
        valid_ids = {record["id"] for record in master_data}
        invalid_ids = set(drop_ids) - set(valid_ids)
        if invalid_ids:
            logging.warning(
                f"Warning: {len(invalid_ids)} ids not found in master data: {invalid_ids}"
            )
        start_len = len(df)
        df = df[~df[search_col].isin(drop_ids)]
        df.reset_index(drop=True, inplace=True)
        logging.info(f"Dropped {start_len - len(df)} rows, new length: {len(df)}")
    # Save the dataframe to the output CSV file
    df.to_csv(output_file, index=False)
    logging.info(f"Saved mapped dataframe to:\n\t{str(output_file)}")
    if parquet:
        # Save the dataframe to a parquet file
        df.to_parquet(output_file.with_suffix(".parquet"), index=False)
        logging.info(
            f"Saved data as parquet to:\n\t{str(output_file.with_suffix('.parquet'))}"
        )


if __name__ == "__main__":
    fire.Fire(main)
