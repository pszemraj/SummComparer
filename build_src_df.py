"""
build_src_df.py - Create a DataFrame from the master data and the original gauntlet docs text files
"""
import json
import logging
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm


def setup_logging():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_master_data(master_data_file):
    # Load the master data from the JSON file
    with master_data_file.open("r", encoding="utf-8") as f:
        master_data = json.load(f)
    return master_data


def create_dataframe(src_dir, master_data):
    # Create a DataFrame from the master data and the original gauntlet docs text files
    df = pd.DataFrame(master_data)
    df["text"] = ""
    errored_files = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_path = src_dir / row["source_file"]
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                df.at[i, "text"] = f.read()
        else:
            logging.warning(f"{file_path} does not exist")
            errored_files.append(file_path)
    if errored_files:
        logging.warning(f"Errored files:\n\t{errored_files}")
    return df


def main(
    src_dir: str,
    master_data_file: str = "gauntlet_master_data.json",
    output_file: str = "gauntlet_source_documents.csv",
    save_parquet: bool = False,
):
    """
    main function for build_src_df.py

    :param str src_dir: source directory containing the original gauntlet docs text files
    :param str master_data_file: path to the master data JSON file, defaults to "gauntlet_master_data.json"
    :param str output_file: path to the output CSV file, defaults to "gauntlet_source_documents.csv"
    :param bool save_parquet: whether to save the DataFrame to a parquet file, defaults to False
    """
    setup_logging()

    src_dir = Path(src_dir)
    master_data_file = Path(master_data_file)
    output_file = Path(output_file)
    assert src_dir.exists(), f"{src_dir} not found"
    assert master_data_file.exists(), f"{master_data_file} not found"
    logging.info(f"Output file: {output_file}")
    master_data = load_master_data(master_data_file)

    # Create a DataFrame from the master data and the original gauntlet docs text files
    df = create_dataframe(src_dir, master_data)
    df = df.reset_index(drop=True).convert_dtypes()

    # Save the dataframe to the output CSV file
    df.to_csv(output_file, index=False)
    logging.info(f"Saved DataFrame to:\n\t{str(output_file)}")
    if save_parquet:
        parquet_file = output_file.with_suffix(".parquet")
        df.to_parquet(parquet_file, index=False)
        logging.info(f"Saved DataFrame to:\n\t{str(parquet_file)}")


if __name__ == "__main__":
    fire.Fire(main)
