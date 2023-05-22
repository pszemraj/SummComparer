"""
build_src_df.py - Create a DataFrame containing original gauntlet docs text files and their metadata

Usage:
    build_src_df.py <src_dir> [--master_data=<master_data>] [--output_file=<output_file>] [--parquet] [--text_col=<text_col>] [--src_prefix=<src_prefix>]
"""
import json
import logging
from pathlib import Path

import fire
import pandas as pd
from tqdm.auto import tqdm


def setup_logging():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_master_data(master_data_file):
    # Load the master data from the JSON file
    with master_data_file.open("r", encoding="utf-8") as f:
        master_data = json.load(f)
    return master_data


def create_dataframe(
    src_dir,
    master_data,
    text_col="document_text",
    src_prefix: str = "source_doc",
):
    # Create a DataFrame from the master data and the original gauntlet docs text files
    df = pd.DataFrame(master_data).convert_dtypes()
    # rename the columns to start with src_prefix
    df = df.rename(columns={k: f"{src_prefix}_{k}" for k in df.columns})
    df[text_col] = ""
    errored_files = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        file_path = src_dir / row[f"{src_prefix}_filename"]
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                df.at[i, text_col] = f.read()
        else:
            logging.warning(f"{file_path} does not exist")
            errored_files.append(file_path)
    if errored_files:
        logging.warning(f"Errored files:\n\t{errored_files}")
    return df


def main(
    src_dir: str,
    master_data_file: str = "gauntlet_master_data.json",
    output_file: str = None,
    parquet: bool = False,
    text_col="document_text",
    src_prefix: str = "source_doc",
    drop_ids: list = None,
):
    """
    main function for build_src_df.py

    :param str src_dir: source directory containing the original gauntlet docs text files
    :param str master_data_file: path to the master data JSON file, default: "gauntlet_master_data.json"
    :param str output_file: path to the output CSV file, default: "gauntlet_source_documents.csv"
    :param bool parquet: whether to save the DataFrame to a parquet file, default: False
    :param str text_col: name of the column containing the document text, default: "document_text"
    :param str src_prefix: prefix to use for the source document columns, default: "source_doc"
    :param list drop_ids: list of ids to drop from the DataFrame, default: None
    """
    setup_logging()

    src_dir = Path(src_dir)
    master_data_file = Path(master_data_file)
    output_file = (
        Path(output_file)
        if output_file
        else Path.cwd() / "as-dataset" / "gauntlet_input_documents.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    assert src_dir.exists(), f"{src_dir} not found"
    assert master_data_file.exists(), f"{master_data_file} not found"
    logging.info(f"Output file: {output_file}")

    master_data = load_master_data(master_data_file)

    # Create a DataFrame from the master data and the original gauntlet docs text files
    df = create_dataframe(
        src_dir, master_data, text_col=text_col, src_prefix=src_prefix
    )
    df = df.reset_index(drop=True).convert_dtypes()
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
        df = df.reset_index(drop=True).convert_dtypes()
        logging.info(f"Dropped {start_len - len(df)} rows")

    # Save the dataframe to the output CSV file
    df.to_csv(output_file, index=False)
    logging.info(f"Saved DataFrame to:\n\t{str(output_file)}")
    if parquet:
        parquet_file = output_file.with_suffix(".parquet")
        df.to_parquet(parquet_file, index=False)
        logging.info(f"Saved DataFrame to:\n\t{str(parquet_file)}")

    logging.info("Done")


if __name__ == "__main__":
    fire.Fire(main)
