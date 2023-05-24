"""
create_merged_df.py - Merge the summary and input dataframes for the gauntlet dataset
    NOTE: run this from the root of the repo
Usage:
    create_merged_df.py [options]
"""
import fire
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(
    summary_data_path: str = "as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet",
    input_data_path: str = "as-dataset/gauntlet_input_documents.parquet",
    merge_on: str = "source_doc_id",
    output_path: str = None,
    verbose: bool = False,
):
    summary_data_path = Path(summary_data_path)
    input_data_path = Path(input_data_path)
    logging.info(f"Readin data from {summary_data_path}\nand {input_data_path}")
    output_path = (
        Path(output_path)
        if output_path
        else summary_data_path.parent / "summcomparer_gauntlet_merged.parquet"
    )
    summary_df = pd.read_parquet(summary_data_path)
    if verbose:
        print(summary_df.info())

    input_df = pd.read_parquet(input_data_path)
    if verbose:
        print(input_df.info())

    assert merge_on in input_df.columns, f"{merge_on} not in input_df columns"
    assert merge_on in summary_df.columns, f"{merge_on} not in summary_df columns"
    merged_df = summary_df.merge(input_df, on=merge_on, suffixes=("", "_input"))

    # Drop columns that are duplicated between the two dataframes
    cols_to_drop = [col for col in merged_df.columns if col.endswith("_input")]
    merged_df = merged_df.drop(columns=cols_to_drop)

    logging.info(f"Writing merged dataframe to {output_path}")
    merged_df.to_parquet(output_path)
    logging.info("Done!")


if __name__ == "__main__":
    fire.Fire(main)
