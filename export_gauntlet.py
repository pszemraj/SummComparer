"""
export_gauntlet - export the summary Gauntlet dataset to a CSV data file/parquet file

Usage:
    export_gauntlet.py <dropbox_link> [--output_folder=<output_folder>] [--keep_zip] [--keep_extracted] [--parquet] [--score_split_token=<score_split_token>]
"""
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import fire
import pandas as pd
import requests
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

KEY_MAPPING = {
    "run_date": "date",
    "date-run": "date",
    "META_date": "date",
    "huggingface-model-tag": "model_name",
    "META_huggingface_model": "model_name",
    "model": "model_name",
    "METADATA.META_huggingface_model": "model_name",
    "META_textsum_version": "textsum_version",
}


# def standardize_keys(json_obj: dict, mapping: dict) -> dict:
#     standardized_json = {}
#     for key, value in json_obj.items():
#         if isinstance(value, dict):
#             value = standardize_keys(value, mapping)
#         if key in mapping:
#             standardized_key = mapping[key]
#         else:
#             standardized_key = key
#         standardized_json[standardized_key] = value
#     return standardized_json


def standardize_keys(
    json_obj: dict, mapping: dict, collapse_sub_dicts: bool = True
) -> dict:
    def flatten_dict(d, parent_key="", sep=".", track_key: bool = False):
        items = []
        for k, v in d.items():
            new_key = parent_key + sep + k if parent_key and track_key else k
            if isinstance(v, dict) and collapse_sub_dicts:
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    standardized_json = flatten_dict(json_obj)
    standardized_json = {mapping.get(k, k): v for k, v in standardized_json.items()}
    return standardized_json


def download_and_extract_data(
    dropbox_link: str, data_zip_path: Path, extract_root_dir: Path = None
):
    extract_root_dir = extract_root_dir or Path.cwd() / "gauntlet"
    if extract_root_dir.exists():
        logging.warning(f"Removing existing directory: {extract_root_dir}")
        shutil.rmtree(extract_root_dir)

    response = requests.get(dropbox_link)
    with data_zip_path.open("wb") as f:
        f.write(response.content)
    with ZipFile(data_zip_path, "r") as zf:
        zf.extractall(extract_root_dir)
    logging.info(f"Extracted data to {extract_root_dir}")
    return extract_root_dir


def export_summary_gauntlet(
    dropbox_link: str,
    output_folder: Optional[str] = None,
    no_skip_textsum_cfg: bool = False,
    score_split_token: str = "Section Scores",
    keep_zip: bool = False,
    keep_extracted: bool = False,
    parquet: bool = False,
    debug: bool = False,
) -> None:
    """
    export_summary_gauntlet - export the summary Gauntlet dataset to a CSV data file

    :param str dropbox_link: link to the Dropbox zip file containing the Gauntlet dataset
    :param Optional[str] output_folder: path to the output folder, default: None
    :param bool parquet: whether to save the DataFrame to a parquet file, default: False
    :param bool keep_zip: whether to keep the downloaded zip file, default: False
    :param bool keep_extracted: whether to keep the extracted data, default: False
    :param str score_split_token: token to split the summary text files on, default: "Section Scores"
    """
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    logging.info("Starting export of the summary Gauntlet dataset...")
    logging.info(f"Downloading data from {dropbox_link}")
    data_zip_path = Path.cwd() / "data.zip"
    extract_root_dir = download_and_extract_data(dropbox_link, data_zip_path)

    if not keep_zip:
        data_zip_path.unlink()
    assert extract_root_dir.exists(), "The specified directory does not exist."

    df_list = []
    bottom_dirs = [
        Path(dir) for dir, subdirs, files in os.walk(extract_root_dir) if not subdirs
    ]
    for bottom_dir in tqdm(bottom_dirs, desc="Processing gauntlet output"):
        params_dict = {}
        _local_files = [f for f in bottom_dir.iterdir() if f.is_file()]
        _local_files.sort(
            key=lambda f: (f.suffix != ".json", f)
        )  # all json files first
        _local_suffixes = {f.suffix for f in _local_files}
        if not ".txt" in _local_suffixes:
            logging.debug(f"Skipping non-text directory: {bottom_dir}")
            continue
        if not ".json" in _local_suffixes:
            logging.warning(f"No JSON file found for: {bottom_dir}")

        for f_path in _local_files:
            if f_path.is_dir() or (f_path.suffix not in {".json", ".txt"}):
                logging.debug(f"Skipping non-text file: {f_path}")
                continue

            if not no_skip_textsum_cfg and f_path.name == "textsum_config.json":
                logging.debug(f"Skipping {f_path}")
                continue
            if f_path.suffix == ".json":
                params_dict = {}
                try:
                    with f_path.open("r", encoding="utf-8", errors="ignore") as f:
                        params_dict = json.load(f)
                except json.JSONDecodeError:
                    logging.error(f"Error loading JSON file: {f_path}")
                    continue

                params_dict = standardize_keys(params_dict, KEY_MAPPING)
                continue

            try:
                with f_path.open("r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text = text.split(score_split_token, maxsplit=1)[0].strip()
            except FileNotFoundError:
                logging.error(f"File not found: {f_path}")
                continue

            row_dict = {
                "GAUNTLET_PATH": f_path.relative_to(extract_root_dir),
                "file_name": f_path.name,
                "summary": text,
            }
            for key in params_dict.keys():
                row_dict[key] = params_dict[key]
            df_list.append(pd.DataFrame([row_dict]))

    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True).reset_index(drop=True)
    else:
        raise ValueError("No data found in the specified directory.")

    df["GAUNTLET_PATH"] = df.GAUNTLET_PATH.apply(lambda x: str(x))
    df.dropna(axis=1, how="all", inplace=True)

    output_folder = Path(output_folder) if output_folder else Path.cwd() / "as-dataset"
    output_folder.mkdir(exist_ok=True, parents=True)
    logging.info(f"Saving the summary data to {output_folder} ...")
    output_csv = output_folder / "summary_gauntlet_dataset.csv"

    df = df.convert_dtypes()
    print(df.info())
    df.to_csv(output_csv, index=False)
    logging.info(f"Export saved to:\n\t{output_csv}")
    if parquet:
        output_parquet = output_csv.with_suffix(".parquet")
        df.to_parquet(output_parquet, index=False)
        logging.info(f"Export saved as parquet:\n\t{output_parquet}")

    if not keep_extracted:
        logging.info("Removing extracted data folder...")
        shutil.rmtree(extract_root_dir)

    logging.info("Done!")


if __name__ == "__main__":
    fire.Fire(export_summary_gauntlet)
