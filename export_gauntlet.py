"""
compile_gauntlet.py is a script that compiles the Gauntlet dataset into a single CSV file.
"""
import json
import logging
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
}


def standardize_keys(json_obj: dict, mapping: dict) -> dict:
    standardized_json = {}
    for key, value in json_obj.items():
        if isinstance(value, dict):
            value = standardize_keys(value, mapping)
        if key in mapping:
            standardized_key = mapping[key]
        else:
            standardized_key = key
        standardized_json[standardized_key] = value
    return standardized_json


def download_and_extract_data(dropbox_link: str, data_zip_path: Path) -> None:
    response = requests.get(dropbox_link)
    with data_zip_path.open("wb") as f:
        f.write(response.content)
    with ZipFile(data_zip_path, "r") as zf:
        zf.extractall("gauntlet")


def export_summary_gauntlet(
    dropbox_link: str,
    output_folder: Optional[str] = None,
    keep_zip: bool = False,
    save_parquet: bool = False,
) -> None:
    """
    export_summary_gauntlet - export the summary Gauntlet dataset to a CSV data file

    :param str dropbox_link: link to the Dropbox zip file containing the Gauntlet dataset
    :param Optional[str] output_folder: path to the output folder, defaults to None
    :param bool save_parquet: whether to save the DataFrame to a parquet file, defaults to False
    """
    logging.info("Downloading data...")
    data_zip_path = Path.cwd() / "data.zip"
    download_and_extract_data(dropbox_link, data_zip_path)
    if not keep_zip:
        data_zip_path.unlink()
    root_dir = Path("gauntlet")
    assert root_dir.exists(), "The specified directory does not exist."

    df_list = []

    for f_path in tqdm(root_dir.glob("**/*"), desc="Processing files"):
        if f_path.is_dir() or (f_path.suffix not in {".json", ".txt"}):
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
            text = text.split("---", maxsplit=1)[0].strip()
        except FileNotFoundError:
            logging.error(f"File not found: {f_path}")
            continue

        row_dict = {
            "GAUNTLET_PATH": f_path.relative_to(root_dir),
            "file_name": f_path.name,
            "text": text,
        }
        for key in params_dict.keys():
            row_dict[key] = params_dict[key]
        df_list.append(pd.DataFrame([row_dict]))

    if len(df_list) > 0:
        df = pd.concat(df_list, ignore_index=True)
    else:
        df = pd.DataFrame()

    df["GAUNTLET_PATH"] = df.GAUNTLET_PATH.apply(lambda x: str(x))

    if not df.empty:
        output_folder = (
            Path(output_folder) if output_folder else Path.cwd() / "as-dataset"
        )
        output_folder.mkdir(exist_ok=True, parents=True)
        logging.info(f"Saving the summary data to {output_folder}")
        output_csv = output_folder / "gauntlet_summary_data.csv"
        df = df.convert_dtypes()
        print(df.info())
        df.to_csv(output_csv, index=False)
        if save_parquet:
            output_parquet = output_folder / "gauntlet_summary_data.parquet"
            df.to_parquet(output_parquet, index=False)
            logging.info(f"Saved the summary data to {output_parquet}")
        logging.info(f"Done! saved the summary data to {output_csv}")
    else:
        logging.warning("No data found in the specified directory.")


if __name__ == "__main__":
    fire.Fire(export_summary_gauntlet)
