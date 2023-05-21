import json
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import fire
import logging
from typing import Optional
import requests
from zipfile import ZipFile

logging.basicConfig(level=logging.INFO)


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


def generate_gauntlet_summary(
    dropbox_link: str, output_folder: Optional[str] = None
) -> None:
    logging.info("Downloading data...")
    data_zip_path = Path("data.zip")
    download_and_extract_data(dropbox_link, data_zip_path)

    root_dir = Path("gauntlet")
    assert root_dir.exists(), "The specified directory does not exist."

    KEY_MAPPING = {
        "run_date": "date",
        "date-run": "date",
        "META_date": "date",
        "huggingface-model-tag": "model_name",
        "META_huggingface_model": "model_name",
        "model": "model_name",
        "METADATA.META_huggingface_model": "model_name",
    }

    df_list = []

    for f_path in tqdm(root_dir.glob("**/*"), desc="Processing files"):
        if f_path.is_dir() or (f_path.suffix not in {".json", ".txt"}):
            continue

        if f_path.suffix == ".json":
            params_dict = {}
            try:
                with f_path.open("r") as f:
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
            Path(output_folder) if output_folder else Path.cwd() / "gauntlet-as-dataset"
        )
        output_folder.mkdir(exist_ok=True, parents=True)
        output_csv = output_folder / "gauntlet_summary_data.csv"
        output_parquet = output_folder / "gauntlet_summary_data.parquet"
        df = df.convert_dtypes()
        print(df.info())
        df.to_csv(output_csv, index=False)
        df.to_parquet(output_parquet, index=False)
        logging.info(
            f"Successfully saved the summary data to {output_csv} and {output_parquet}"
        )
    else:
        logging.warning("No data found in the specified directory.")


if __name__ == "__main__":
    fire.Fire(generate_gauntlet_summary)
