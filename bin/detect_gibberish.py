import logging
from pathlib import Path

import fire
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from cleantext import clean


def setup_logging():
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_model(
    model_name_or_path: str = "madhurjindal/autonlp-Gibberish-Detector-492513457",
    do_compile: bool = False,
    bettertransformer: bool = False,
):
    logging.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
    )
    if torch.__version__ > "2.0.0" and do_compile:
        logging.info("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")
    elif bettertransformer:
        logging.info("using bettertransformer...")
        from optimum.bettertransformer import BetterTransformer

        model = BetterTransformer.transform(model)
    else:
        logging.info("no post-training optimizations applied")
    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, device_map="auto"
    )
    return classifier


def predict_label_and_score(text, classifier):
    # Predict a label and score for the text
    proc_text = clean(text, lower=False, no_line_breaks=True, no_urls=True)
    result = classifier(proc_text, truncation=True)[0]
    label = result["label"]
    score = round(result["score"], 6)
    return label, score


def convert2textquality(label: str, score: float):
    # Assign a raw score to the label
    label_to_raw_score = {
        "clean": 4,
        "mild gibberish": 3,
        "word salad": 2,
        "noise": 1,
    }
    raw_score = label_to_raw_score[label]

    # Calculate a weighted score by multiplying the raw score by the predicted score
    weighted_score = raw_score * score

    # Normalize the weighted score to the range 0-1
    normalized_text_quality = weighted_score / 4

    return normalized_text_quality


def main(
    dataframe_file: str,
    text_col_name: str = "text",
    output_file: str = None,
    cls_col_name: str = "label",
    score_col_name: str = "score",
    do_compile: bool = False,
    bettertransformer: bool = False,
    parquet: bool = False,
):
    setup_logging()

    dataframe_file = Path(dataframe_file)
    assert dataframe_file.exists(), f"{dataframe_file} not found"
    output_file = (
        Path(output_file)
        if output_file
        else Path.cwd() / "output" / f"{dataframe_file.stem}_gibberish_preds.csv"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output file: {output_file}")
    classifier = load_model(
        do_compile=do_compile,
        bettertransformer=bettertransformer,
    )

    # Load the dataframe from the CSV file
    df = pd.read_csv(dataframe_file).convert_dtypes()
    logging.info(f"Loaded dataframe, shape: {df.shape}")
    assert (
        text_col_name in df.columns
    ), f"{text_col_name} not in dataframe columns. Found: {df.columns}"
    # Predict a label and score for each row in the specified column
    tqdm.pandas(desc="Predicting labels and scores...")
    df[cls_col_name] = ""
    df[score_col_name] = -1.0
    df[cls_col_name], df[score_col_name] = zip(
        *df[text_col_name].progress_apply(
            lambda x: predict_label_and_score(x, classifier)
        )
    )
    df["text_quality"] = -1.0
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc="Converting to text quality"
    ):
        df.loc[index, "text_quality"] = convert2textquality(
            row[cls_col_name], row[score_col_name]
        )
    logging.info(f"Predicted labels and scores for {len(df)} rows")
    logging.info(f"Distribution of scores:\n{df.text_quality.describe()}")
    # Save the dataframe to the output CSV file
    df.to_csv(output_file, index=False)
    logging.info(
        f"Saved DataFrame with predicted labels and scores to:\n\t{str(output_file)}"
    )
    if parquet:
        # Save the dataframe to a parquet file
        df.to_parquet(output_file.with_suffix(".parquet"), index=False)
        logging.info(
            f"Saved data as parquet to:\n\t{str(output_file.with_suffix('.parquet'))}"
        )


if __name__ == "__main__":
    fire.Fire(main)
