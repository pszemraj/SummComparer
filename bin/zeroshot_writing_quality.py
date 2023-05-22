"""
zeroshot_writing_quality - predict the quality of a text string or file (dataframe or plain text)

Usage:
    zeroshot_writing_quality.py <input> [--output_folder=<output_folder>] [--model_name=<model_name>] [--batch_size=<batch_size>] [--fp16] [--tf32] [--bf16] [--8bit] [--device=<device>] [--detail] [--verbose] [--log_file=<log_file>]
"""
import io
import json
import logging
import pprint as pp
import time
from datetime import datetime
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def setup_logging(level=logging.INFO):
    """setup_logging - set up logging"""
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def get_timestamp(detail=False):
    """get_timestamp - get a timestamp string"""
    if detail:
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    return datetime.now().strftime("%b-%d-%Y_%H-%M")


def enable_tf32():
    """enable_tf32 - enable TensorFloat32 (TF32) computation"""
    logging.debug("Enabling TF32 computation")
    torch.backends.cuda.matmul.allow_tf32 = True


DTYPE_MAP = {"fp32": torch.float32, "bf16": torch.bfloat16, "8bit": torch.uint8}
DEFAULT_LABELS = [
    "logically coherent",
    "well-written",
    "confusing",
    "concise",
    "detailed",
    "comprehensive",
    "objective",
    "well-structured",
]


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, pd.Index):
            return obj.tolist()
        if isinstance(obj, pd.Categorical):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.StringDtype):
            return str(obj)
        if isinstance(obj, pd.Int64Dtype):
            return str(obj)
        if isinstance(obj, np.dtype):
            return str(obj)
        return super().default(obj)


def infer_quality(
    sequence_to_classify: str,
    classifier: pipeline,
    candidate_labels: list = None,
    n_digits: int = 6,
) -> dict:
    """
    infer_quality - perform zero-shot classification on a text string

    :param str sequence_to_classify: the text string to classify
    :param pipeline classifier: zero-shot classifier pipeline
    :param list candidate_labels: list of candidate labels, defaults to DEFAULT_LABELS
    :param int n_digits: number of digits to round scores to, defaults to 6
    :return dict: dictionary of labels and scores
    """
    st = time.perf_counter()
    candidate_labels = candidate_labels or DEFAULT_LABELS
    logging.debug(f"candidate_labels:\n{pp.pformat(candidate_labels)}")
    result = classifier(
        sequence_to_classify.strip(),
        candidate_labels,
        hypothesis_template="This document summary is {}",
        multi_label=True,
        truncation=True,
    )

    output_dict = {
        label: round(score, n_digits)
        for label, score in zip(result["labels"], result["scores"])
    }
    sorted_output_dict = {key: output_dict[key] for key in sorted(output_dict)}

    rt = round((time.perf_counter() - st), 3)
    logging.debug(f"runtime:\t{rt} sec")
    return sorted_output_dict


def predict_string(input_text, classifier, verbose=False):
    """predict_string - predict the quality of a string of text"""
    logging.info(f"Processing text data:\n\t{input_text}")
    if verbose:
        print(f"Processing text string:\n\t{input_text}")
    result = infer_quality(input_text, classifier)
    print(f"Results:\n{pp.pformat(result)}")


def predict_file(
    input_file: str or Path,
    classifier: pipeline,
    output_path: str or Path = None,
    verbose=False,
):
    """
    predict_file - predict the quality of a file of text

    :param strorPath input_file: _description_
    :param pipeline classifier: _description_
    :param strorPath output_path: _description_, defaults to None
    :param bool verbose: _description_, defaults to False
    """
    input_path = Path(input_file)
    output_path = (
        Path(output_path)
        if output_path
        else input_path.parent / f"{input_path.stem}_WQ_predictions.json"
    )
    with open(input_file, "r", encoding="utf-8") as file:
        text = file.read()
    result = infer_quality(text, classifier)
    if verbose:
        print(f"Results:\n{pp.pformat(result)}")
    outdata = {
        "input_file": input_file,
        "results": result,
        "model": classifier.model.config.name_or_path,
        "timestamp": get_timestamp(),
    }
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(outdata, file, indent=4)
    logging.info(f"Results saved to {output_path}")


def process_dataframe(
    df: pd.DataFrame,
    classifier: pipeline,
    text_column: str = "text",
    output_path: str = None,
    parquet=False,
    prefix: str = "pred_WQ",
):
    """
    process_dataframe - helper function to process a DataFrame with a text column

    :param pd.DataFrame df: dataframe to process
    :param pipeline classifier: zero-shot classifier pipeline
    :param str text_column: name of text column in DataFrame, defaults to "text"
    :param str output_path: path to save results to, defaults to None
    :param bool parquet: save results as parquet file, defaults to False
    :param str prefix: prefix for predicted label columns, defaults to "pred_WQ"
    """
    assert isinstance(df, pd.DataFrame), "Input must be a DataFrame"
    assert (
        text_column in df.columns
    ), f"Text column {text_column} not found in DataFrame"

    output_path = (
        Path(output_path)
        if output_path
        else Path.cwd() / f"{df.name}_WQ_predictions.csv"
    )
    logging.info(
        f"Processing DataFrame with {len(df)} rows - text column: {text_column}"
    )
    tqdm.pandas(desc="Processing DataFrame")
    results = df[text_column].progress_apply(lambda x: infer_quality(x, classifier))
    results = results.reset_index(drop=True)
    logging.info("finished predicting on DataFrame")
    # initialize new columns
    for label in results[0].keys():
        df[f"{prefix}_{label}"] = None
    # append results
    for idx, result in enumerate(tqdm(results, desc="Appending Results")):
        for label, score in result.items():
            df.loc[idx, f"{prefix}_{label}"] = score
    df = df.convert_dtypes()
    if parquet:
        df.to_parquet(output_path.with_suffix(".parquet"))
        logging.info(f"Results saved to {output_path.with_suffix('.parquet')}")
    else:
        df.to_csv(output_path, index=False)
        logging.info(f"Results saved to {output_path}")

    buffer = io.StringIO()
    df.info(buf=buffer)
    metadata = {
        "input_dataframe": pp.pformat(buffer.getvalue()),
        "pred_stats": pd.DataFrame(results).convert_dtypes().describe().to_dict(),
        "model": classifier.model.config.name_or_path,
        "timestamp": get_timestamp(),
    }
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, cls=CustomEncoder)
    logging.info(f"Metadata saved to {output_path.with_suffix('.json')}")


def main(
    input_data: str,
    model_name="microsoft/deberta-large-mnli",
    dtype_model="fp32",
    truncation_max_length: int = None,
    text_column="summary",
    test_df: bool = False,
    parquet=False,
    tf32=False,
    loglevel="INFO",
    verbose=False,
):
    """
    main function for running zero-shot classification on text data to assess writing quality

    :param str input_data: input data file or string to process
    :param str model_name: model name or path, defaults to "microsoft/deberta-large-mnli"
    :param str dtype_model: dtype to load model in, defaults to "fp32"
    :param int truncation_max_length: override default truncation length to N tokens, defaults to None
    :param str text_column: name of text column if using dataframe input, defaults to "summary"
    :param bool parquet: save output as parquet file, defaults to False
    :param bool tf32: enable tf32 precision, defaults to False
    :param str loglevel: logging level, defaults to "INFO"
    :param bool verbose: print verbose output, defaults to False
    """
    setup_logging(level=logging.getLevelName(loglevel.upper()))
    logger = logging.getLogger(__name__)
    if tf32:
        enable_tf32()
    logger.info(f"loading model: {model_name}")
    tokenizer = (
        AutoTokenizer.from_pretrained(
            model_name, model_max_length=truncation_max_length
        )
        if truncation_max_length
        else AutoTokenizer.from_pretrained(model_name)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        torch_dtype=DTYPE_MAP[dtype_model],
    )
    device = 0 if torch.cuda.is_available() else -1
    logger.info(f"using device ID: {device}")
    classifier = pipeline(
        "zero-shot-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=DTYPE_MAP[dtype_model],
    )

    input_path = Path(input_data)
    if input_path.is_file() and input_path.exists():
        if input_path.suffix in [".csv", ".parquet"]:
            logger.info("Detected DataFrame input. Processing...")
            df = (
                pd.read_csv(input_path)
                if input_path.suffix == ".csv"
                else pd.read_parquet(input_path)
            )
            output_path = input_path.parent / f"{input_path.stem}_WQ_predictions.csv"
            df = df.sample(n=10).convert_dtypes() if test_df else df.convert_dtypes()

            logger.info(
                f"Processing DataFrame with {len(df)} rows (test mode: {test_df})"
            )
            process_dataframe(
                df,
                classifier,
                text_column=text_column,
                output_path=output_path,
                parquet=parquet,
            )
        else:
            logger.info("Detected text file input. Processing as text file...")
            predict_file(input_path, classifier, verbose=verbose)
    else:
        logger.info("Processing direct text input...")
        predict_string(str(input_data), classifier, verbose=verbose)
    logger.info("Done!")


if __name__ == "__main__":
    fire.Fire(main)
