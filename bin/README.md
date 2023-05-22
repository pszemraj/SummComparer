# SummComparer scripts

> Various scripts for computing metrics on summary/document text results. in the dataset.

## Installation

_Note: the requirements in this directory are not the same as the requirements in the root directory._

```bash
pip install -r ./requirements.txt
```

## Scripts

The following scripts are available:

1. `detect_gibberish.py`: Runs a text classifier to detect gibberish text on each row in `<text_column>` in `<input_csv>`. Saves the results in `<output_csv>`
2. `pred_CoLA_scores.py`: Runs a text classifier to predict CoLA scores on each row in `<text_column>` in `<input_csv>`. Saves the results in `<output_csv>`
3. `zeroshot_writing_quality.py`: Runs a text classifier to predict writing quality on each row in `<text_column>` in `<input_csv>`. Saves the results in `<output_csv>`
