# SummComparer: Comparative analysis of summarization models

SummComparer is an initiative aimed at compiling, scrutinizing, and analyzing a [Summarization Gauntlet](https://www.dropbox.com/sh/axu1xlscrrexy55/AADAm01-4Zs3POyHQrgbDAsda?dl=0) with the goal of understanding/improving _what makes a summarization model do well_ in practical everyday use cases.

⚠️ This project is currently under active development and will continue to evolve over time. ⚠️

---

- [SummComparer: Comparative analysis of summarization models](#summcomparer-comparative-analysis-of-summarization-models)
  - [About](#about)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Compiling the Gauntlet](#compiling-the-gauntlet)
    - [Working with the Dataset](#working-with-the-dataset)
      - [Input Documents](#input-documents)
      - [Exploring the Dataset](#exploring-the-dataset)

---

## About

SummComparer's main aim is to test how well various summarization models work on long documents from a wide range of topics, **none of which** are part of standard training data[^1]. This "gauntlet" of topics helps us see how well the models can summarize both familiar and unfamiliar content. By doing this, we can understand how these models might perform in real-world situations where the content is unpredictable[^2]. This also helps us identify their limitations and ideally, understand what makes them work well.

[^1]: As it turns out, the practical application of summarization models **is not** the ritual of summarizing documents _you already know the summary of_ and benchmarking their ability to regurgitate these back to you via ROUGE scores as a testament of their performance. Who knew?
[^2]: i.e. you are not trying to hit a high score on the test set of [arXiv summarization](https://paperswithcode.com/dataset/arxiv-summarization-dataset) as a measure of a "good model", but rather actually read and use the summaries in real life.

Put another way, SummComparer can be thought of as a case study for the following scenario:

- You have a collection of documents that you need to summarize/understand for `<reason>`.
- You don't know what domain(s) these documents belong to **because you haven't read them**, and you don't have the time or inclination to read them fully.
  - You're hoping to get a general understanding of these documents from summaries, and then plan to decide which ones to do more in-depth reading on.
- You're not sure what the ideal summaries of these documents are **because if you knew that, you wouldn't need to summarize them with a language model**.
- So: Which model(s) should you use? How can you determine if the outputs are faithful without reading the source documents? Can you determine whether the model is performing well or not?

The idea for this project was born out of necessity: to test whether a summarization model was "good" or not, I would run it on a consistent set of documents and compare the generated summaries with the outputs of other models and my growing understanding of the documents themselves.

If `<new summarization model or technique>` claiming to be amazing is unable to summarize the [navy seals copypasta](https://knowyourmeme.com/memes/navy-seal-copypasta), OCR'd powerpoint slides, or a [short story](https://en.wikipedia.org/wiki/The_Most_Dangerous_Game), then it's probably not going to be very useful in the real world.

## Installation

To install the necessary packages, run the following command:

```bash
pip install -r requirements.txt
```

To install the package requirements for using the scripts in `bin/`, navigate to that directory and run:

```bash
pip install -r bin/requirements.txt
```

## Usage

### Compiling the Gauntlet

The current version supports Command Line Interface (CLI) usage. The recommended sequence of operations is as follows:

```bash
export_gauntlet.py
map_gauntlet_files.py
build_src_df.py
```

All CLI scripts utilize the `fire` package for CLI generation. For more information on how to use the CLI, run:

```bash
python <script_name>.py --help
```

### Working with the Dataset

> **Note:** The current version of the dataset is in a "raw" format. It has not been cleaned or pruned of unnecessary columns. This will be addressed in a future release.

The dataset files are located in `as-dataset/` and are saved as `.parquet` files. The dataset comprises two files, which can be conceptualized as two tables in a relational database:

- `as-dataset/gauntlet_input_documents.parquet`: This file contains the input documents for the gauntlet along with metadata/`id` fields as defined in `gauntlet_master_data.json`.
- `as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet`: This file contains the output summaries for the gauntlet with hyperparameters/models as columns. All summaries (rows) are mapped to their source documents (columns) by columns prefixed with `source_doc`.

You can load the data using `pandas`:

```python
import pandas as pd
df = pd.read_parquet('as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet')
df.info()
```

#### Input Documents

The `gauntlet_input_documents.parquet` file is required only if you need to examine the source documents themselves or perform any analysis using their text. Most of the necessary information is available in the `summary_gauntlet_dataset_mapped_src_docs.parquet` file.

The `gauntlet_input_documents.parquet` file contains the following columns:

```python
>>> import pandas as pd
>>> df = pd.read_parquet("as-dataset/gauntlet_input_documents.parquet").convert_dtypes()
>>> df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19 entries, 0 to 18
Data columns (total 4 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   source_doc_filename  19 non-null     string

1   source_doc_id        19 non-null     string
 2   source_doc_domain    19 non-null     string
 3   document_text        19 non-null     string
dtypes: string(4)
memory usage: 736.0 bytes
```

The `source_doc_id` column, present in both files, can be used to join them together.

#### Exploring the Dataset

There are numerous Exploratory Data Analysis (EDA) tools available. For initial exploration and testing, `dtale` is recommended due to its flexibility and user-friendly interface. Install it with:

```bash
pip install dtale
```

You can then launch a UI instance from the command line with:

```bash
dtale --parquet-path as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet
```

Please note that this project is a work in progress. Future updates will include data cleaning, removal of unnecessary columns, and additional features to enhance the usability and functionality of the project.
