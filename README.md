# SummComparer

> A V1 take at compiling [the summarization gauntlet](https://www.dropbox.com/sh/axu1xlscrrexy55/AADAm01-4Zs3POyHQrgbDAsda?dl=0)

**NOTE: THIS IS A WORK IN PROGRESS**

The purpose of this project/dataset is to examine the performance of various summarization models on a variety of long documents **none of which were part of the model's training data**[^1]. The goal is to gain some insight into what generalizes "in the wild", what doesn't, and ideally why.

[^1]: As it turns out, the practical application of summarization models is **not** to summarize documents _you already know the summary of_ and present their ability to repeat them back to you with ROUGE scores as a measure of performance. Who knew?

---

## Install

```bash
pip install -r requirements.txt
```

package requirements for using the scripts in `bin/` can be installed from that directory with:

```bash
pip install -r bin/requirements.txt
```

## Usage

### Compiling the Gauntlet

Currently limited to CLI usage. Recommended order of operations:

```bash
export_gauntlet.py
map_gauntlet_files.py
build_src_df.py
```

all CLI scripts use the `fire` package for CLI generation. For more information on how to use the CLI, run:

```bash
python <script_name>.py --help
```

### Using the dataset

> **NOTE:** data is currently in a "raw" format and has not had useless columns removed or been cleaned in any way. This will be done in a future release.

Files are in `as-dataset/` and are saved as `.parquet`. There are two files, which can be thought of as two tables in a relational database:

- `as-dataset/gauntlet_input_documents.parquet`: contains the input documents for the gauntlet with metadata/`id` fields as defined in `gauntlet_master_data.json`
- `as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet` contains the output summaries for the gauntlet with hyperparameters/models as columns.
  - Additionally, all summaries (rows) are mapped to their source documents (columns) by columns prefixed with `source_doc`.

The data can be loaded with `pandas`:

```python
import pandas as pd
df = pd.read_parquet('as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet')
df.info()
```

#### Input Documents

Almost all of the information needed is in the `summary_gauntlet_dataset_mapped_src_docs.parquet`, and the `gauntlet_input_documents.parquet` is only needed if you want to look at the source documents themselves, or do some sort of analysis using their text.

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

The `source_doc_id` column is present in both files and can be used to join them together.

#### Exploring the dataset

There are many EDA tools out there, but for initial exploration and testing, I'd recommend `dtale` for it's flexibility and ease of use. It can be installed with:

```bash
pip install dtale
```

You can then spin up a UI instance from the command line with:

```bash
dtale --parquet-path as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet
```

---
