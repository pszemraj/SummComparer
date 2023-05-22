# SummComparer

> A V1 take at compiling [the summarization gauntlet](https://www.dropbox.com/sh/axu1xlscrrexy55/AADAm01-4Zs3POyHQrgbDAsda?dl=0)

**NOTE: THIS IS A WORK IN PROGRESS**

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

Files are in `as-dataset/` and are saved as `.parquet`. There are two files:

- `as-dataset/gauntlet_input_documents.parquet`: contains the input documents for the gauntlet with metadata/`id` fields as defined in `gauntlet_master_data.json`
- `as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet` contains the output summaries for the gauntlet with hyperparameters/models as columns.
  - Additionally, all summaries (rows) are mapped to their source documents (columns) by columns prefixed with `source_doc`.

The data can be loaded with `pandas`:

```python
import pandas as pd
df = pd.read_parquet('as-dataset/summary_gauntlet_dataset_mapped_src_docs.parquet')
df.info()
```

---
