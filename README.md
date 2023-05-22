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


---