# Week 3: Trace processing and workload characterization

Week 3 turned TraceLab's raw logs into a pause dataset for our cost model, and described what those pauses look like.

Start with **`Week3_Trace_Processing_Report.pdf`** for the findings, the definition of a pause, and what each workstream should take from it.

## What's in this folder

| File | What it does |
| --- | --- |
| `Week3_Trace_Processing_Report.pdf` | The write-up: findings, definitions, limitations, expected results |
| `build_pause_table.py` | Builds `pauses.parquet`, one row per pause. **Run this first.** |
| `look_at_pauses.py` | Prints the table's columns, sample rows, and a check session |
| `look_at_session.py` | Prints one session's raw events, round by round |
| `plot_pause_cdf.py` | Finding 1: pause lengths vs. the break-even time |
| `plot_pause_by_tool.py` | Finding 2: pause length by tool |
| `plot_context_size.py` | Finding 3: idle cache sizes and the per-pause break-even time |
| `figures/` | The three figures as PNGs, ready for slides and the final report |
| `data/pauses.parquet` | The pause table itself, so you can start without setting up TraceLab |

## Using the pause table directly

If you only need the pauses (for the predictor or the simulator), you can skip the setup below and load `data/pauses.parquet`:

```python
import duckdb
pauses = duckdb.sql("SELECT * FROM 'data/pauses.parquet' WHERE usable").df()
```

Or, with pandas (needs `pyarrow` installed): `pd.read_parquet("data/pauses.parquet")`.

The file was built by `build_pause_table.py` from TraceLab release v0.0.1. If the script changes, rebuild the file and upload it again so the two stay in sync.

The pause table is derived from the [TraceLab dataset](https://github.com/uw-syfi/TraceLab) (Zhu et al., [arXiv:2606.30560](https://arxiv.org/abs/2606.30560)), which is licensed under CC BY 4.0. Credit TraceLab wherever the table or its figures are used.

The raw trace is not in this repo. It's TraceLab's public release (`syfi_coding_trace.duckdb`, about 100 MB), and the steps below regenerate everything from it in under a minute.

## How to run it (Windows PowerShell)

**1. Set up TraceLab once.** Clone it outside OneDrive or any other synced folder.

```powershell
cd $HOME
git clone https://github.com/uw-syfi/TraceLab.git
cd TraceLab
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close and reopen the terminal so it finds `uv`, then, from the TraceLab folder:

```powershell
uv sync
```

**2. Download the data.**

```powershell
curl.exe -L --fail --ssl-no-revoke -o trace\syfi_coding_trace.duckdb https://github.com/uw-syfi/TraceLab/releases/download/v0.0.1/syfi_coding_trace.duckdb
```

**3. Copy these scripts into TraceLab.**

```powershell
mkdir our_work
Copy-Item <path-to-our-repo>\week3\*.py our_work\
```

**4. Run them from the TraceLab folder, in this order.**

```powershell
uv run python our_work\build_pause_table.py
uv run python our_work\look_at_pauses.py
uv run python our_work\plot_pause_cdf.py
uv run python our_work\plot_pause_by_tool.py
uv run python our_work\plot_context_size.py
```

On Mac or Linux: install uv with `curl -LsSf https://astral.sh/uv/install.sh | sh`, use `curl` instead of `curl.exe` and drop `--ssl-no-revoke`, and use `/` instead of `\` in paths.

## Check that it worked

- `pauses.parquet` has 357,161 rows, of which 346,820 are usable.
- `look_at_pauses.py` shows the check session's pauses as 0.043, 4.56, 3.285, 5.809, 37.615, 40.533, 1.206, 94.702 and 42.993 s, then `end`.

The PDF's "How to reproduce" section lists the expected numbers for every script.