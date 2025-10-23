# Classifier Statistics Reviewer

A simple PyQt6 + Pillow desktop app to quickly review image snippets produced by an AI/ML classifier and record how well it performed.

It expects your data to be organized like this:

```
output/label1/UUID.png
output/label1/UUID2.png
output/label2/UUID3.png
```

Where each subfolder under `output/` is a classifier label containing image snippets.

## Features

- Load an `output/` directory and iterate through snippets
- For each snippet, choose one of: "Valid classification", "Similar", or "Incorrect"
- Progress is continuously tracked in-app (current item is removed once labeled)
- Save progress to a JSON file at any time and resume later by loading it
- Warn to save on exit if there are unsaved changes
- Keyboard shortcuts: 1 = Valid, 2 = Similar, 3 = Incorrect
- Export statistics: overall decision counts, per-label distributions, per-label hard/soft accuracy, and summary metrics

## Install

Create a virtual environment (optional but recommended) and install dependencies:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```
python -m app.main
```

Then use File -> "Open Output Directory..." to select your `output/` folder.

To save at any time, use File -> "Save Progress...". To resume later, use File -> "Load Progress...".

To export charts and metrics based on your decisions, use File -> "Export Statistics..." and choose a folder. The app will write:
- `overall_decisions.png` — overall counts of Valid/Similar/Incorrect
- `per_label_stacked.png` — stacked counts per predicted label
- `per_label_metrics.csv` — per-label totals and hard/soft accuracy (with 95% Wilson CI for hard)
- `counts_by_label_and_selection.csv` — counts per label and selection
- `metrics.json` — overall totals and accuracies

## Notes

- Only one level of subfolders under the chosen `output/` directory is scanned, and common image extensions are supported (png, jpg, jpeg, bmp, gif, tif, tiff).
- Progress JSON stores the absolute output directory path and your decisions keyed by relative snippet paths.
- You can safely add new snippets to the folders between sessions; the app will pick them up when you load progress.
- Since there is no external ground-truth in this workflow, ROC and kappa are not computed. Instead, we provide counts and acceptance-based metrics (hard/soft accuracy) derived from your review decisions.

## License

MIT
