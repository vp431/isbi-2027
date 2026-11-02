# Rest and photic EEG for dementia classification

Code for the ISBI 2027 paper *A study of the impact of photic stimulation EEG on dementia classification*.

## Data

Download the two open datasets from OpenNeuro:

- D1, resting EEG with eyes closed: [ds004504](https://openneuro.org/datasets/ds004504)
- D2, photic stimulation EEG with eyes open: [ds006036](https://openneuro.org/datasets/ds006036)

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Extract the features first. This writes `data/features.csv`:

```bash
python src/features.py --d1 /path/to/ds004504 --d2 /path/to/ds006036
```

Then run any experiment from the repository root. Each one writes a JSON file to `results/`.

| Script | Experiment |
|---|---|
| `src/main_table.py` | Main classification results for D1, D2, Concat and Mean |
| `src/classifiers.py` | Eight classifiers on D1, D2 and Concat |
| `src/nested.py` | Nested selection of the number of features |
| `src/subject_level.py` | Subject-level DeLong tests |
| `src/pairing.py` | Pairing control and subject identification |
| `src/biomarkers.py` | Effect sizes and top-ranked features |
| `src/figures.py` | Paper figures, after `classifiers.py` and `pairing.py` |
