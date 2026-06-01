# Data Folder

Place the toddler autism CSV file in this folder with the following filename:

```text
Toddler Autism dataset July 2018.csv
```

The classification scripts use this path by default:

```text
data/Toddler Autism dataset July 2018.csv
```

If the dataset is stored elsewhere, pass the path manually:

```bash
python toddler_autism_classification/run_all_models.py --csv-path "path/to/Toddler Autism dataset July 2018.csv"
```

The facial emotion recognition image dataset is not stored in this folder by default. See `emotion_webapp/README.md` for the expected image dataset structure.

