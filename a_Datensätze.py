# a_Datensätze.py

import os
import sys
import pickle
import time
import json

import aa_detection_compression as detection_module
from aa_utils import set_seed

# Optional: für Memory-Check
try:
    import psutil
    USE_PSUTIL = True
except ImportError:
    USE_PSUTIL = False


def run_info_script():
    """
    Führt nacheinander das Raha-Detection-System (ohne Kompression) 
    auf verschiedenen Datensätzen aus und extrahiert:
      - Name des Datensatzes
      - #Spalten, #Zeilen
      - Feature-Dimension je Spalte (nach remove-identical aber ohne PCA/AE)
      - (optional) Speicherusage
    """

    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets"))
    # Liste von Datensätzen
    my_datasets = [
        {
            "name": "beers",
            "path": os.path.join(base_path, "beers", "dirty.csv"),
            "clean_path": os.path.join(base_path, "beers", "clean.csv")
        },
        {
            "name": "flights",
            "path": os.path.join(base_path, "flights", "dirty.csv"),
            "clean_path": os.path.join(base_path, "flights", "clean.csv")
        },
        {
            "name": "hospital",
            "path": os.path.join(base_path, "hospital", "dirty.csv"),
            "clean_path": os.path.join(base_path, "hospital", "clean.csv")
        },
        {
            "name": "movies_1",
            "path": os.path.join(base_path, "movies_1", "dirty.csv"),
            "clean_path": os.path.join(base_path, "movies_1", "clean.csv")
        },
        {
            "name": "rayyan",
            "path": os.path.join(base_path, "rayyan", "dirty.csv"),
            "clean_path": os.path.join(base_path, "rayyan", "clean.csv")
        },
        {
            "name": "toy",
            "path": os.path.join(base_path, "toy", "dirty.csv"),
            "clean_path": os.path.join(base_path, "toy", "clean.csv")
        },
        
    ]

    results = []

    # Seed setzen
    set_seed(42)

    for ds in my_datasets:
        ds_name = ds["name"]
        print(f"\n=== Datensatz: {ds_name} ===")

        # 1) Detection-Objekt anlegen
        app = detection_module.Detection()
        app.VERBOSE = False
        app.SAVE_RESULTS = True
        app.compression_method = "none"  # Wichtig: keine Kompression
        app.latent_dim = 0  # irrelevant bei "none"

        # Optionale Memory-Abfrage (nur wenn psutil installiert)
        mem_before = None
        if USE_PSUTIL:
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

        # 2) Lauf starten
        t0 = time.time()
        detected_cells = app.run(ds)
        runtime = time.time() - t0

        mem_after = None
        if USE_PSUTIL:
            mem_after = process.memory_info().rss / (1024 * 1024)  # in MB

        # 3) Pickle-Datei mit d-Objekt laden
        results_folder = os.path.join(
            os.path.dirname(ds["path"]), 
            "raha-baran-results-" + ds_name,
            "error-detection"
        )
        dset_pkl = os.path.join(results_folder, "detection.dataset")

        if not os.path.exists(dset_pkl):
            print(f"  Keine detection.dataset unter {dset_pkl} gefunden.")
            continue

        with open(dset_pkl, "rb") as f:
            d_obj = pickle.load(f)

        # 4) Zeilen & Spalten
        n_rows = d_obj.dataframe.shape[0]
        n_cols = d_obj.dataframe.shape[1]

        # 5) Feature-Dimension pro Spalte
        feature_dims_per_column = []
        for j in range(n_cols):
            # Jede Spalte hat ein NumPy-Array (Zeilen, dimension)
            # => shape[1] = generierte Feature-Dimension ohne Kompression
            arr = d_obj.column_features[j]
            dim = arr.shape[1] if arr is not None else 0
            feature_dims_per_column.append(dim)

        # 6) (Optional) Memory-Differenz ermitteln
        mem_usage_str = "N/A"
        if USE_PSUTIL and mem_before is not None and mem_after is not None:
            mem_diff_mb = mem_after - mem_before
            mem_usage_str = f"{mem_diff_mb:.1f} MB (diff)"

        # 7) Infos zusammenfassen
        info_dict = {
            "dataset_name": ds_name,
            "num_rows": n_rows,
            "num_cols": n_cols,
            "feature_dims_per_col": feature_dims_per_column,
            "approx_memory_usage_mb": mem_usage_str,  # optional
            "runtime_secs": f"{runtime:.2f} s",
        }
        results.append(info_dict)

        # Ausgabe
        print(f"  -> Rows={n_rows}, Cols={n_cols}, FeatureDims={feature_dims_per_column}")
        print(f"     Memory={mem_usage_str}, Runtime={runtime:.2f}s")

    # Abschließende Zusammenfassung 
    print("\n=== Zusammenfassung ===")
    for item in results:
        print(item)


if __name__ == "__main__":
    run_info_script()
