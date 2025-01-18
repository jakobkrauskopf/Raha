##############################################
# aaa_experiment_RUNTIME.py
##############################################
import time
import os
import csv
import numpy as np
import raha

from aa_detection_compression import Detection
from aa_utils import set_seed

set_seed(42)

def run_one_experiment(dataset_dict, compression_method, latent_dim=45, labeling_budget=20):
    """
    Führt einen kompletten Durchlauf von Raha (Detection.run) mit der angegebenen Kompressionsmethode durch.
    Misst die Zeiten der Hauptschritte und gibt sie zurück (inkl. Precision, Recall, F1).
    """
    # Seed setzen
    # set_seed(42)

    # Erzeuge eine Instanz der Detection-Klasse
    app = Detection()
    # Passe die gewünschten Parameter an
    app.compression_method = compression_method
    app.latent_dim = latent_dim
    app.LABELING_BUDGET = labeling_budget

    # ---------------------Initialisierung--------------------
    t0 = time.time()
    d = app.initialize_dataset(dataset_dict)
    t_init = time.time() - t0

    # --------------------Strategien laufen lassen-----------
    t0 = time.time()
    app.run_strategies(d)
    t_strategies = time.time() - t0

    # --------------------Feature-Generierung----------------
    t0 = time.time()
    app.generate_features(d) 
    t_features = time.time() - t0

    # --------------------Clustering-------------------------
    t0 = time.time()
    app.build_clusters(d)
    t_clustering = time.time() - t0

    # --------------Labeling-Schleife (Sampling etc.)--------
    t0 = time.time()
    while len(d.labeled_tuples) < app.LABELING_BUDGET:
        app.sample_tuple(d)
        if d.has_ground_truth:
            app.label_with_ground_truth(d)
    t_labeling = time.time() - t0

    # --------------------Propagating Labels-----------------
    t0 = time.time()
    app.propagate_labels(d)
    t_propagation = time.time() - t0

    # --------------------Klassifikation----------------------
    t0 = time.time()
    app.predict_labels(d)
    t_classification = time.time() - t0

    # --------------------Speichern (optional)---------------
    t0 = time.time()
    if app.SAVE_RESULTS:
        app.store_results(d)
    t_store = time.time() - t0

    # Evaluation (Precision, Recall, F1)
    data = raha.dataset.Dataset(dataset_dict)
    p, r, f1 = data.get_data_cleaning_evaluation(d.detected_cells)[:3]

    # Zeilenweises Dictionary mit den Zeiten und Metriken
    timing_dict = {
        "init": t_init,
        "strategies": t_strategies,
        "features": t_features,
        "clustering": t_clustering,
        "labeling": t_labeling,
        "propagation": t_propagation,
        "classification": t_classification,
        "store": t_store,
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1, 4)
    }

    return timing_dict


def main():
    # Wir legen die vier gewünschten Datensätze fest:
    dataset_dicts = [
        {
            "name": "beers",
            "path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "beers", "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "beers", "clean.csv"))
        },
        {
            "name": "flights",
            "path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "flights", "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "flights", "clean.csv"))
        },
        {
            "name": "movies_1",
            "path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "movies_1", "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "movies_1", "clean.csv"))
        },
        {
            "name": "rayyan",
            "path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "rayyan", "dirty.csv")),
            "clean_path": os.path.abspath(os.path.join(
                "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
                "rayyan", "clean.csv"))
        },
    ]

    # Hier legen wir die Methoden fest, die getestet werden sollen
    methods = ["none", "pca", "autoencoder"]

    # Wir legen fest, welchen latent_dim wir verwenden wollen
    latent_dim = 0.5

    # Festes Label-Budget
    labeling_budget = 20

    # Dateiname, in den wir alles schreiben
    output_csv = "experiment_results_all.csv"

    # CSV vorbereiten und Header schreiben
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        fieldnames = [
            "dataset", "method", "latent_dim",
            "init", "strategies", "features", "clustering", "labeling", 
            "propagation", "classification", "store",
            "precision", "recall", "f1"
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        # Jetzt iterieren wir über alle Datensätze und Methoden
        for ds in dataset_dicts:
            ds_name = ds["name"]
            for method in methods:
                print(f"\n===== STARTE EXPERIMENT: Datensatz={ds_name}, Methode={method}, latent_dim={latent_dim} =====")
                
                # Rufe die unveränderte run_one_experiment-Funktion auf
                timings = run_one_experiment(
                    dataset_dict=ds,
                    compression_method=method,
                    latent_dim=latent_dim,
                    labeling_budget=labeling_budget
                )

                # Wir wollen das Ergebnis in die CSV schreiben
                row_to_write = {
                    "dataset": ds_name,
                    "method": method,
                    "latent_dim": latent_dim,
                    "init": timings["init"],
                    "strategies": timings["strategies"],
                    "features": timings["features"],
                    "clustering": timings["clustering"],
                    "labeling": timings["labeling"],
                    "propagation": timings["propagation"],
                    "classification": timings["classification"],
                    "store": timings["store"],
                    "precision": timings["precision"],
                    "recall": timings["recall"],
                    "f1": timings["f1"]
                }
                writer.writerow(row_to_write)

                # Ausgabe zum Debuggen
                print(f"Ergebnis für {ds_name} / {method}: {timings}")


if __name__ == "__main__":
    main()
