# aaa_experiment_DIM.py

import os
import csv
import time
import psutil
import numpy as np

from raha.dataset import Dataset
from aa_detection_compression import Detection 

def run_experiment_all_datasets(datasets_list, ratios, methods, label_budget, output_csv):
    """
    Führt ein Experiment für mehrere Datensätze, Kompressions-Ratios und -Methoden durch
    und schreibt alle Ergebnisse (Precision, Recall, F1, Laufzeit und Speicherdifferenz)
    in EINE gemeinsame CSV-Datei.

    Parameter:
    -----------
    datasets_list : list of dict
        Liste von Dataset-Dictionaries im Raha-Format, z.B.:
        [
           {"name": "beers", "path": ".../beers/dirty.csv", "clean_path": ".../beers/clean.csv"},
           ...
        ]
    ratios : list of float
        Liste von Kompressionsraten, z. B. [0.1, 0.2, 0.3, ...].
    methods : list of str
        Liste der Kompressionsmethoden, z. B. ["none", "pca", "autoencoder"].
    label_budget : int
        Anzahl der zu labelnden Tupel (LABELING_BUDGET).
    output_csv : str
        Pfad zur Ergebnis-CSV-Datei, in der alles gesammelt wird.
    """

    # Datei zum Schreiben eröffnen (bzw. erstellen) und den Header schreiben
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Zusätzliche Spalten für Laufzeit und Speicherverbrauch
        writer.writerow(["Dataset", "Method", "Ratio", 
                         "Precision", "Recall", "F1", 
                         "TotalRuntime_s", "MemDiff_Bytes"])

        # Iteration über alle Datensätze
        for dataset_dict in datasets_list:
            dataset_name = dataset_dict["name"]
            print(f"\n=== STARTE EXPERIMENT FÜR DATASET: {dataset_name} ===")

            # Für jede Ratio
            for ratio in ratios:
                # Für jede Kompressionsmethode
                for method in methods:
                    print(f"\n--- {dataset_name} | method={method} | ratio={ratio} ---")

                    # Detection-Instanz erzeugen
                    app = Detection()
                    # LABEL_BUDGET festsetzen
                    app.LABELING_BUDGET = label_budget
                    # Kompressionsmethoden-Parameter
                    app.compression_method = method
                    app.latent_dim = ratio

                    # Speicher- und Zeitmessung vor dem Lauf
                    process = psutil.Process(os.getpid())
                    mem_before = process.memory_info().rss
                    start_time = time.time()

                    # Pipeline ausführen
                    detection_dictionary = app.run(dataset_dict)

                    # Speicher- und Zeitmessung nach dem Lauf
                    end_time = time.time()
                    mem_after = process.memory_info().rss

                    runtime = end_time - start_time
                    mem_diff = mem_after - mem_before

                    # Evaluierung
                    data = Dataset(dataset_dict)
                    # get_data_cleaning_evaluation(...) -> (precision, recall, f1, ...)
                    p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]

                    # Ergebnisse ausgeben
                    print(f"Precision = {p:.4f} | Recall = {r:.4f} | F1 = {f:.4f} | "
                          f"Runtime = {runtime:.2f}s | MemDiff = {mem_diff} bytes")

                    # In CSV schreiben
                    writer.writerow([dataset_name, method, ratio, 
                                     p, r, f, 
                                     round(runtime, 3), mem_diff])

    print(f"\nExperiment abgeschlossen! Gesamtergebnisse in: {output_csv}")



if __name__ == "__main__":

    # ===========================================================
    #              LISTEN FÜR RATIOS, METHODS, LABEL-BUDGET
    # ===========================================================
    RATIOS = [round(i, 1) for i in np.arange(0.1, 1.0, 0.1)]  # 0.1, 0.2, ..., 0.9
    METHODS = ["none", "pca", "autoencoder"]
    LABEL_BUDGET = 20

    # ===========================================================
    #      DEFINITION DER DATENSATZ-DICTIONARIES
    # ===========================================================
    base_dir = "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets"

    beers_dict = {
        "name": "beers",
        "path": os.path.join(base_dir, "beers", "dirty.csv"),
        "clean_path": os.path.join(base_dir, "beers", "clean.csv")
    }
    flights_dict = {
        "name": "flights",
        "path": os.path.join(base_dir, "flights", "dirty.csv"),
        "clean_path": os.path.join(base_dir, "flights", "clean.csv")
    }
    movies_dict = {
        "name": "movies_1",
        "path": os.path.join(base_dir, "movies_1", "dirty.csv"),
        "clean_path": os.path.join(base_dir, "movies_1", "clean.csv")
    }
    rayyan_dict = {
        "name": "rayyan",
        "path": os.path.join(base_dir, "rayyan", "dirty.csv"),
        "clean_path": os.path.join(base_dir, "rayyan", "clean.csv")
    }

    # In einer Liste bündeln, damit wir alles gemeinsam iterieren können
    DATASETS_LIST = [
        beers_dict,
        flights_dict,
        movies_dict,
        rayyan_dict
    ]

    # ===========================================================
    #  GEMEINSAME CSV FÜR ALLE DATENSÄTZE, RATIOS UND METHODEN
    # ===========================================================
    OUTPUT_CSV = "experiment_results_DIM_all.csv"

    # Aufruf der Experiment-Funktion
    run_experiment_all_datasets(
        datasets_list=DATASETS_LIST,
        ratios=RATIOS,
        methods=METHODS,
        label_budget=LABEL_BUDGET,
        output_csv=OUTPUT_CSV
    )
