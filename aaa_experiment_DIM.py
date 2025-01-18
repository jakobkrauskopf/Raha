# aaa_experiment_DIM.py

import os
import csv
import numpy as np

from raha.dataset import Dataset
from aa_detection_compression import Detection 

def run_experiment_on_single_dataset(dataset_dict, ratios, methods, label_budget, output_csv):
    """
    Führt ein Experiment für alle Datensätze und diverse Kompressions-Einstellungen durch.

    Parameter:
    -----------
    dataset_dict : dict
        Ein Datensatz-Dictionary im Raha-Format: 
        {
            "name": str, 
            "path": str (Pfad zur dirty.csv),
            "clean_path": str (Pfad zur clean.csv)
        }
    ratios : list of float
        Liste von Kompressionsraten (z. B. [0.1, 0.2, 0.3, ...]).
    methods : list of str
        Liste der Kompressionsmethoden (z. B. ["none", "pca", "autoencoder"]).
    label_budget : int
        Anzahl der zu labelnden Tupel (LABELING_BUDGET).
    output_csv : str
        Pfad zur Ergebnis-CSV-Datei.
    """

    # Datei zum Schreiben öffnen (bzw. erstellen) und Header schreiben
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Dataset", "Method", "Ratio", "Precision", "Recall", "F1"])

        dataset_name = dataset_dict["name"]
        print(f"\n=== STARTE EXPERIMENT FÜR DATASET: {dataset_name} ===")

        # Für jede Ratio
        for ratio in ratios:
            # Für jede Kompressionsmethode
            for method in methods:
                # Detection-Instanz erzeugen
                app = Detection()
                # LABEL_BUDGET festsetzen
                app.LABELING_BUDGET = label_budget
                # Kompressionsmethoden-Parameter
                app.compression_method = method
                app.latent_dim = ratio

                print(f"\n--- {dataset_name} | method={method} | ratio={ratio} ---")
                # Pipeline ausführen
                detection_dictionary = app.run(dataset_dict)

                # Evaluierung
                data = Dataset(dataset_dict)
                p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]

                # Ergebnisse ausgeben
                print(f"Precision = {p:.4f} | Recall = {r:.4f} | F1 = {f:.4f}")

                # In CSV schreiben
                writer.writerow([dataset_name, method, ratio, p, r, f])

    print(f"\nExperiment abgeschlossen! Ergebnisse liegen in: {output_csv}")


if __name__ == "__main__":
    # =============================================================================
    #              HIER DIE LISTEN FÜR RATIOS, METHODS, LABEL-BUDGET FESTLEGEN
    # =============================================================================
    RATIOS = [round(i, 1) for i in np.arange(0.1, 1.0, 0.1)]
    METHODS = ["none", "pca", "autoencoder"]
    LABEL_BUDGET = 20

    # =============================================================================
    #                           1) BEERS DATASET
    # =============================================================================
    dataset_name = "beers"
    beers_dict = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "dirty.csv"
        )),
        "clean_path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "clean.csv"
        ))
    }
    OUTPUT_CSV_BEERS = "experiment_results_beers.csv"
    run_experiment_on_single_dataset(
        dataset_dict=beers_dict,
        ratios=RATIOS,
        methods=METHODS,
        label_budget=LABEL_BUDGET,
        output_csv=OUTPUT_CSV_BEERS
    )

    # =============================================================================
    #                           2) FLIGHTS DATASET
    # =============================================================================
    dataset_name = "flights"
    flights_dict = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "dirty.csv"
        )),
        "clean_path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "clean.csv"
        ))
    }
    OUTPUT_CSV_FLIGHTS = "experiment_results_flights.csv"
    run_experiment_on_single_dataset(
        dataset_dict=flights_dict,
        ratios=RATIOS,
        methods=METHODS,
        label_budget=LABEL_BUDGET,
        output_csv=OUTPUT_CSV_FLIGHTS
    )

    # =============================================================================
    #                           3) MOVIES_1 DATASET
    # =============================================================================
    dataset_name = "movies_1"
    movies_1_dict = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "dirty.csv"
        )),
        "clean_path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "clean.csv"
        ))
    }
    OUTPUT_CSV_MOVIES = "experiment_results_movies_1.csv"
    run_experiment_on_single_dataset(
        dataset_dict=movies_1_dict,
        ratios=RATIOS,
        methods=METHODS,
        label_budget=LABEL_BUDGET,
        output_csv=OUTPUT_CSV_MOVIES
    )

    # =============================================================================
    #                           4) RAYYAN DATASET
    # =============================================================================
    dataset_name = "rayyan"
    rayyan_dict = {
        "name": dataset_name,
        "path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "dirty.csv"
        )),
        "clean_path": os.path.abspath(os.path.join(
            "/Users/jakobmac/Desktop/RahaProjekt/raha/datasets", 
            dataset_name, 
            "clean.csv"
        ))
    }
    OUTPUT_CSV_RAYYAN = "experiment_results_rayyan.csv"
    run_experiment_on_single_dataset(
        dataset_dict=rayyan_dict,
        ratios=RATIOS,
        methods=METHODS,
        label_budget=LABEL_BUDGET,
        output_csv=OUTPUT_CSV_RAYYAN
    )
