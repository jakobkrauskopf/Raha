# aaa_experiment_LABEL.py

import os
import csv
import numpy as np
import raha

# Detection-Klasse importieren
from aa_detection_compression import Detection

def run_detection(dataset_dictionary, label_budget, latent_dim, method):
    """
    Führt die Fehlererkennung mit der angegebenen Detection-Konfiguration durch.
    Gibt (Precision, Recall, F1) zurück.
    """
    # Erzeuge Detection-Objekt
    app = Detection()
    app.LABELING_BUDGET = label_budget
    app.compression_method = method

    # Wenn method="none", dann latent_dim ignorieren (hier None = keine Kompression)
    if method == "none":
        app.latent_dim = 0
    else:
        app.latent_dim = latent_dim

    # Starte die Pipeline
    detection_dictionary = app.run(dataset_dictionary)

    # Evaluation
    data = raha.dataset.Dataset(dataset_dictionary)
    p, r, f = data.get_data_cleaning_evaluation(detection_dictionary)[:3]

    return p, r, f


def run_label_budget_experiment(dataset_dicts, methods, label_budgets, latent_dim, output_csv):
    """
    Führt ein Experiment durch, in dem für jeden Datensatz und jede Methode 
    verschiedene label_budgets (1 bis X) ausprobiert werden.
    
    Parameter:
    -----------
    dataset_dicts : list
        Liste von Datensatz-Dictionaries im Raha-Format.
    methods : list
        Liste der Methoden (z.B. ['none', 'pca', 'autoencoder']).
    label_budgets : list of int
        Liste von Label-Budgets (z.B. [1,2,3,...,20]).
    latent_dim : float
        Das Ratio oder die gewünschte Latent-Dimension (z.B. 0.5).
    output_csv : str
        Pfad zur Ergebnis-CSV-Datei.
    """

    # CSV öffnen und Header schreiben
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Kopfzeile definieren
        writer.writerow([
            "dataset", 
            "label_budget", 
            "method", 
            "latent_dim", 
            "precision", 
            "recall", 
            "f1_score"
        ])

        for ds in dataset_dicts:
            ds_name = ds["name"]
            for lb in label_budgets:
                for method in methods:
                    print(f"\nStarte Erkennung: Dataset={ds_name}, label_budget={lb}, method={method}, latent_dim={latent_dim}")
                    p, r, f = run_detection(ds, lb, latent_dim, method)

                    # Resultat in CSV schreiben
                    writer.writerow([
                        ds_name,
                        lb,
                        method,
                        latent_dim if method != "none" else 0, 
                        round(p, 4),
                        round(r, 4),
                        round(f, 4)
                    ])

                    print(f"  => Ergebnis: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

    print(f"\nExperiment abgeschlossen! Ergebnisse liegen in: {output_csv}")


if __name__ == "__main__":
    # ------------------------------------------------------------------------
    # Liste von Datensätzen:
    # ------------------------------------------------------------------------
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

    # ------------------------------------------------------------------------
    # Methoden, die getestet werden sollen
    # ------------------------------------------------------------------------
    methods = ['none', 'pca', 'autoencoder']

    # ------------------------------------------------------------------------
    # Label-Budgetst (hier bis 20)
    # ------------------------------------------------------------------------
    label_budgets = list(range(1, 21))  # [1, 2, 3, ..., 20] 21

    # ------------------------------------------------------------------------
    # Latent-Dimension (z.B. ratio = 0.5) 
    # Für "none" verwenden wir automatisch 0.
    # ------------------------------------------------------------------------
    latent_dim = 0.5

    # ------------------------------------------------------------------------
    # Ergebnis-CSV festlegen
    # ------------------------------------------------------------------------
    output_csv = "label_budget_experiment_results.csv"

    # ------------------------------------------------------------------------
    # Experiment starten
    # ------------------------------------------------------------------------
    run_label_budget_experiment(
        dataset_dicts=dataset_dicts,
        methods=methods,
        label_budgets=label_budgets,
        latent_dim=latent_dim,
        output_csv=output_csv
    )
