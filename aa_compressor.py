# aa_compressor.py

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler # normalize
import torch
from pythae.pipelines import TrainingPipeline
from pythae.models import AE, AEConfig
from pythae.trainers import BaseTrainerConfig
import pickle
import os
from aa_utils import set_seed  # Importiere set_seed aus utils.py
set_seed(42)


def compress_features(features, method, ratio):
    """
    Diese Funktion wählt die Kompressionsmethode basierend auf dem angegebenen Parameter.
    """

    # Berechne Ziel-Dimension
    latent_dim = int(ratio * features.shape[1])
    # Zur Sicherheit mindestens 1
    if latent_dim < 1:
        latent_dim = 1

    if method == "autoencoder":
        #latent_dim = int(0.003 * features.shape[1])
        #print("Kompressing into: "+str(latent_dim))
        return train_autoencoder(features, latent_dim)
    elif method == "pca":
        #latent_dim = int(1 * features.shape[1])
        #print("compressing into: "+str(latent_dim))
        return train_pca(features, latent_dim)
    elif method == "none":
        return features  # Keine Komprimierung



def train_autoencoder(features, latent_dim):
    set_seed(42)
    """
    Trainiert einen Autoencoder auf den Eingabe-Features.
    """

    # Train-Test-Split
    train_data, eval_data = train_test_split(features, test_size=0.2, random_state=42) # 0.2
    
    # Konfiguration für das Training
    training_config = BaseTrainerConfig(
        output_dir='my_model',
        num_epochs=50,
        learning_rate=1e-3,
        per_device_train_batch_size=200,
        per_device_eval_batch_size=200,
        random_seed=42,
        steps_saving=None,
        optimizer_cls="AdamW",
        optimizer_params={"weight_decay": 0.05, "betas": (0.91, 0.995)},
        scheduler_cls="ReduceLROnPlateau",
        scheduler_params={"patience": 5, "factor": 0.5}
    )

    # Definiere AE Config
    INPUT_DIM = features.shape[1] # Eingabe Dim
    ae_config = AEConfig(input_dim=(INPUT_DIM,), latent_dim=latent_dim)

    # Erstelle Modell und Trainingspipeline
    model = AE(model_config=ae_config)
    pipeline = TrainingPipeline(training_config=training_config, model=model)

    # Trainiere das Modell
    pipeline(train_data=train_data, eval_data=eval_data)

    # Speichere das Modell zur späteren Verwendung
    # compressor_path = os.path.join(training_config.output_dir, "autoencoder.pkl")
    # with open(compressor_path, 'wb') as f:
    #     pickle.dump(model, f)
    
    # Konvertiere Features in Torch Tensor und generiere die latente Repräsentation
    features_tensor = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        latent_representation = model.encoder(features_tensor).embedding
    
    # Stelle sicher, dass die Ausgabe ein 2D-Numpy-Array ist
    return latent_representation.numpy()

def load_compressor(path="my_model/autoencoder.pkl"):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def train_pca(features, latent_dim):
    """
    Führt eine PCA zur Dimensionsreduktion durch.
    """
    pca = PCA(n_components=latent_dim)
    if features.shape[1] > 1:  # Falls genügend Dimensionen vorhanden sind
        return pca.fit_transform(features)
    return features  # Keine Reduktion, wenn nur eine Dimension vorhanden
