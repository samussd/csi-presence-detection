
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SUBCARRIER_INDEX = 16
WINDOW_SIZE = 33
TEST_SIZE = 0.2
RANDOM_SEED = 42

def read_csi_csv(filepath):
    df = pd.read_csv(filepath)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df = df.astype(str).apply(lambda col: col.str.strip("()").map(complex))
    return df.to_numpy()

def extract_windows_from_file(filepath, subcarrier_idx=SUBCARRIER_INDEX):
    try:
        csi = read_csi_csv(filepath)
        signal = np.real(csi[:, subcarrier_idx])
        n_windows = len(signal) // WINDOW_SIZE
        return [signal[i*WINDOW_SIZE:(i+1)*WINDOW_SIZE] for i in range(n_windows)]
    except Exception as e:
        print(f"[!] Erro ao processar {filepath}: {e}")
        return []

def load_all_data(data_path):
    X, y = [], []
    vazio_path = os.path.join(data_path, "scans_vazio")
    for f in sorted(os.listdir(vazio_path)):
        if f.endswith(".csv"):
            full_path = os.path.join(vazio_path, f)
            windows = extract_windows_from_file(full_path)
            X.extend(windows)
            y.extend([0] * len(windows))

    cheio_path = os.path.join(data_path, "scans_cheio")
    for participant in sorted(os.listdir(cheio_path)):
        participant_path = os.path.join(cheio_path, participant)
        if os.path.isdir(participant_path):
            for f in sorted(os.listdir(participant_path)):
                if f.endswith(".csv"):
                    full_path = os.path.join(participant_path, f)
                    windows = extract_windows_from_file(full_path)
                    X.extend(windows)
                    y.extend([1] * len(windows))

    print(f"==> Total de janelas coletadas: {len(X)}")
    return np.array(X), np.array(y)

def preprocess_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)
