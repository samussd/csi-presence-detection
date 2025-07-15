
import os
from data_loader import load_all_data, preprocess_data, read_csi_csv, SUBCARRIER_INDEX
from model import build_model
from utils import plot_training_history, plot_example_signal
from tensorflow.keras.callbacks import EarlyStopping

DADOS_PATH = "./dados_reduzido"

print("==> Carregando e processando dados...")
X, y = load_all_data(DADOS_PATH)
X_train, X_test, y_train, y_test = preprocess_data(X, y)

print("==> Treinando modelo...")
model = build_model()
es = EarlyStopping(patience=5, restore_best_weights=True)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=50,
    batch_size=32,
    callbacks=[es],
    verbose=1
)

loss, acc = model.evaluate(X_test, y_test)
print(f"✅ Accuracy final no conjunto de teste: {acc:.4f}")

plot_training_history(history)

exemplo_path = os.path.join(DADOS_PATH, "scans_vazio", "001.csv")
raw_data = read_csi_csv(exemplo_path)
signal = raw_data[:33, SUBCARRIER_INDEX].real
plot_example_signal(signal)
