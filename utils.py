
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='Acurácia Treino')
    plt.plot(history.history['val_accuracy'], label='Acurácia Validação')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.title('Desempenho do Treinamento')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_example_signal(signal):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, linewidth=2.5, color='black')
    plt.title(f"Sinal de 1 subportadora (33 amostras = 1 segundo)")
    plt.xlabel("Amostra no tempo")
    plt.ylabel("Intensidade do sinal (parte real)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
