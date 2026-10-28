"""Extrai MFCCs + Deltas com alinhamento vocal (VAD) e Data Augmentation para robustez máxima.

Treina rede neural imune a variações de timing, volume e ruído.
"""
import os
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, accuracy_score
from tensorflow import keras
from tensorflow.keras import layers
from vox_config import COMMANDS, SAMPLE_RATE, DURATION, N_MFCC, MODEL_PATH, LABELS_PATH

DATASET = "dataset"
SCALER_PATH = "modelos/scaler.npz"
RESULTS = "resultados"


def align_voice(samples: np.ndarray, sr: int, target_duration: float = DURATION) -> np.ndarray:
    """Centraliza a porção audível (voz) na janela alvo de duração para tolerar atrasos na fala."""
    target_samples = int(sr * target_duration)
    frame_len = int(sr * 0.02)
    hop = int(sr * 0.01)
    
    rms = librosa.feature.rms(y=samples, frame_length=frame_len, hop_length=hop)[0]
    thresh = max(0.005, np.max(rms) * 0.12)
    active = np.where(rms > thresh)[0]
    
    if len(active) == 0:
        if len(samples) >= target_samples:
            return samples[:target_samples]
        return np.pad(samples, (0, target_samples - len(samples)))
        
    start = max(0, active[0] * hop - int(sr * 0.05))
    end = min(len(samples), active[-1] * hop + int(sr * 0.05))
    voice = samples[start:end]
    
    if len(voice) >= target_samples:
        return voice[:target_samples]
        
    pad_left = (target_samples - len(voice)) // 2
    pad_right = target_samples - len(voice) - pad_left
    return np.pad(voice, (pad_left, pad_right))


def extract_features(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extrai 52 características: MFCC (13) + Delta (13) com média e desvio padrão."""
    audio = align_voice(audio, sr)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta = librosa.feature.delta(mfcc)
    return np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        delta.mean(axis=1), delta.std(axis=1)
    ]).astype("float32")


def augment_audio(audio: np.ndarray, sr: int) -> list:
    """Gera variações acústicas realistas (ganho, leve ruído e pitch) para o modelo generalizar."""
    variants = [audio]
    
    # 1. Ganho mais baixo (falou mais baixo / longe)
    variants.append(audio * 0.75)
    
    # 2. Ganho mais alto (falou mais perto)
    variants.append(np.clip(audio * 1.25, -1.0, 1.0))
    
    # 3. Leve ruído de fundo (sala de aula / computador)
    noise = np.random.randn(len(audio)) * 0.004
    variants.append(audio + noise)
    
    # 4. Leve time-shift (voz 0.1s adiantada/atrasada)
    shift = int(sr * 0.08)
    variants.append(np.roll(audio, shift))
    variants.append(np.roll(audio, -shift))
    
    return variants


def main():
    os.makedirs("modelos", exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    
    X, y = [], []
    print("Processando áudios com alinhamento vocal e Data Augmentation...")
    
    for label, command in enumerate(COMMANDS):
        folder = os.path.join(DATASET, command)
        files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(".wav")
        ]) if os.path.isdir(folder) else []
        
        if len(files) < 5:
            raise SystemExit(f"A classe '{command}' tem poucas amostras ({len(files)}).")
            
        print(f" -> Classe '{command}': {len(files)} áudios originais")
        for path in files:
            try:
                audio, sr = librosa.load(path, sr=SAMPLE_RATE, mono=True)
                for var in augment_audio(audio, sr):
                    X.append(extract_features(var, sr))
                    y.append(label)
            except Exception as exc:
                print(f"Aviso ao ler {path}: {exc}")

    X = np.asarray(X, dtype="float32")
    y = np.asarray(y, dtype="int64")
    print(f"\nTotal de amostras aumentadas para treino: {len(X)} (52 features cada)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype("float32")
    X_test = scaler.transform(X_test).astype("float32")
    np.savez(SCALER_PATH, mean=scaler.mean_, scale=scaler.scale_)

    # Rede Neural otimizada com 52 entradas e regularização
    model = keras.Sequential([
        layers.Input(shape=(52,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.20),
        layers.Dense(len(COMMANDS), activation="softmax")
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.20,
        epochs=120,
        batch_size=16,
        callbacks=[early_stop],
        verbose=1
    )

    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)
    acc = accuracy_score(y_test, pred)
    print(f"\n==========================================")
    print(f"Acurácia no conjunto de teste: {acc:.3f} ({acc*100:.1f}%)")
    print(f"==========================================\n")
    report = classification_report(y_test, pred, target_names=COMMANDS, zero_division=0)
    print(report)
    with open(os.path.join(RESULTS, "avaliacao.txt"), "w", encoding="utf-8") as f:
        f.write(f"Acurácia no teste: {acc:.4f} ({acc*100:.1f}%)\n\n")
        f.write(report)
        f.write("\n\nObservação: o resultado depende da quantidade e diversidade das gravações.\n")

    model.save(MODEL_PATH)
    weights = {f'w_{i}': w for i, w in enumerate(model.get_weights())}
    np.savez(os.path.join("modelos", "weights.npz"), **weights)
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(COMMANDS, f, ensure_ascii=False, indent=2)

    # Gráfico de evolução
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")
    plt.title("Evolução da Acurácia — VOX Robust Neural Net")
    plt.xlabel("Época")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(RESULTS, "acuracia.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Matriz de Confusão
    disp = ConfusionMatrixDisplay.from_predictions(y_test, pred, display_labels=COMMANDS,
                                                  cmap="Blues", normalize="true")
    disp.ax_.set_title("Matriz de Confusão Normalizada")
    plt.savefig(os.path.join(RESULTS, "matriz_confusao.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Treinamento robusto finalizado e modelo salvo com sucesso!")


if __name__ == "__main__":
    main()
