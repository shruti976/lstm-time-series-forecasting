"""
LSTM time-series forecasting on the Jena Climate dataset, benchmarked against
MLP, CNN, and CNN-LSTM baselines.

Task: given the previous WINDOW hours of weather observations, predict the air
temperature (T, degC) at the next hour.

Pipeline: advanced preprocessing (wind-vector + cyclical time features) ->
temporal train/val/test split -> standardization (fit on train) -> sliding
windows -> train four architectures with early stopping and LR scheduling ->
benchmark on the held-out test set.
"""
import argparse, json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

SEED = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "jena_climate_2009_2016.csv")
ASSETS = os.path.join(HERE, "assets"); os.makedirs(ASSETS, exist_ok=True)
RESULTS = os.path.join(HERE, "results"); os.makedirs(RESULTS, exist_ok=True)

WINDOW = 48          # hours of history
TARGET = "T (degC)"
FEATURES = ["p (mbar)", "T (degC)", "Tpot (K)", "Tdew (degC)", "rh (%)",
            "VPmax (mbar)", "VPdef (mbar)", "rho (g/m**3)"]


def load_and_engineer():
    df = pd.read_csv(DATA)
    df["Date Time"] = pd.to_datetime(df["Date Time"], format="%d.%m.%Y %H:%M:%S")
    df = df.iloc[5::6].reset_index(drop=True)              # 10-min -> hourly
    # fix known sentinel errors in wind columns
    for c in ["wv (m/s)", "max. wv (m/s)"]:
        df.loc[df[c] == -9999.0, c] = 0.0
    # wind as a vector (direction is circular -> unusable as a raw number)
    wv = df["wv (m/s)"]; wd = df["wd (deg)"] * np.pi / 180.0
    df["Wx"] = wv * np.cos(wd); df["Wy"] = wv * np.sin(wd)
    # cyclical time-of-day / time-of-year signals
    ts = df["Date Time"].map(pd.Timestamp.timestamp)
    day, year = 24 * 3600, 365.2425 * 24 * 3600
    df["day_sin"] = np.sin(ts * 2 * np.pi / day); df["day_cos"] = np.cos(ts * 2 * np.pi / day)
    df["year_sin"] = np.sin(ts * 2 * np.pi / year); df["year_cos"] = np.cos(ts * 2 * np.pi / year)
    cols = FEATURES + ["Wx", "Wy", "day_sin", "day_cos", "year_sin", "year_cos"]
    return df[cols].astype("float32")


def make_windows(arr, target_idx):
    X = np.lib.stride_tricks.sliding_window_view(arr, WINDOW, axis=0)[:-1]   # (N, F, WINDOW)
    X = np.transpose(X, (0, 2, 1))                                           # (N, WINDOW, F)
    y = arr[WINDOW:, target_idx]                                            # next-hour target
    return X.astype("float32"), y.astype("float32")


def build(name, input_shape):
    inp = layers.Input(shape=input_shape)
    if name == "MLP":
        x = layers.Flatten()(inp)
        x = layers.Dense(128, activation="relu")(x); x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation="relu")(x)
    elif name == "CNN":
        x = layers.Conv1D(64, 5, activation="relu", padding="same")(inp)
        x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
    elif name == "LSTM":
        x = layers.LSTM(64, return_sequences=True)(inp)
        x = layers.LSTM(64)(x); x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
    elif name == "CNN-LSTM":
        x = layers.Conv1D(64, 5, activation="relu", padding="same")(inp)
        x = layers.MaxPooling1D(2)(x)
        x = layers.LSTM(64)(x); x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)
    m = models.Model(inp, out, name=name)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    return m


def main(epochs, batch):
    df = load_and_engineer()
    n = len(df); tr, va = int(n * 0.7), int(n * 0.85)
    target_idx = list(df.columns).index(TARGET)
    train, val, test = df.values[:tr], df.values[tr:va], df.values[va:]
    mean, std = train.mean(0), train.std(0)
    norm = lambda a: (a - mean) / std
    Xtr, ytr = make_windows(norm(train), target_idx)
    Xva, yva = make_windows(norm(val), target_idx)
    Xte, yte = make_windows(norm(test), target_idx)
    t_mean, t_std = mean[target_idx], std[target_idx]
    print(f"train {Xtr.shape}  val {Xva.shape}  test {Xte.shape}")

    cbs = [callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)]
    results, histories, preds = {}, {}, {}
    for name in ["MLP", "CNN", "LSTM", "CNN-LSTM"]:
        print(f"\n=== {name} ===")
        m = build(name, Xtr.shape[1:])
        h = m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs,
                  batch_size=batch, callbacks=cbs, verbose=2)
        p = m.predict(Xte, verbose=0).ravel() * t_std + t_mean
        truth = yte * t_std + t_mean
        mae = float(np.mean(np.abs(p - truth)))
        rmse = float(np.sqrt(np.mean((p - truth) ** 2)))
        results[name] = {"MAE_degC": round(mae, 4), "RMSE_degC": round(rmse, 4),
                         "params": int(m.count_params()), "epochs_run": len(h.history["loss"])}
        histories[name] = h.history; preds[name] = p
        print(f"{name}: MAE={mae:.3f} degC  RMSE={rmse:.3f} degC")

    truth = yte * t_std + t_mean
    pd.DataFrame(results).T.to_csv(os.path.join(RESULTS, "metrics.csv"))
    json.dump(results, open(os.path.join(RESULTS, "metrics.json"), "w"), indent=2)
    make_figures(results, histories, preds, truth)
    print("\nSaved metrics + figures.")
    print(json.dumps(results, indent=2))


def make_figures(results, histories, preds, truth):
    names = list(results.keys())
    colors = {"MLP": "#9aa7b8", "CNN": "#6f9bd8", "LSTM": "#56a98c", "CNN-LSTM": "#dca06a"}

    # 1) benchmark bars (MAE + RMSE)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    x = np.arange(len(names)); w = 0.38
    ax.bar(x - w/2, [results[n]["MAE_degC"] for n in names], w, label="MAE (degC)",
           color=[colors[n] for n in names])
    ax.bar(x + w/2, [results[n]["RMSE_degC"] for n in names], w, label="RMSE (degC)",
           color=[colors[n] for n in names], alpha=0.55)
    for i, n in enumerate(names):
        ax.text(i - w/2, results[n]["MAE_degC"], f"{results[n]['MAE_degC']:.2f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("error (degC)")
    ax.set_title("Test-set forecasting error by model (lower is better)")
    ax.legend(); ax.grid(axis="y", alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "benchmark.png")); plt.close(fig)

    # 2) validation-loss curves
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    for n in names:
        ax.plot(histories[n]["val_loss"], label=n, color=colors[n], lw=2)
    ax.set_xlabel("epoch"); ax.set_ylabel("validation loss (MSE)")
    ax.set_title("Training: validation loss with early stopping + LR scheduling")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "training_curves.png")); plt.close(fig)

    # 3) LSTM predictions vs actual (a slice of the test set)
    s = slice(200, 440)
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=150)
    ax.plot(truth[s], label="actual", color="#1a2230", lw=2)
    ax.plot(preds["LSTM"][s], label="LSTM forecast", color="#56a98c", lw=2, ls="--")
    ax.set_xlabel("hours (test set)"); ax.set_ylabel("temperature (degC)")
    ax.set_title("LSTM next-hour temperature forecast vs. actual")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "lstm_predictions.png")); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    main(epochs=1 if a.smoke else a.epochs, batch=a.batch)
