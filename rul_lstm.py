"""
Predicting Remaining Useful Life (RUL) of turbofan engines using an LSTM.

Implementation scaffold based on the workflow and hyperparameters described
in the supplied reference report. Dataset files must be downloaded separately.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"

WINDOW_LENGTH = 30
RUL_CAP = 125
RANDOM_SEED = 42
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
DROP_SENSORS = {"s1", "s5", "s6", "s10", "s16", "s18", "s19"}
FEATURE_COLUMNS = [c for c in SENSOR_COLUMNS if c not in DROP_SENSORS]
COLUMN_NAMES = ["engine_id", "cycle", "setting1", "setting2", "setting3"] + SENSOR_COLUMNS


def set_seed(seed: int = RANDOM_SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_data():
    train_path = DATA_DIR / "train_FD001.txt"
    test_path = DATA_DIR / "test_FD001.txt"
    rul_path = DATA_DIR / "RUL_FD001.txt"

    missing = [str(p) for p in (train_path, test_path, rul_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Required NASA C-MAPSS FD001 files are missing:\n  "
            + "\n  ".join(missing)
            + "\nDownload and extract C-MAPSS, then place the three files in data/."
        )

    train = pd.read_csv(train_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test = pd.read_csv(test_path, sep=r"\s+", header=None, names=COLUMN_NAMES)
    test_rul = pd.read_csv(rul_path, sep=r"\s+", header=None, names=["rul"])["rul"].to_numpy()
    return train, test, test_rul


def add_train_rul(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    last_cycle = result.groupby("engine_id")["cycle"].transform("max")
    result["RUL"] = (last_cycle - result["cycle"]).clip(upper=RUL_CAP)
    return result


def add_test_rul(frame: pd.DataFrame, final_rul: np.ndarray) -> pd.DataFrame:
    result = frame.copy()
    engine_ids = sorted(result["engine_id"].unique())
    if len(engine_ids) != len(final_rul):
        raise ValueError(
            f"Test engine count ({len(engine_ids)}) does not match "
            f"RUL_FD001.txt rows ({len(final_rul)})."
        )
    rul_by_engine = dict(zip(engine_ids, final_rul))
    last_cycle = result.groupby("engine_id")["cycle"].transform("max")
    remaining_at_last_observation = result["engine_id"].map(rul_by_engine)
    raw_rul = (last_cycle - result["cycle"]) + remaining_at_last_observation
    result["RUL"] = raw_rul.clip(upper=RUL_CAP)
    return result


def prepare_features(train: pd.DataFrame, test: pd.DataFrame):
    train = add_train_rul(train)
    test = add_test_rul(test, test.attrs["final_rul"])
    return train, test


def split_by_engine(train: pd.DataFrame):
    engine_ids = np.array(sorted(train["engine_id"].unique()))
    train_ids, val_ids = train_test_split(
        engine_ids, test_size=0.20, random_state=RANDOM_SEED
    )
    train_rows = train[train["engine_id"].isin(train_ids)].copy()
    val_rows = train[train["engine_id"].isin(val_ids)].copy()
    return train_rows, val_rows


def scale_sensor_rows(train_rows, val_rows, test_rows):
    scaler = MinMaxScaler()
    train_rows.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(train_rows[FEATURE_COLUMNS])
    val_rows.loc[:, FEATURE_COLUMNS] = scaler.transform(val_rows[FEATURE_COLUMNS])
    test_rows.loc[:, FEATURE_COLUMNS] = scaler.transform(test_rows[FEATURE_COLUMNS])
    return train_rows, val_rows, test_rows, scaler


def make_windows(frame: pd.DataFrame, window_length: int = WINDOW_LENGTH):
    xs, ys, last_rows = [], [], []
    for engine_id in sorted(frame["engine_id"].unique()):
        engine = frame[frame["engine_id"] == engine_id].sort_values("cycle")
        features = engine[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        labels = engine["RUL"].to_numpy(dtype=np.float32)
        for end in range(window_length - 1, len(engine)):
            start = end - window_length + 1
            xs.append(features[start:end + 1])
            ys.append(labels[end])
            last_rows.append(features[end])
    if not xs:
        return (
            np.empty((0, window_length, len(FEATURE_COLUMNS)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0, len(FEATURE_COLUMNS)), dtype=np.float32),
        )
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.stack(last_rows)


def make_last_test_windows(frame: pd.DataFrame, window_length: int = WINDOW_LENGTH):
    xs, ys, last_rows = [], [], []
    for engine_id in sorted(frame["engine_id"].unique()):
        engine = frame[frame["engine_id"] == engine_id].sort_values("cycle")
        if len(engine) < window_length:
            raise ValueError(
                f"Test engine {engine_id} has only {len(engine)} cycles; "
                f"{window_length} are required by this no-padding implementation."
            )
        features = engine[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        xs.append(features[-window_length:])
        ys.append(float(engine["RUL"].iloc[-1]))
        last_rows.append(features[-1])
    return np.stack(xs), np.asarray(ys, dtype=np.float32), np.stack(last_rows)


class RULLSTM(nn.Module):
    def __init__(self, input_size: int = 14):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc_out = nn.Linear(32, 1)

    def forward(self, x):
        sequence_output, _ = self.lstm(x)
        last_step = sequence_output[:, -1, :]
        x = self.dropout(last_step)
        x = self.relu(self.fc1(x))
        return self.fc_out(x).squeeze(-1)


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def nasa_score(y_true, y_pred) -> float:
    """NASA C-MAPSS asymmetric score; positive error means prediction is late."""
    errors = np.asarray(y_pred) - np.asarray(y_true)
    score = 0.0
    for error in errors:
        if error < 0:
            score += np.exp(-error / 13.0) - 1.0
        else:
            score += np.exp(error / 10.0) - 1.0
    return float(score)


def train_lstm(x_train, y_train, x_val, y_val, device):
    model = RULLSTM(input_size=len(FEATURE_COLUMNS)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train / RUL_CAP, dtype=torch.float32),
    )
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_y = torch.tensor(y_val / RUL_CAP, dtype=torch.float32, device=device)
    loader = DataLoader(train_ds, batch_size=256, shuffle=True)

    best_val_rmse = float("inf")
    best_state = None
    patience, epochs_without_improvement = 8, 0
    history = {"train_rmse": [], "val_rmse": []}

    for epoch in range(60):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            predictions = model(xb)
            loss = loss_fn(predictions, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_predictions = predict_lstm(model, x_train, device)
            val_predictions = model(val_x).detach().cpu().numpy() * RUL_CAP

        train_epoch_rmse = rmse(y_train, train_predictions)
        val_epoch_rmse = rmse(y_val, val_predictions)
        history["train_rmse"].append(train_epoch_rmse)
        history["val_rmse"].append(val_epoch_rmse)

        print(
            f"Epoch {epoch + 1:02d}/60 | "
            f"train RMSE: {train_epoch_rmse:.3f} | "
            f"validation RMSE: {val_epoch_rmse:.3f}"
        )

        if val_epoch_rmse < best_val_rmse:
            best_val_rmse = val_epoch_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping: validation RMSE did not improve.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


def predict_lstm(model, x, device, batch_size=512):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start:start + batch_size], dtype=torch.float32, device=device)
            predictions.append(model(xb).cpu().numpy() * RUL_CAP)
    return np.concatenate(predictions)


def save_plots(history, y_true, predictions, metrics):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_rmse"], label="Training RMSE")
    plt.plot(history["val_rmse"], label="Validation RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (cycles)")
    plt.title("Training and Validation RMSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=160)
    plt.close()

    names = list(metrics.keys())
    values = [metrics[name]["RMSE"] for name in names]
    plt.figure(figsize=(8, 5))
    plt.bar(names, values)
    plt.ylabel("RMSE (cycles)")
    plt.title("Test RMSE Comparison (lower is better)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, predictions, alpha=0.75)
    limit = max(float(np.max(y_true)), float(np.max(predictions)), 1)
    plt.plot([0, limit], [0, limit], linestyle="--")
    plt.xlabel("True RUL (cycles)")
    plt.ylabel("Predicted RUL (cycles)")
    plt.title("Predicted vs True RUL")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predicted_vs_true.png", dpi=160)
    plt.close()

    errors = predictions - y_true
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=20)
    plt.xlabel("Prediction error: predicted - true (cycles)")
    plt.ylabel("Number of engines")
    plt.title("LSTM Error Distribution")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "error_distribution.png", dpi=160)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_raw, test_raw, test_rul = load_data()
    test_raw.attrs["final_rul"] = test_rul

    train = add_train_rul(train_raw)
    test = add_test_rul(test_raw, test_rul)

    train_rows, val_rows = split_by_engine(train)
    train_rows, val_rows, test_rows, _ = scale_sensor_rows(
        train_rows, val_rows, test
    )

    x_train, y_train, rf_x_train = make_windows(train_rows)
    x_val, y_val, _ = make_windows(val_rows)
    x_test, y_test, rf_x_test = make_last_test_windows(test_rows)

    if len(x_train) == 0 or len(x_val) == 0:
        raise RuntimeError("No training/validation windows were created. Check the dataset.")

    mean_prediction = np.full_like(y_test, fill_value=float(np.mean(y_train)))
    rf = RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=RANDOM_SEED, n_jobs=-1
    )
    # Match the reference baseline: current-cycle sensor values, not full sequences.
    rf.fit(rf_x_train, y_train)
    rf_predictions = rf.predict(rf_x_test)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Windows: train={len(x_train)}, validation={len(x_val)}, test={len(x_test)}")

    model, history = train_lstm(x_train, y_train, x_val, y_val, device)
    lstm_predictions = predict_lstm(model, x_test, device)

    predictions_by_model = {
        "Mean RUL baseline": mean_prediction,
        "Random Forest": rf_predictions,
        "LSTM": lstm_predictions,
    }

    metrics = {}
    for name, pred in predictions_by_model.items():
        metrics[name] = {
            "RMSE": rmse(y_test, pred),
            "MAE": float(mean_absolute_error(y_test, pred)),
            "NASA_score": nasa_score(y_test, pred),
        }
        print(
            f"{name}: RMSE={metrics[name]['RMSE']:.3f}, "
            f"MAE={metrics[name]['MAE']:.3f}, "
            f"NASA score={metrics[name]['NASA_score']:.3f}"
        )

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "candidate": "Maaninder Singh",
                "dataset": "NASA C-MAPSS FD001",
                "window_length": WINDOW_LENGTH,
                "rul_cap": RUL_CAP,
                "features": FEATURE_COLUMNS,
                "device": str(device),
                "metrics": metrics,
            },
            f,
            indent=2,
        )

    torch.save(model.state_dict(), OUTPUT_DIR / "lstm_rul_model.pt")
    save_plots(history, y_test, lstm_predictions, metrics)
    print(f"\nFinished. Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
