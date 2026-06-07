# RNN Visualization Functions
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error


def make_sequences(X, y, window_size):
    Xs = []
    ys = []

    for i in range(len(X) - window_size + 1):
        Xs.append(X[i:i + window_size])
        ys.append(y[i + window_size - 1])

    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def get_rnn_window_size(best_rnn_validation_table, target_name):
    return int(
        best_rnn_validation_table[
            best_rnn_validation_table["Target"] == target_name
        ]["window_size"].iloc[0]
    )


def build_rnn_test_sequences(
    y_train, y_valid, y_test,
    X_train_scaled, X_valid_scaled, X_test_scaled,
    window_size
):
    X_all = np.concatenate([
        np.asarray(X_train_scaled),
        np.asarray(X_valid_scaled),
        np.asarray(X_test_scaled)
    ])

    y_all = np.concatenate([
        y_train.values.astype(np.float32),
        y_valid.values.astype(np.float32),
        y_test.values.astype(np.float32)
    ])

    X_seq, y_seq = make_sequences(X_all, y_all, window_size)

    n_train = len(y_train)
    n_valid = len(y_valid)

    test_start = n_train + n_valid - window_size + 1

    return X_seq[test_start:], y_seq[test_start:]


def predict_rnn(model, X_seq):
    model_device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        y_pred = model(
            torch.tensor(
                X_seq,
                dtype=torch.float32,
                device=model_device
            )
        ).cpu().numpy().ravel()

    return y_pred


def plot_rnn_actual_vs_predicted_single_model(
    model,
    y_train,
    y_valid,
    y_test,
    X_train_scaled,
    X_valid_scaled,
    X_test_scaled,
    window_size,
    model_name,
    target_name,
    dataset_name="Test"
):
    X_test_seq, y_test_seq = build_rnn_test_sequences(
        y_train, y_valid, y_test,
        X_train_scaled, X_valid_scaled, X_test_scaled,
        window_size
    )

    y_pred = predict_rnn(model, X_test_seq)

    aligned_index = y_test.index[-len(y_test_seq):]

    plt.figure(figsize=(12, 4), dpi=120)
    plt.plot(aligned_index, y_test_seq, label="Actual", linewidth=2)
    plt.plot(aligned_index, y_pred, label="Predicted", alpha=0.8)

    plt.title(f"{model_name} {dataset_name} Actual vs Predicted: {target_name}")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return y_pred