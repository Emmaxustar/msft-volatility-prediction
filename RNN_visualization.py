# RNN Visualization Functions
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

def make_rnn_test_sequences(
    X_train_scaled,
    X_valid_scaled,
    X_test_scaled,
    y_train,
    y_valid,
    y_test,
    window_size
):

    X_all = np.vstack([
        np.asarray(X_train_scaled),
        np.asarray(X_valid_scaled),
        np.asarray(X_test_scaled)
    ])

    y_all = pd.concat([y_train, y_valid, y_test])

    test_start_idx = len(y_train) + len(y_valid)
    test_end_idx = len(y_all)

    X_test_seq = []
    y_test_aligned = []
    test_index = []

    for i in range(test_start_idx, test_end_idx):

        start_idx = i - window_size

        if start_idx < 0:
            continue

        X_test_seq.append(X_all[start_idx:i])
        y_test_aligned.append(y_all.iloc[i])
        test_index.append(y_all.index[i])

    X_test_seq = np.array(X_test_seq)
    y_test_aligned = pd.Series(y_test_aligned, index=test_index)

    return X_test_seq, y_test_aligned


def plot_rnn_residuals_one_row(
    models,
    best_validation_table,
    targets_info,
    X_train_scaled,
    X_valid_scaled,
    X_test_scaled,
    dataset_name="Test"
):

    n_targets = len(targets_info)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_targets,
        figsize=(5.5 * n_targets, 4),
        dpi=120
    )

    if n_targets == 1:
        axes = [axes]

    residuals_dict = {}

    for ax, (target_name, data_dict) in zip(axes, targets_info.items()):

        y_train = data_dict["y_train"]
        y_valid = data_dict["y_valid"]
        y_test = data_dict["y_test"]

        window_size = get_rnn_window_size(
            best_validation_table,
            target_name
        )

        X_test_seq, y_test_seq = build_rnn_test_sequences(
            y_train=y_train,
            y_valid=y_valid,
            y_test=y_test,
            X_train_scaled=X_train_scaled,
            X_valid_scaled=X_valid_scaled,
            X_test_scaled=X_test_scaled,
            window_size=window_size
        )

        model = models[target_name]
        y_pred = predict_rnn(model, X_test_seq)

        aligned_index = y_test.index[-len(y_test_seq):]
        y_test_aligned = pd.Series(y_test_seq, index=aligned_index)

        residuals = y_test_aligned - y_pred
        residuals_dict[target_name] = residuals

        ax.scatter(y_pred, residuals, alpha=0.6, s=28)
        ax.axhline(y=0, linestyle="--")

        ax.set_title(f"{target_name}", fontsize=12)
        ax.set_xlabel("Predicted Volatility", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        f"RNN {dataset_name} Residual Plots",
        fontsize=15,
        y=1.05
    )

    plt.tight_layout()
    plt.show()

    return residuals_dict


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


def get_model_prediction_for_residual(
    model_name,
    target_name,
    model_dicts,
    X_test_final,
    best_ensemble_weights=None
):
    
    if model_name != "Ensemble":
        model = model_dicts[model_name][target_name]
        return model.predict(X_test_final)

    gb_model = model_dicts["Gradient Boosting"][target_name]
    lasso_model = model_dicts["Lasso"][target_name]

    gb_weight = best_ensemble_weights[target_name]["GB Weight"]
    lasso_weight = best_ensemble_weights[target_name]["Lasso Weight"]

    gb_pred = gb_model.predict(X_test_final)
    lasso_pred = lasso_model.predict(X_test_final)

    ensemble_pred = gb_weight * gb_pred + lasso_weight * lasso_pred

    return ensemble_pred


def plot_residuals_one_model_row(
    model_name,
    model_dicts,
    targets,
    X_test_final,
    best_ensemble_weights=None,
    dataset_name="Test"
):
    
    n_targets = len(targets)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_targets,
        figsize=(5.5 * n_targets, 4),
        dpi=120
    )

    if n_targets == 1:
        axes = [axes]

    residuals_dict = {}

    for ax, (target_name, y_true) in zip(axes, targets.items()):

        y_pred = get_model_prediction_for_residual(
            model_name=model_name,
            target_name=target_name,
            model_dicts=model_dicts,
            X_test_final=X_test_final,
            best_ensemble_weights=best_ensemble_weights
        )

        residuals = y_true - y_pred
        residuals_dict[target_name] = residuals

        ax.scatter(y_pred, residuals, alpha=0.6, s=28)
        ax.axhline(y=0, linestyle="--")

        ax.set_title(f"{target_name}", fontsize=12)
        ax.set_xlabel("Predicted Volatility", fontsize=10)
        ax.set_ylabel("Residuals", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        f"{model_name} {dataset_name} Residual Plots",
        fontsize=15,
        y=1.05
    )

    plt.tight_layout()
    plt.show()

    return residuals_dict


def get_basic_feature_importance(model, feature_cols):

    if hasattr(model, "feature_importances_"):
        importance_values = model.feature_importances_

    elif hasattr(model, "coef_"):
        importance_values = np.abs(model.coef_)

        if importance_values.ndim > 1:
            importance_values = importance_values.ravel()

    else:
        raise ValueError(
            "This model does not have feature_importances_ or coef_."
        )

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importance_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    return importance_df


def get_ensemble_feature_importance(
    target_name,
    model_dicts,
    feature_cols,
    best_ensemble_weights
):

    gb_model = model_dicts["Gradient Boosting"][target_name]
    lasso_model = model_dicts["Lasso"][target_name]

    gb_weight = best_ensemble_weights[target_name]["GB Weight"]
    lasso_weight = best_ensemble_weights[target_name]["Lasso Weight"]

    gb_importance = get_basic_feature_importance(
        gb_model,
        feature_cols
    ).set_index("Feature")

    lasso_importance = get_basic_feature_importance(
        lasso_model,
        feature_cols
    ).set_index("Feature")

    gb_values = gb_importance.reindex(feature_cols)["Importance"].values
    lasso_values = lasso_importance.reindex(feature_cols)["Importance"].values

    # Normalize before combining
    if gb_values.sum() != 0:
        gb_values = gb_values / gb_values.sum()

    if lasso_values.sum() != 0:
        lasso_values = lasso_values / lasso_values.sum()

    ensemble_values = gb_weight * gb_values + lasso_weight * lasso_values

    importance_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": ensemble_values
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    return importance_df


def plot_basic_importance_one_model_row(
    model_name,
    model_dicts,
    targets,
    feature_cols,
    best_ensemble_weights=None,
    top_n=5,
    dataset_name="Test"
):

    n_targets = len(targets)

    fig, axes = plt.subplots(
        nrows=1,
        ncols=n_targets,
        figsize=(5.8 * n_targets, 4),
        dpi=120
    )

    if n_targets == 1:
        axes = [axes]

    importance_dict = {}

    for ax, target_name in zip(axes, targets.keys()):

        if model_name == "Ensemble":
            importance_df = get_ensemble_feature_importance(
                target_name=target_name,
                model_dicts=model_dicts,
                feature_cols=feature_cols,
                best_ensemble_weights=best_ensemble_weights
            )
        else:
            model = model_dicts[model_name][target_name]
            importance_df = get_basic_feature_importance(
                model,
                feature_cols
            )

        importance_dict[target_name] = importance_df

        top_importance_df = importance_df.head(top_n)

        ax.barh(
            top_importance_df["Feature"][::-1],
            top_importance_df["Importance"][::-1]
        )

        ax.set_title(f"{target_name}", fontsize=12)
        ax.set_xlabel("Feature Importance", fontsize=10)
        ax.set_ylabel("Feature", fontsize=10)
        ax.tick_params(axis="both", labelsize=9)

    fig.suptitle(
        f"{model_name} {dataset_name} Feature Importance",
        fontsize=15,
        y=1.05
    )

    plt.tight_layout()
    plt.show()

    return importance_dict