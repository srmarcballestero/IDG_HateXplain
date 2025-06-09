"""Utils for the model selection phase.

* @File    :   model_utils.py
* @Time    :   2025/03/29 09:01:05
* @Author  :   Marc Ballestero Ribó
* @Version :   0
* @Contact :   marcballesteroribo@gmail.com
* @License :   MIT
* @Desc    :   None
"""
from __future__ import annotations
from typing import Callable

import numpy as np
import torch

import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    RocCurveDisplay,
)

from tqdm import tqdm

from torch.utils.data import (
    DataLoader,
)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


def encode(
    text_items: list[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> dict:
    """Encode the input text using the provided tokenizer.

    Args:
        text_items: list[str] -- List of text items to encode.
        tokenizer: AutoTokenizer -- Tokenizer to use for encoding.
        max_length: int -- Maximum length of the encoded sequences (default: 128).

    Returns:
        dict -- Encoded input with input_ids, attention_mask, and token_type_ids.

    """
    return tokenizer(text_items,
                is_split_into_words=True,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt")


def train_model(  # noqa: PLR0913, PLR0915
    model_clf: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Callable,
    epochs: int,
    device: torch.DeviceObjType,
    out_dir: Path,
    model_name: str,
    monitor: str = "loss",
) -> tuple[list]:
    """Train the model with the given parameters.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to train.
        tokenizer: AutoTokenizer -- Tokenizer to use.
        train_dataloader: DataLoader -- Dataloader for the training set.
        val_dataloader: DataLoader -- Dataloader for the validation set.
        optimizer: torch.optim.Optimizer -- Optimizer to use.
        scheduler: Callable -- Scheduler to use.
        epochs: int -- Number of epochs to train.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).
        out_dir: Path -- Directory to save the model.
        model_name: str -- Name of the model.
        monitor: str -- Metric to monitor for saving the model ("loss" or "f1").

    Returns:
        tuple[list] -- Train and validation losses and F1-scores.

    """
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    train_f1_scores = []
    val_f1_scores = []
    best_val_f1 = -float("inf")

    for epoch in range(epochs):
        # Training phase
        model_clf.train()
        total_train_loss = 0
        epoch_train_losses = []
        train_labels, train_preds = [], []

        progress_bar = tqdm(total=len(train_dataloader) + len(val_dataloader),
                            desc=f"Epoch {epoch+1}/{epochs}",
                            unit="batch")

        for batch in train_dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

            model_clf.zero_grad()
            outputs = model_clf(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs.loss
            logits = outputs.logits

            epoch_train_losses.append(loss.item())
            total_train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_clf.parameters(), max_norm=2.0)
            optimizer.step()
            scheduler.step()

            # Store predictions & labels for F1 computation
            train_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            train_labels.extend(b_labels.cpu().numpy())

            progress_bar.set_postfix({"train_loss": loss.item()})
            progress_bar.update(1)

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(epoch_train_losses)

        # Compute train F1-score
        train_f1 = f1_score(train_labels, train_preds, average="binary")
        train_f1_scores.append(train_f1)

        # Validation phase
        model_clf.eval()
        total_val_loss = 0
        epoch_val_losses = []
        val_labels, val_preds = [], []

        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)

                outputs = model_clf(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                loss = outputs.loss
                logits = outputs.logits

                epoch_val_losses.append(loss.item())
                total_val_loss += loss.item()

                # Store predictions & labels for F1 computation
                val_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_labels.extend(b_labels.cpu().numpy())

                progress_bar.set_postfix({"train_loss": avg_train_loss, "val_loss": loss.item()})
                progress_bar.update(1)

        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(epoch_val_losses)

        # Compute validation F1-score
        val_f1 = f1_score(val_labels, val_preds, average="binary")
        val_f1_scores.append(val_f1)

        progress_bar.set_postfix({"train_loss": avg_train_loss, "val_loss": avg_val_loss, "val_f1": val_f1})
        progress_bar.close()

        # Save the model if the validation loss improves
        match monitor:
            case "loss":
                if avg_val_loss < best_val_loss:
                    print(f"Validation loss improved from {best_val_loss: .3f} to {avg_val_loss: .3f}. Saving model...")
                    model_clf.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    torch.save(model_clf.state_dict(), out_dir / f"{model_name}.pt")
                    print(f"Model saved to {out_dir}")
                    best_val_loss = avg_val_loss
                else:
                    print(f"Validation loss did not improve from {best_val_loss: .3f}. Not saving model.")

            case "f1":
                if val_f1 > best_val_f1:
                    print(f"Validation f1-score improved from {best_val_f1: .3f} to {val_f1: .3f}. Saving model...")
                    model_clf.save_pretrained(out_dir)
                    tokenizer.save_pretrained(out_dir)
                    torch.save(model_clf.state_dict(), out_dir / f"{model_name}.pt")
                    print(f"Model saved to {out_dir}")
                    best_val_f1 = val_f1
                else:
                    print(f"Validation f1-score did not improve from {best_val_f1: .3f}. Not saving model.")

            case _:
                msg = "Non-valid monitor metric!"
                raise ValueError(msg)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Train F1: {train_f1:.4f} | Val Loss: {avg_val_loss:.6f} | Val F1: {val_f1:.4f}")
        print()

    return train_losses, val_losses, train_f1_scores, val_f1_scores


def evaluate_model(
    model_clf: AutoModelForSequenceClassification,
    dataloader: DataLoader,
    device: torch.DeviceObjType,
) -> tuple[list]:
    """Evaluate the model with the given parameters.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to evaluate.
        dataloader: DataLoader -- Dataloader for the validation set.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).

    Returns:
        tuple[list] -- Correct predictions, total predictions, all predictions, all labels, and all probabilities.

    """
    # Model evaluation
    model_clf.eval()
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []
    all_probabilites = []

    with tqdm(total=len(dataloader), desc="Running validation", unit="batch") as progress_bar:
        for batch in dataloader:
            b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)


            with torch.no_grad():
                outputs = model_clf(b_input_ids, attention_mask=b_input_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                correct += (predictions == b_labels.squeeze()).sum().item()
                total += b_labels.size(0)
                probabilities = torch.nn.functional.softmax(logits, dim=1)[:, 1]

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(b_labels.cpu().numpy())
                all_probabilites.extend(probabilities.cpu().numpy())
            progress_bar.update(1)
        progress_bar.close()

    return correct, total, all_predictions, all_labels, all_probabilites


def plot_evaluation_report(  # noqa: PLR0913
    correct: int,
    total: int,
    epochs: int,
    target_labels: list,
    all_predictions: list,
    all_labels: list,
    all_probabilites: list,
    plot_learning_curves: bool = True,
    *,
    train_losses: list,
    val_losses: list,
    train_f1_scores: list,
    val_f1_scores: list,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot the evaluation report with confusion matrix, ROC curve, and learning curves.

    Args:
        correct: int -- Number of correct predictions.
        total: int -- Total number of predictions.
        epochs: int -- Number of epochs.
        target_labels: list -- List of target labels.
        all_predictions: list -- List of all predictions.
        all_labels: list -- List of all true labels.
        all_probabilites: list -- List of all predicted probabilities.
        plot_learning_curves: bool -- Whether to plot learning curves (default: True).
        train_losses: list -- List of training losses per epoch.
        val_losses: list -- List of validation losses per epoch.
        train_f1_scores: list -- List of training F1-scores per epoch.
        val_f1_scores: list -- List of validation F1-scores per epoch.

    Returns:
        tuple[plt.Figure, list[plt.Axes]] -- Figure and axes of the plots.

    """
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2f}")
    print(classification_report(all_labels, all_predictions, target_names=target_labels, digits=3))

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(14, 14))
    axs = axs.flatten()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=axs[0])
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xticklabels(target_labels)
    axs[0].set_yticklabels(target_labels)
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")
    axs[0].set_aspect("equal")

    # ROC-AUC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilites)
    aucurve = auc(fpr, tpr)
    print(f"ROC AUC: {aucurve:.4f}")
    display_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucurve).plot(ax=axs[1])
    display_roc.ax_.plot([0, 1], [0, 1], "k--")
    display_roc.ax_.set_title("ROC Curve")

    # Learning curves
    if plot_learning_curves:
        train_losses = np.array([np.mean(epoch) for epoch in train_losses])
        val_losses = np.array([np.mean(epoch) for epoch in val_losses])

        train_std = np.array([np.std(epoch) for epoch in train_losses])
        val_std = np.array([np.std(epoch) for epoch in val_losses])

        epochs_range = np.arange(1, epochs + 1)

        axs[2].plot(epochs_range, train_losses, label="Train Loss", color="blue")
        axs[2].fill_between(epochs_range, train_losses - train_std, train_losses + train_std, color="blue", alpha=0.2)
        axs[2].plot(epochs_range, val_losses, label="Val Loss", color="red")
        axs[2].fill_between(epochs_range, val_losses - val_std, val_losses + val_std, color="red", alpha=0.2)

        axs[2].set_xticks(epochs_range, epochs_range)
        axs[2].set_xlabel("Epoch")
        axs[2].set_ylabel("Loss")
        axs[2].set_title("Learning Curve")
        axs[2].legend()

        # F1-scores
        axs[3].plot(epochs_range, train_f1_scores, label="Train F1", color="blue")
        axs[3].plot(epochs_range, val_f1_scores, label="Val F1", color="red")

        axs[3].set_xticks(epochs_range, epochs_range)
        axs[3].set_xlabel("Epoch")
        axs[3].set_ylabel("F1 Score")
        axs[3].set_title("F1 Score")
        axs[3].legend()

    return fig, axs


def plot_inference_report(  # noqa: PLR0913
    correct: int,
    total: int,
    target_labels: list,
    all_predictions: list,
    all_labels: list,
    all_probabilites: list,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot the inference report with confusion matrix and ROC curve.

    Args:
        correct: int -- Number of correct predictions.
        total: int -- Total number of predictions.
        target_labels: list -- List of target labels.
        all_predictions: list -- List of all predictions.
        all_labels: list -- List of all true labels.
        all_probabilites: list -- List of all predicted probabilities.

    Returns:
        tuple[plt.Figure, list[plt.Axes]] -- Figure and axes of the plots.

    """
    # Accuracy and classification report
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.2f}")
    print(classification_report(all_labels, all_predictions, target_names=target_labels, digits=4))

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(14, 7))
    axs = axs.flatten()

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", ax=axs[0])
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xticklabels(target_labels)
    axs[0].set_yticklabels(target_labels)
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")
    axs[0].set_aspect("equal")

    # ROC-AUC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probabilites)
    aucurve = auc(fpr, tpr)
    display_roc = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=aucurve).plot(ax=axs[1])
    display_roc.ax_.plot([0, 1], [0, 1], "k--")
    display_roc.ax_.set_title("ROC Curve")

    return fig, axs
