"""Utils for the explainability phase.

* @File    :   expl_utils.py
* @Time    :   2025/03/29 09:01:05
* @Author  :   Marc Ballestero Ribó
* @Version :   0
* @Contact :   marcballesteroribo@gmail.com
* @License :   MIT
* @Desc    :   None
"""
from __future__ import annotations

import ast
import json

import polars as pl
import regex as re
import numpy as np
import torch
import networkx as nx
import uuid

import matplotlib.pyplot as plt


import sklearn

from pathlib import Path


from .model_utils import encode
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

try:
    from sklearnex import patch_sklearn
    patch_sklearn()
except ImportError:
    pass


def get_idg_explanations(
    idg_dir: Path,
    df_data: pl.DataFrame,
    guid: str,
    *,
    normalize: bool = True,
) -> dict:
    """Get the IDG explanation for a specific post.

    Args:
        idg_dir: Path -- Directory where IDG files are stored.
        df_data: pl.DataFrame -- DataFrame containing the data.
        guid: str -- Unique identifier for the post.
        normalize: bool -- Whether to normalize the explanation values.

    Returns:
        tuple -- A tuple containing the IDG values and tokens.

    """
    with Path.open(idg_dir / f"{guid}_idg_expl_data.json", "r") as f:
        idg_expl = json.load(f)

    coalitions = idg_expl["coalitions"]
    val_score = idg_expl["value_func"]
    div_dir = idg_expl["dividend_dir"]

    idg_values = {}
    for i, c in enumerate(coalitions):
        if len(c) == 1:
            idg_values[c[0]] = div_dir[i]*val_score[i]

    if normalize:
        norm_factor = np.sum(np.abs(np.array(list(idg_values.values()))))
        idg_values = {
            k: v / norm_factor for k, v in idg_values.items()
            if norm_factor != 0
        }

    # Get the tokens for the IDG values
    post = df_data.filter(pl.col("post_id") == guid)
    tokens = post["tokens"].to_list()[0]
    tokens = positional_encode_tokens(tokens)

    # Check if the tokens in the IDG values match the tokens in the post
    if set(tokens) != set(idg_values.keys()):
        raise ValueError(  # noqa: TRY003
            "The tokens in the IDG values do not match the tokens in the post.\n"  # noqa: EM102
            f"GUID = {guid}\n"
            f"Tokens in post:       {tokens}\n"
            f"Tokens in IDG values: {list(idg_values.keys())}",
        )

    # Order the IDG values according to the tokens and return them
    return dict(zip(tokens, [idg_values[tok] for tok in tokens]))



def positional_encode_tokens(
    tokens: list[str],
) -> list[str]:
    """Encode tokens with positional information.

    Args:
        tokens: list[str] -- List of tokens.

    Returns:
        list[str] -- List of encoded tokens with positional information.

    """
    _token_dict = {}
    _encoded = []

    for token in tokens:
        if token not in _token_dict:
            _encoded.append(token)
            _token_dict[token] = 1
        else:
            _encoded.append(f"{token}{_token_dict[token]}")
            _token_dict[token] += 1

    return _encoded


def get_rationales(
    df_data: pl.DataFrame,
    guid: str,
    *,
    normalize: bool = True,
) -> dict:
    """Get the rationales for a specific post.

    Args:
        df_data: pl.DataFrame -- DataFrame containing the data.
        guid: str -- Unique identifier for the post.
        normalize: bool -- Whether to normalize the explanation values.

    Returns:
        dict -- A dictionary containing the rationales.

    """
    post = df_data.filter(pl.col("post_id") == guid)
    tokens = post["tokens"].to_list()[0]
    rationale = post["rationale"].to_list()[0]

    tokens = positional_encode_tokens(tokens)

    rationale_values = dict(zip(tokens, rationale))

    if normalize:
        norm_factor = np.sum(np.abs(np.array(list(rationale_values.values()))))
        rationale_values = {
            k: v / norm_factor for k, v in rationale_values.items()
            if norm_factor != 0
        }

    return rationale_values

def check_attribution_tokens(
    attr1_values: dict,
    attr2_values: dict,
) -> None:
    """Check if the tokens in the two attribution methods are the same.

    Args:
        attr1_values: dict -- Dictionary containing attribution 1 method token values.
        attr2_values: dict -- Dictionary containing attribution 2 method token values.

    Returns:
        bool -- True if the tokens are the same, False otherwise.

    """
    tokens_attr1 = list(attr1_values.keys())
    tokens_attr2 = list(attr2_values.keys())

    try:
        assert tokens_attr1 == tokens_attr2  # noqa: S101
    except AssertionError:
        return False
    else:
        return True

def relevance_at_k(
    attr1_values: dict,
    attr2_values: dict,
    k: int,
    token: str,
) -> float:
    """Calculate the relevance at k for a specific token.

    Args:
        attr1_values: dict -- Dictionary containing attribution 1 method token values.
        attr2_values: dict -- Dictionary containing attribution 2 method token values.
        k: int -- The number of top tokens to consider.
        token: str -- The token to evaluate.

    Returns:
        float -- The relevance at k for the specified token.

    """
    # Check if the tokens in the two attribution methods are the same
    if not check_attribution_tokens(attr1_values, attr2_values):
        raise ValueError(  # noqa: TRY003
            "The tokens in the two attribution methods are not the same.\n"  # noqa: EM102
            f"Tokens in attribution 1: {list(attr1_values.keys())}\n"
            f"Tokens in attribution 2: {list(attr2_values.keys())}",
        )


    # Sort attributions
    sorted_attr1 = sorted(
        attr1_values.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    sorted_attr2 = sorted(
        attr2_values.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    # Retrieve top k tokens and rationales
    top_k_attr1 = dict(sorted_attr1[:k])
    top_k_attr2 = dict(sorted_attr2[:k])

    # Draw case: Include additional tokens with the same value as the k-th token
    if k < len(sorted_attr1):
        kth_value = sorted_attr1[k - 1][1]
        top_k_attr1.update(
            {
                tok: val for tok, val in sorted_attr1[k:] if val == kth_value
            },
        )
    if k < len(sorted_attr2):
        kth_value = sorted_attr2[k - 1][1]
        top_k_attr2.update(
            {
                tok: val for tok, val in sorted_attr2[k:] if val == kth_value
            },
        )

    return .5 * (token in top_k_attr1 and token in top_k_attr2) + \
           .5 * (token in top_k_attr1 or token in top_k_attr2)


def agreement_at_k(
    attr1_values: dict,
    attr2_values: dict,
    k: int,
) -> float:
    """Calculate the agreement at k for a specific token.

    Args:
        attr1_values: dict -- Dictionary containing attribution 1 method token values.
        attr2_values: dict -- Dictionary containing attribution 2 method token values.
        k: int -- The number of top tokens to consider.

    Returns:
        float -- The agreement at k for the specified token.

    """
    # Compute relevances @k
    relevances = [
        relevance_at_k(attr1_values, attr2_values, k, token)
        for token in attr1_values
    ]

    return sum(relevances) / len([relevance for relevance in relevances if relevance > 0])

def check_label(
    df_data: pl.DataFrame,
    guid: str,
    label: str,
) -> bool:
    """Check if the label of a specific post matches the given label.

    Args:
        df_data: pl.DataFrame -- DataFrame containing the data.
        guid: str -- Unique identifier for the post.
        label: str -- The label to check.

    Returns:
        bool -- True if the label matches, False otherwise.

    """
    post = df_data.filter(pl.col("post_id") == guid)
    true_label = post["label"].to_list()[0]
    return true_label == label

def merge_tokens_to_words(
    token_values: dict,
) -> dict:
    """Merge token values into words and sum the values.

    Args:
        token_values: dict -- Dictionary containing token values.

    Returns:
        dict -- Dictionary containing word values.

    """
    merged_token_values = {}
    current_token = ""
    current_value = 0.

    tokens = list(token_values.items())

    for raw_token, value in tokens:

        if raw_token.startswith("##"):
            current_token = re.sub(r"\d+$", "", current_token)
            current_token += raw_token[2:]
        else:
            if current_token:
                merged_token_values[current_token] = current_value
            current_token = raw_token
            current_value = value

    if current_token:
        merged_token_values[current_token] = current_value

    return merged_token_values

def get_ig_explanations(
    ig_dir: Path,
    df_data: pl.DataFrame,
    guid: str,
    *,
    normalize: bool = True,
) -> dict:
    """Get the IG explanation for a specific post.

    Args:
        ig_dir: Path -- Directory where IG files are stored.
        df_data: pl.DataFrame -- DataFrame containing the data.
        guid: str -- Unique identifier for the post.
        normalize: bool -- Whether to normalize the explanation values.

    Returns:
        dict -- A dictionary containing the IG values and tokens.

    """
    with Path.open(ig_dir / f"{guid}_ig_expl_data.json", "r") as f:
        ig_expl = json.load(f)

    ig_values = np.array(ig_expl["attributions"])

    ig_values[:, 0] = positional_encode_tokens(ig_values[:, 0])

    # Remove [CLS] and [SEP] tokens and cast to float
    ig_values = {
        ig_values[:, 0][i]: float(ig_values[:, 1][i]) for i in range(len(ig_values))
        if ig_values[:, 0][i] not in ["[CLS]", "[SEP]"]
    }

    # Merge tokens to words
    ig_values = merge_tokens_to_words(ig_values)

    # Reencode tokens
    tokens = list(ig_values.keys())
    ig_values = np.array(list(ig_values.values()))
    tokens = [re.sub(r"\d+$", "", token) for token in tokens]
    tokens = positional_encode_tokens(tokens)
    ig_values = dict(zip(tokens, ig_values))


    if normalize:
        norm_factor = np.sum(np.abs(np.array(list(ig_values.values()))))
        ig_values = {
            k: v / norm_factor for k, v in ig_values.items()
            if norm_factor != 0
        }

    # Get the tokens for the IG values
    post = df_data.filter(pl.col("post_id") == guid)
    tokens = post["tokens"].to_list()[0]
    tokens = positional_encode_tokens(tokens)

    # Check if the tokens in the IG values match the tokens in the post
    if set(tokens) != set(ig_values.keys()):
        raise ValueError(  # noqa: TRY003
            "The tokens in the IG values do not match the tokens in the post.\n"  # noqa: EM102
            f"GUID = {guid}\n"
            f"Tokens in post:       {tokens}\n"
            f"Tokens in IG values:  {list(ig_values.keys())}",
        )

    # Order the IG values according to the tokens and return them
    return dict(zip(tokens, [ig_values[tok] for tok in tokens]))



def get_shap_explanations(
    shap_dir: Path,
    df_data: pl.DataFrame,
    guid: str,
    *,
    normalize: bool = True,
) -> dict:
    """Get the IG explanation for a specific post.

    Args:
        shap_dir: Path -- Directory where SHAP files are stored.
        df_data: pl.DataFrame -- DataFrame containing the data.
        guid: str -- Unique identifier for the post.
        normalize: bool -- Whether to normalize the explanation values.

    Returns:
        dict -- A dictionary containing the SHAP values and tokens.

    """
    with Path.open(shap_dir / f"{guid}_shap_expl_data.json", "r") as f:
        shap_expl = json.load(f)

    shap_values = np.array(shap_expl["attributions"])

    shap_values[:, 0] = positional_encode_tokens(shap_values[:, 0])

    # Remove [CLS] and [SEP] tokens and cast to float
    shap_values = {
        shap_values[:, 0][i]: float(shap_values[:, 1][i]) for i in range(len(shap_values))
        if shap_values[:, 0][i] not in ["[CLS]", "[SEP]"]
    }

    # Merge tokens to words
    shap_values = merge_tokens_to_words(shap_values)

    # Reencode tokens
    tokens = list(shap_values.keys())
    shap_values = np.array(list(shap_values.values()))
    tokens = [re.sub(r"\d+$", "", token) for token in tokens]
    tokens = positional_encode_tokens(tokens)
    shap_values = dict(zip(tokens, shap_values))


    if normalize:
        norm_factor = np.sum(np.abs(np.array(list(shap_values.values()))))
        shap_values = {
            k: v / norm_factor for k, v in shap_values.items()
            if norm_factor != 0
        }

    # Get the tokens for the IG values
    post = df_data.filter(pl.col("post_id") == guid)
    tokens = post["tokens"].to_list()[0]
    tokens = positional_encode_tokens(tokens)

    # Check if the tokens in the IG values match the tokens in the post
    if set(tokens) != set(shap_values.keys()):
        raise ValueError(  # noqa: TRY003
            "The tokens in the IG values do not match the tokens in the post.\n"  # noqa: EM102
            f"GUID = {guid}\n"
            f"Tokens in post:       {tokens}\n"
            f"Tokens in IG values:  {list(shap_values.keys())}",
        )

    # Order the IG values according to the tokens and return them
    return dict(zip(tokens, [shap_values[tok] for tok in tokens]))




def precision_recall_curve(
    attr_values: dict,
    rationale_values: dict,
    recall_points: int = 100,
) -> tuple:
    """Calculate precision and recall values for two sets of attribution values.

    Args:
        attr_values: dict -- Dictionary containing the attribution values.
        rationale_values: dict -- Dictionary containing the human rationale values.
        recall_points: int -- Number of points to calculate recall.

    Returns:
       tuple: Tuple containing precision and recall values.

    """
    check_attribution_tokens(attr_values, rationale_values)

    tokens = list(attr_values.keys())
    gold_labels = np.array([
        1. if rationale_values[token] > 0. else 0. for token in tokens
    ])
    model_scores = np.array([
        attr_values[token] for token in tokens
    ])

    # Compute the precision-recall curve
    precision, recall, _ = sklearn.metrics.precision_recall_curve(
        gold_labels,
        model_scores,
    )

    # Compute the area under the curve (AUPRC)
    auprc = sklearn.metrics.auc(recall, precision)

    # Interpolate the precision and recall values
    precision = np.interp(
        np.linspace(0, 1, recall_points),
        recall[::-1],
        precision[::-1],
    )

    return precision, np.linspace(0, 1, recall_points), auprc


def receiver_operating_characteristic(
    attr_values: dict,
    rationale_values: dict,
    recall_points: int = 100,
) -> tuple:
    """Calculate the receiver operating characteristic (ROC) curve values.

    Args:
        attr_values: dict -- Dictionary containing the attribution values.
        rationale_values: dict -- Dictionary containing the human rationale values.
        recall_points: int -- Number of points to calculate recall.

    Returns:
       tuple: Tuple containing false positive rate, true positive rate, and AUC.

    """
    check_attribution_tokens(attr_values, rationale_values)

    tokens = list(attr_values.keys())
    gold_labels = np.array([
        1. if rationale_values[token] > 0. else 0. for token in tokens
    ])
    model_scores = np.array([
        attr_values[token] for token in tokens
    ])

    # Compute the ROC curve
    fpr, tpr, _ = sklearn.metrics.roc_curve(
        gold_labels,
        model_scores,
    )

    # Compute the area under the curve (AUC)
    roc_auc = sklearn.metrics.auc(fpr, tpr)

    # Interpolate the ROC curve values
    tpr = np.interp(
        np.linspace(0, 1, recall_points),
        fpr,
        tpr,
    )

    return np.linspace(0, 1, recall_points), tpr, roc_auc

def filter_attributions_w_threshold(
    attributions: dict,
    threshold: float,
) -> dict:
    """Filter attributions based on a threshold.

    Args:
        attributions: dict -- Dictionary containing the attribution values.
        threshold: float -- Threshold value.

    Returns:
        dict: Filtered attributions.

    """
    return {
        token: value if value >= threshold else 0. for token, value in attributions.items()
    }

def filter_attributions_w_threshold_tuple(
    attributions: list[tuple[str, float]],
    threshold: float,
) -> list[tuple[str, float]]:
    """Filter attributions based on a threshold.

    Args:
        attributions: list[tuple[str, float]] -- List of tuples containing the attribution values.
        threshold: float -- Threshold value.

    Returns:
        list[tuple[str, float]]: Filtered attributions.

    """
    return [
        (token, value) if value >= threshold else (token, 0.) for token, value in attributions
    ]


def predict_sample(
    model_clf: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokens: list[str],
    device: torch.DeviceObjType,
    *,
    model_to_gpu_mode: bool = True,
) -> np.ndarray:
    """Predict the class of a sample using the model.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to use for prediction.
        tokenizer: AutoTokenizer -- Tokenizer to use for encoding the sample.
        tokens: list[str] -- Sample text to predict.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).
        model_to_gpu_mode: bool -- Whether to move the model to GPU.

    Returns:
        np.ndarray -- Predicted probabilities.

    """
    inputs = encode(tokens, tokenizer)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    if model_to_gpu_mode:
        model_clf.to(device)
        model_clf.eval()

    with torch.no_grad():
        outputs = model_clf(**inputs)
        logits = outputs.logits
        return torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]


def compute_comprehensiveness(  # noqa: PLR0913
    model_clf: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokens: list[str],
    attribution_values: dict,
    threshold: float,
    device: torch.DeviceObjType,
    *,
    original_probabilities: np.ndarray | None = None,
    predicted_class: int | None = None,
) -> float:
    """Compute the comprehensiveness of the attributions.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to use for prediction.
        tokenizer: AutoTokenizer -- Tokenizer to use for encoding the sample.
        tokens: list[str] -- Sample text to predict.
        attribution_values: dict -- Dictionary containing the attribution values.
        threshold: float -- Threshold value.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).
        *
        original_probabilities: np.ndarray -- Original probabilities of the model.
        predicted_class: int -- Predicted class of the model.

    Returns:
        float -- Comprehensiveness score.

    """
    # Filter attributions
    filtered_attributions = filter_attributions_w_threshold(
        attribution_values,
        threshold,
    )

    # Generate the counterexample
    masked_tokens = [
        token if not filtered_attributions[token] > 0 else "[PAD]"
        for token in tokens
    ]

    if original_probabilities is None or predicted_class is None:
        # Predict the probabilities of the original input
        original_probabilities = predict_sample(
            model_clf,
            tokenizer,
            tokens,
            device,
            model_to_gpu_mode=False,
        )
        predicted_class = original_probabilities.argmax()

    # Predict the probabilities of the masked input
    masked_probabilities = predict_sample(
        model_clf,
        tokenizer,
        masked_tokens,
        device,
        model_to_gpu_mode=False,
    )

    # Calculate the comprehensiveness score
    return original_probabilities[predicted_class] - masked_probabilities[predicted_class]


def compute_sufficiency(  # noqa: PLR0913
    model_clf: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokens: list[str],
    attribution_values: dict,
    threshold: float,
    device: torch.DeviceObjType,
    *,
    original_probabilities: np.ndarray | None = None,
    predicted_class: int | None = None,
) -> float:
    """Compute the sufficiency of the attributions.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to use for prediction.
        tokenizer: AutoTokenizer -- Tokenizer to use for encoding the sample.
        tokens: list[str] -- Sample text to predict.
        attribution_values: dict -- Dictionary containing the human rationale values.
        threshold: float -- Threshold value.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).
        *
        original_probabilities: np.ndarray -- Original probabilities of the model.
        predicted_class: int -- Predicted class of the model.

    Returns:
        float -- Sufficiency score.

    """
    # Filter attributions
    filtered_attributions = filter_attributions_w_threshold(
        attribution_values,
        threshold,
    )

    # Generate the counterexample
    masked_tokens = [
        token if filtered_attributions[token] > 0 else "[PAD]"
        for token in tokens
    ]

    if original_probabilities is None or predicted_class is None:
        # Predict the probabilities of the original input
        original_probabilities = predict_sample(
            model_clf,
            tokenizer,
            tokens,
            device,
            model_to_gpu_mode=False,
        )
        predicted_class = original_probabilities.argmax()

    # Predict the probabilities of the masked input
    masked_probabilities = predict_sample(
        model_clf,
        tokenizer,
        masked_tokens,
        device,
        model_to_gpu_mode=False,
    )

    # Calculate the comprehensiveness score
    return original_probabilities[predicted_class] - masked_probabilities[predicted_class]



class WrapperNode:
    """A wrapper class for a node in a tree structure."""

    uid: str | None
    data: dict | None
    children: list[WrapperNode] | None

    def __init__(
        self,
        uid: str | None = None,
        data: dict | None = None,
        children: list[WrapperNode] | None = None,
    ) -> None:
        """Initialize the WrapperNode.

        Args:
            uid: str -- Unique identifier for the node.
            data: dict -- Data associated with the node.
            children: list[WrapperNode] -- List of child nodes.

        """
        self.uid = uid
        self.data = data
        self.children = children if children is not None else []

    def __repr__(
        self,
    ) -> str:
        """Return a string representation of the WrapperNode.

        Returns:
            str -- String representation of the WrapperNode.

        """
        return f"WrapperNode(uid={self.uid}, data={self.data}, children={self.children})"

def nx_to_wrapped_tree(
    G: nx.Graph,  # noqa: N803
    root: str,
    visited: set[str] | None = None,
) -> WrapperNode:
    """Construct a wrapped tree from a graph.

    Args:
        G: nx.Graph -- Graph to construct the tree from.
        root: str -- Root node of the tree.
        visited: set[str] -- Set of visited nodes.

    Returns:
        WrapperNode -- Wrapped tree structure.

    """
    if visited is None:
        visited = set()

    visited.add(root)
    children = []

    for neighbor in G.neighbors(root):
        if neighbor not in visited:
            child_node = nx_to_wrapped_tree(G, neighbor, visited)
            children.append(child_node)

    return WrapperNode(uid=root, data=G.nodes[root], children=children)


def equalize_depths(
    node: WrapperNode,
) -> tuple[WrapperNode, int]:
    """Equalize the depths of the tree.

    Args:
        node (WrapperNode): The root node of the tree.

    Returns:
        tuple[WrapperNode, int]: The balanced tree and its height.

    """
    if not node.children:
        return node, 1

    new_children = []
    heights = []

    for child in node.children:
        new_child, h = equalize_depths(child)
        new_children.append(new_child)
        heights.append(h)

    max_height = max(heights)

    # Pad each child to the maximum height
    padded_children = []
    for child, h in zip(new_children, heights):
        padded_child = child
        for _ in range(max_height - h):
            pad_id = f"PAD_{uuid.uuid4().hex[:6]}"
            # Inherit data from the deepest original node
            padded_child = WrapperNode(uid=pad_id, data=child.data, children=[padded_child])
        padded_children.append(padded_child)

    return WrapperNode(node.uid, node.data, padded_children), max_height + 1


def wrapped_tree_to_nx(
    tree: WrapperNode,
) -> nx.DiGraph:
    """Convert a wrapped tree to a NetworkX directed graph.

    Args:
        tree (WrapperNode): The root node of the wrapped tree.

    Returns:
        nx.DiGraph: The directed graph representation of the tree.

    """
    G = nx.DiGraph()  # noqa: N806

    def add(
        node: WrapperNode,
        parent: int | None = None,
    ) -> None:
        """Add a node to the graph.

        Args:
            node (WrapperNode): The node to add.
            parent (int | None): The parent node.

        """
        node.data = {"value": node.data["label"], "color": node.data["color"]}
        G.add_node(node.uid, **node.data)
        if parent:
            G.add_edge(parent, node.uid)
        for child in node.children:
            add(child, node.uid)

    add(tree)

    return G


def get_tree_height(
    tree: WrapperNode,
) -> int:
    """Calculate the height of a tree.

    Args:
        tree (WrapperNode): The root node of the tree.

    Returns:
        int: the height of the tree.

    """
    if not tree.children:
        return 1

    return 1 + max(get_tree_height(child) for child in tree.children)


def get_nodes_at_depth(
    tree: WrapperNode,
    depth: int,
) -> list[WrapperNode]:
    """Get all the nodes at a specific depth from the leaves in the tree.

    Arguments:
        tree: WrapperNode -- The root of the tree
        depth: int -- The depth level from the leaves to extract nodes from.

    Returns:
        list[WrapperNode]: List of nodes at the specified depth.

    """
    tree_height = get_tree_height(tree)
    target_depth = tree_height - depth

    if target_depth < 0:
        return []

    def get_nodes_at_depth_(
        node: WrapperNode,
        current_depth: int,
    ) -> list[WrapperNode]:
        """Get all the nodes at a specific depth from the leaves in the tree.

        Arguments:
            node: WrapperNode -- The root of the tree
            current_depth: int -- The current depth level.

        Returns:
            list[WrapperNode]: List of nodes at the specified depth.

        """
        if current_depth == 0:
            return [node]

        nodes = []
        for child in node.children:
            nodes.extend(get_nodes_at_depth_(child, current_depth + 1))

        return nodes

    return get_nodes_at_depth_(tree, 0)


def populate_nodes_with_data(  # noqa: C901, PLR0915
    tree: nx.DiGraph,
    tokens: list[str],
) -> None:
    """Populate the nodes of the padded tree with data.

    Args:
        tree: nx.DiGraph -- The padded tree to populate.
        tokens: list[str] -- List of tokens to use for populating the nodes.

    """
    # Clean the nodes data
    for _, data in tree.nodes(data=True):
        data["value"] = data["value"].replace("\\<", "<").replace("\\>", ">")

    # Find the leaves of the node
    leaves = [node for node in tree.nodes() if tree.out_degree(node) == 0]

    # Add the dividend direction based on the color of the node
    for _, data in tree.nodes(data=True):
        if data["color"].startswith("0.835 1.000"):
            data["div_dir"] = -1
        elif data["color"].startswith("0.334 1.000"):
            data["div_dir"] = 1
        else:
            data["div_dir"] = 0
        data.pop("color", None)

    # Node to label map
    node2label = dict(tree.nodes(data=True))

    tokens = positional_encode_tokens(tokens)

    # Order the leaves based on token order
    ordered_leaves = sorted(
        leaves,
        key=lambda node: tokens.index(node2label[node]["value"])
            if node2label[node]["value"] in tokens else float("inf"),
    )

    # Initialize positions for known leaves
    positions = {}
    for idx, leaf in enumerate(ordered_leaves):
        positions[leaf] = (idx, 0)

    # Fallback counter for unexpected leaves
    next_leaf_x = len(ordered_leaves)

    # Find depths (distance from the root)
    root = [node for node in tree.nodes() if tree.in_degree(node) == 0][0]  # noqa: RUF015
    depths = {}

    def dfs_depth_(
        node: str,
        current_depth: int = 0,
    ) -> None:
        """Depth-first search to find the depth of each node.

        Args:
            node: str -- The current node.
            current_depth: int -- The current depth level.

        """
        depths[node] = current_depth
        for child in tree.successors(node):
            dfs_depth_(child, current_depth + 1)

    dfs_depth_(root)
    max_depth = max(depths.values())

    # Set nodes positions recursively
    def set_positions_(
        node: str,
        next_leaf_x: int,
    ) -> int:
        """Set the positions of the nodes in the tree.

        Args:
            node: str -- The current node.
            next_leaf_x: int -- The next available x-coordinate for leaf nodes.

        Returns:
            int: The current depth level.

        """
        children = list(tree.successors(node))

        if not children:
            if node not in positions:
                positions[node] = (next_leaf_x, 0)
                next_leaf_x += 1
            return positions[node][0]

        child_positions = [
            (child, set_positions_(child, next_leaf_x))
            for child in children
        ]

        x = sum(
            pos for _, pos in child_positions
        ) / len(child_positions)
        y = max_depth - depths[node]

        positions[node] = (x, y)

        return x

    set_positions_(root, next_leaf_x)

    # Assign each node in the tree with its position
    def assign_positions_(
        node: str,
    ) -> None:
        """Assign positions to the nodes in the tree.

        Args:
            node: str -- The current node.

        """
        node_pos = positions[node]

        tree.nodes[node]["pos"] = node_pos
        tree.nodes[node]["level"] = int(node_pos[1])

        for child in tree.successors(node):
            assign_positions_(child)


    assign_positions_(root)

    # Assign a level-index to each node
    def assign_lvl_idx_(
        level: int,
    ) -> None:
        """Assign a level-index to each node in the tree.

        Args:
            level: int -- The level.

        """
        nodes_at_lvl = [
            node for node in tree.nodes()
            if tree.nodes[node]["level"] == level
        ]
        x_positions_at_lvl = {
            node: tree.nodes[node]["pos"][0]
            for node in nodes_at_lvl
        }

        x_positions_at_lvl = dict(
            sorted(
                x_positions_at_lvl.items(),
                key=lambda x: x[1],
            ),
        )
        for idx, node_ in enumerate(x_positions_at_lvl):
            tree.nodes[node_]["index"] = idx

    max_level = max(
        tree.nodes[node]["level"]
        for node in tree.nodes()
    )

    for level in range(max_level + 1):
        assign_lvl_idx_(level)

    # Assign the text span that corresponds to each node
    for leaf in leaves:
        tree.nodes[leaf]["text_span"] = [[tokens.index(node2label[leaf]["value"].replace("\\<", "<").replace("\\>", ">"))]]

    def assign_lvl_text_span_(
        level: int,
    ) -> None:
        """Assign a level-text-span to each node in the tree.

        Args:
            level: int -- The level.

        """
        nodes_at_lvl = [
            node for node in tree.nodes() if tree.nodes[node]["level"] == level
        ]

        for node in nodes_at_lvl:
            children = list(
                tree.successors(node),
            )
            children = sorted(
                children,
                key=lambda x: tree.nodes[x]["index"],
            )
            tokens = [
                tree.nodes[child]["text_span"]
                for child in children
            ]
            tree.nodes[node]["text_span"] = tokens

    for level in range(1, max_level+1):
        assign_lvl_text_span_(level)

    def flatten_(
        x: list | str | float | None,
    ) -> list:
        """Flatten a nested list or return a single value.

        Args:
            x: list | str | float | None -- The input to flatten.

        Returns:
            list -- Flattened list or a list containing the single value.

        """
        if isinstance(x, list):
            return [item for sublist in x for item in flatten_(sublist)]

        return [x]

    for node in tree.nodes():
        tree.nodes[node]["text_span"] = flatten_(tree.nodes[node]["text_span"])


def text_span_idxs_to_str(
    text_span_idxs: str,
    tokens: list[str],
) -> str:
    """Convert text span indices to a string containing the text span.

    Args:
        text_span_idxs: str -- String representation of the text span indices.
        tokens: list[str] -- List of tokens.

    Returns:
        str -- String representation of the text span.

    """
    text_span_idxs = ast.literal_eval(text_span_idxs)
    text_span = ""
    for idx in sorted(text_span_idxs):
        text_span += "|" + tokens[idx]
    return text_span.replace("|", " ").lstrip()


def get_attributions_by_level(
    balanced_tree_nx: nx.Graph,
    tokens: list[str],
    directionless: bool = True,  # noqa: FBT001, FBT002
    include_whole_sentence: bool = False,  # noqa: FBT001, FBT002
) -> dict[int, dict[str, float]]:
    """Get attributions by level from the balanced tree.

    Args:
        balanced_tree_nx (networkx.Graph): The balanced tree in NetworkX format.
        tokens (list): List of tokens from the original text.
        directionless (bool): Whether to consider directionless attributions.
        include_whole_sentence (bool): Whether to include the whole sentence in the attributions.

    Returns:
        dict: A dictionary where keys are levels and values are lists of tuples containing text spans and their corresponding values.

    """
    attributions_by_lvl = {}

    height = max(
        [int(balanced_tree_nx.nodes[node]["level"]) for node in balanced_tree_nx.nodes()],
    )

    tokens = positional_encode_tokens(tokens)

    for lvl in range(1, height if not include_whole_sentence else height + 1):
        nodes_at_lvl = [
            node for node in balanced_tree_nx.nodes() if int(balanced_tree_nx.nodes[node]["level"]) == lvl
        ]
        # Sort the nodes at the current level by their index
        nodes_at_lvl = sorted(
            nodes_at_lvl,
            key=lambda x: balanced_tree_nx.nodes[x]["index"],
        )

        if directionless:
            attributions_at_lvl = {
            balanced_tree_nx.nodes[node]["text_span"]: float(balanced_tree_nx.nodes[node]["value"])
                for node in nodes_at_lvl
            }
        else:
            attributions_at_lvl = {
                balanced_tree_nx.nodes[node]["text_span"]: float(balanced_tree_nx.nodes[node]["div_dir"])*float(balanced_tree_nx.nodes[node]["value"])
                for node in nodes_at_lvl
            }

        attributions_by_lvl[lvl] = attributions_at_lvl

    return attributions_by_lvl


def compute_comprehensiveness_higher_order(  # noqa: PLR0913
    model_clf: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokens: list[str],
    attribution_values: dict,
    threshold: float,
    device: torch.DeviceObjType,
    *,
    original_probabilities: np.ndarray | None = None,
    predicted_class: int | None = None,
) -> float:
    """Compute the comprehensiveness of the attributions.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to use for prediction.
        tokenizer: AutoTokenizer -- Tokenizer to use for encoding the sample.
        tokens: list[str] -- Sample text to predict.
        attribution_values: dict -- Dictionary containing the attribution values.
        threshold: float -- Threshold value.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).
        *
        original_probabilities: np.ndarray -- Original probabilities of the model.
        predicted_class: int -- Predicted class of the model.

    Returns:
        float -- Comprehensiveness score.

    """
    filtered_attributions = filter_attributions_w_threshold(
        attribution_values,
        threshold,
    )

    masked_tokens = []
    for token_idxs, value in filtered_attributions.items():
        token_idxs_ = ast.literal_eval(token_idxs)
        for token_idx in token_idxs_:
            token = tokens[token_idx]
            masked_tokens.append(token if not value > 0 else "[PAD]")

    if original_probabilities is None or predicted_class is None:
        # Predict the probabilities of the original input
        original_probabilities = predict_sample(
            model_clf,
            tokenizer,
            tokens,
            device,
            model_to_gpu_mode=False,
        )
        predicted_class = original_probabilities.argmax()

    # Predict the probabilities of the masked input
    masked_probabilities = predict_sample(
        model_clf,
        tokenizer,
        masked_tokens,
        device,
        model_to_gpu_mode=False,
    )

    # Calculate the comprehensiveness score
    return original_probabilities[predicted_class] - masked_probabilities[predicted_class]

def compute_sufficiency_higher_order(  # noqa: PLR0913
    model_clf: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    tokens: list[str],
    attribution_values: dict,
    threshold: float,
    device: torch.DeviceObjType,
    *,
    original_probabilities: np.ndarray | None = None,
    predicted_class: int | None = None,
) -> float:
    """Compute the sufficiency of the attributions.

    Args:
        model_clf: AutoModelForSequenceClassification -- Model to use for prediction.
        tokenizer: AutoTokenizer -- Tokenizer to use for encoding the sample.
        tokens: list[str] -- Sample text to predict.
        attribution_values: dict -- Dictionary containing the attribution values.
        threshold: float -- Threshold value.
        device: torch.DeviceObjType -- Device to use (GPU or CPU).
        *
        original_probabilities: np.ndarray -- Original probabilities of the model.
        predicted_class: int -- Predicted class of the model.

    Returns:
        float -- Comprehensiveness score.

    """
    filtered_attributions = filter_attributions_w_threshold(
        attribution_values,
        threshold,
    )

    masked_tokens = []
    for token_idxs, value in filtered_attributions.items():
        token_idxs_ = ast.literal_eval(token_idxs)
        for token_idx in token_idxs_:
            token = tokens[token_idx]
            masked_tokens.append(token if value > 0 else "[PAD]")

    if original_probabilities is None or predicted_class is None:
        # Predict the probabilities of the original input
        original_probabilities = predict_sample(
            model_clf,
            tokenizer,
            tokens,
            device,
            model_to_gpu_mode=False,
        )
        predicted_class = original_probabilities.argmax()

    # Predict the probabilities of the masked input
    masked_probabilities = predict_sample(
        model_clf,
        tokenizer,
        masked_tokens,
        device,
        model_to_gpu_mode=False,
    )

    # Calculate the comprehensiveness score
    return original_probabilities[predicted_class] - masked_probabilities[predicted_class]

def correct_faithfulness(
    faithfulness_by_level: dict,
    support_by_level: dict,
    span_coverages_by_level: dict,
    max_level: int,
) -> dict:
    """Correct the biases in the comprehensiveness and sufficiency curves.

    Args:
        faithfulness_by_level (dict): Dictionary with the faithfulness values for each level.
        support_by_level (dict): Dictionary with the support for each level.
        span_coverages_by_level (dict): Dictionary with the span coverages for each level.
        max_level (int): The maximum level reached by the trees.

    Returns:
        dict: Dictionary with the corrected faithfulness values for each level.

    """
    corrected_faithfulness = {}

    norm_factor = 0
    for lvl in range(1, max_level):
        norm_factor += support_by_level[lvl] / span_coverages_by_level[lvl]

    for lvl, faithfulness in faithfulness_by_level.items():
        if not faithfulness.size:
            continue
        if len(faithfulness.shape) > 1:
            faithfulness = np.mean(faithfulness, axis=0)  # noqa: PLW2901

        # Fraction of trees that have a node at that level
        support_correction = support_by_level[lvl] / span_coverages_by_level[lvl]

        support_correction /= norm_factor

        # Correct the bias
        corrected_faithfulness[lvl] = (
            support_correction * faithfulness
        )

    return corrected_faithfulness


def plot_local_explanation(  # noqa: C901, PLR0912, PLR0913, PLR0915
    attributions_by_level: dict[int, dict[str, float]],
    tokens: list[str],
    true_class: int,
    predicted_class: int,
    hbox: float = 1.0,
    wspace: float = 0.5,
    hspace: float = 0.65,
    fontsize: float = 12,
    ax: plt.Axes = None,
    figsize: tuple[float, float] = None,  # noqa: RUF013
    max_word_length: int = 15,
    color_range: tuple[float, float] = (-0.65, 0.65),
    show_scores: bool = False,  # noqa: FBT001, FBT002
    score_precision: int = 2,
    show_level_grid: bool = True,  # noqa: FBT001, FBT002
    class_labels: tuple[str, str] = ("Class 0", "Class 1"),
    cbar_width: float = 0.02,
    cbar_pad: float = 0.02,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot hierarchical explanations with optimized colorbar positioning.

    Args:
        attributions_by_level (dict): Dictionary with the attributions for each level.
        tokens (list): List of tokens from the original text.
        true_class (int): True class of the sample.
        predicted_class (int): Predicted class of the sample.
        hbox (float): Height of each box.
        wspace (float): Width space between boxes.
        hspace (float): Height space between boxes.
        fontsize (float): Font size for the text.
        ax (plt.Axes): Matplotlib Axes object to plot on.
        figsize (tuple[float, float]): Figure size.
        max_word_length (int): Maximum word length for display.
        color_range (tuple[float, float]): Color range for the colormap.
        show_scores (bool): Whether to show scores on the plot.
        score_precision (int): Precision for displaying scores.
        show_level_grid (bool): Whether to show grid lines for levels.
        class_labels (tuple[str, str]): Class labels for the plot.
        cbar_width (float): Width of the colorbar.
        cbar_pad (float): Padding for the colorbar.

    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects for the plot.

    """
    if figsize is None:
        figsize = (len(tokens) * 1.5, len(attributions_by_level) * 1.5)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        ax.set_axis_off()
    else:
        fig = ax.figure

    if predicted_class == true_class:
        correction_factor = 1
        if predicted_class == 0:
            cmap_name = "coolwarm_r"
            class_labels = (class_labels[1], class_labels[0])
        else:
            cmap_name = "coolwarm"

    else:  # noqa: PLR5501
        if predicted_class == 0:
            cmap_name = "coolwarm_r"
            class_labels = (class_labels[1], class_labels[0])
            correction_factor = -1
        else:
            cmap_name = "coolwarm"
            correction_factor = 1

    max_level = max(attributions_by_level.keys())
    total_tokens = len(tokens)

    # Adaptive sizing
    base_fontsize = max(8, min(12, 120 / total_tokens)) if fontsize is None else fontsize
    hspace = hspace * (1 + max_level / 10)
    wspace = wspace * (1 + total_tokens / 50)

    if show_scores:
        hbox *= 1.3

    # Color map setup
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=color_range[0], vmax=color_range[1])

    # Store positions and maintain order
    level_data = {lvl: {"positions": [], "order": []} for lvl in range(1, max_level + 1)}
    max_width = 0

    def flatten_(
        x: list | str | float | None,
    ) -> list:
        """Flatten a nested list or return a single value.

        Args:
            x: list | str | float | None -- The input to flatten.

        Returns:
            list -- Flattened list or a list containing the single value.

        """
        if isinstance(x, list):
            return [item for sublist in x for item in flatten_(sublist)]

        return [x]

    def get_text_color_(
        bg_color: tuple[float, float, float, float],
    ) -> str:
        """Get the text color based on the background color luminance.

        Args:
            bg_color: tuple -- Background color in RGBA format.

        Returns:
            str: "white" or "black" based on the luminance of the background color.

        """
        r, g, b, _ = bg_color

        luminance = 0.299 * r + 0.587 * g + 0.114 * b

        return "white" if luminance < 0.5 else "black"  # noqa: PLR2004

    def get_ordered_spans_(
        level: int,
    ) -> list[str]:
        """Get the ordered spans for a specific level.

        Args:
            level: int -- The level to get the spans for.

        Returns:
            list[str] -- List of text spans sorted by their minimum index.

        """
        if level not in attributions_by_level:
            return []

        if not attributions_by_level[level]:
            return []
        spans = list(attributions_by_level[level].keys())

        return sorted(spans, key=lambda x: min(flatten_(ast.literal_eval(x))))

    # Draw level grid lines
    if show_level_grid:
        for level in range(1, max_level + 1):
            y_pos = level * (hbox + hspace) + hbox/2
            ax.axhline(y=y_pos, color="gray", linestyle="--", linewidth=1.2, zorder=1)
            ax.text(
                -1.0, y_pos, f"Level {level}",
                fontsize=base_fontsize*0.9, ha="right", va="center",
                color="gray", zorder=2,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 2, "alpha": 1.0},
            )

    # Plot all levels
    for ylevel in range(1, max_level + 1):
        x = 0 if ylevel == 1 else None

        for text_span in get_ordered_spans_(ylevel):
            score = attributions_by_level[ylevel][text_span]
            word = text_span_idxs_to_str(text_span, tokens)
            display_word = word[:max_word_length] + "..." if len(word) > max_word_length else word

            if ylevel == 1:
                width = base_fontsize * len(display_word) * 0.12 + 1.5
            else:
                span_indices = ast.literal_eval(text_span)
                flat_span = flatten_(span_indices)
                child_rects = []
                for child_span, (x1, x2) in level_data[ylevel - 1]["positions"]:
                    if all(idx in flat_span for idx in flatten_(child_span)):
                        child_rects.append((x1, x2, min(flatten_(child_span))))

                if not child_rects:
                    continue

                child_rects.sort(key=lambda x: x[2])
                x_min = child_rects[0][0]
                x_max = child_rects[-1][1]
                width = x_max - x_min
                x = x_min

            bg_color = cmap(norm(correction_factor*score))
            text_color = get_text_color_(bg_color)

            rect = plt.Rectangle(
                (x, ylevel * (hbox + hspace)),
                width, hbox,
                edgecolor="black",
                facecolor=bg_color,
                linewidth=0.8,
                zorder=10,
            )
            ax.add_patch(rect)

            text_y_base = ylevel * (hbox + hspace) + hbox/2
            text_params = {
                "fontsize": base_fontsize,
                "ha": "center",
                "va": "center",
                "color": text_color,
                "zorder": 20,
                "fontweight": "bold",
            }

            if show_scores:
                ax.text(
                    x + width/2, text_y_base + hbox*0.2,
                    f"{correction_factor*score:+.{score_precision}f}",
                    **{k: v for k, v in text_params.items() if k not in ["fontsize", "fontweight"]},
                    fontsize=base_fontsize*0.8,
                )
                word_y = text_y_base - hbox*0.15
            else:
                word_y = text_y_base

            ax.text(
                x + width/2, word_y,
                display_word,
                **text_params,
                bbox={"facecolor": "none", "edgecolor": "none", "pad": 1, "alpha": 0.5},
            )

            level_data[ylevel]["positions"].append((ast.literal_eval(text_span), (x, x + width)))
            level_data[ylevel]["order"].append(min(flatten_(ast.literal_eval(text_span))))

            if ylevel < max_level:
                ax.annotate(
                    "",
                    xy=(x + width/2, (ylevel + 1) * (hbox + hspace) + 0.05),
                    xytext=(x + width/2, ylevel * (hbox + hspace) + hbox - 0.05),
                    arrowprops={"arrowstyle": "->", "lw": 1.25, "color": "black"},
                    zorder=5,
                )

            if ylevel == 1:
                x += width + wspace
                max_width = max(max_width, x)

    # Set bounds with space for colorbar
    padding = 1.0
    left_padding = 2.0 if show_level_grid else padding
    ax.set_xlim(-left_padding, max_width + padding)
    total_height = (max_level + 1) * (hbox + hspace)
    ax.set_ylim(0, total_height)

    # Create colorbar with manual positioning
    fig_width_inches = fig.get_size_inches()[0]
    tree_width_inches = (max_width + left_padding + padding) * (fig_width_inches / (max_width + left_padding + padding + cbar_width + cbar_pad))

    # Create colorbar axes first
    cax = fig.add_axes([
        0.95 - cbar_width,  # left position
        0.2,               # bottom position
        cbar_width,         # width
        0.7,                # height
    ])

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
        sm,
        cax=cax,
        label=r"Dividend Direction $\times$ Attribution Score",
        extend="both",
        orientation="vertical",
    )

    cbar.ax.set_ylim(color_range[0], color_range[1])

    # Add class labels
    cbar.ax.text(0.5, color_range[1]+0.11, class_labels[1],
                ha="center", va="bottom", transform=cbar.ax.get_yaxis_transform())
    cbar.ax.text(0.5, color_range[0]-0.11, class_labels[0],
                ha="center", va="top", transform=cbar.ax.get_yaxis_transform())

    # Adjust colorbar ticks
    cbar.ax.set_yticks(np.arange(color_range[0], color_range[1] + 0.1, 0.25))
    cbar.ax.set_yticklabels([rf"${tick:+.2f}$" for tick in np.arange(color_range[0], color_range[1] + 0.1, 0.25)], fontsize=fontsize*0.6)


    # Adjust figure size to accommodate colorbar
    fig.set_size_inches(tree_width_inches + (cbar_width + cbar_pad) * fig_width_inches, fig.get_size_inches()[1])

    return fig, ax
