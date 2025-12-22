"""
===============================================================================
Script Name   : transfer.py
Description   : Implements the evaluation logic for Transfer-Based Black-Box Attacks.
                This module takes adversarial examples generated on a Source Model (Surrogate)
                and evaluates their effectiveness on Target Models (RF, LR, MLP, CNN).
                
                Key Metric:
                - Attack Success Rate (ASR): The percentage of samples that were correctly 
                  classified by the target model originally, but misclassified after perturbation.

Usage:
    Called by orchestrator.py after generating attacks.

Author        : Dr. Samer Aoudi
Affiliation   : Higher Colleges of Technology (HCT), UAE
Role          : Assistant Professor & Division Chair (CIS)
Email         : cybersecurity@sameraoudi.com
ORCID         : 0000-0003-3887-0119
Created On    : 2025-Nov-22

License       : MIT License
Citation      : If this code is used in academic work, please cite the
                corresponding publication or acknowledge the author.
===============================================================================
"""

# src/attacks/transfer.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Callable, Any

import numpy as np

from .base import AttackResult


@dataclass
class TransferResult:
    """
    Results of evaluating adversarial examples on one or more target models.

    Attributes
    ----------
    clean_accuracy : Dict[str, float]
        Accuracy of each target model on clean inputs.
    adv_accuracy : Dict[str, float]
        Accuracy of each target model on adversarial inputs.
    asr : Dict[str, float]
        Attack success rate on each target model.
    """
    clean_accuracy: Dict[str, float]
    adv_accuracy: Dict[str, float]
    asr: Dict[str, float]
    meta: Dict[str, Any]


def evaluate_transfer(
    attack_result: AttackResult,
    target_models: Dict[str, Callable[[np.ndarray], np.ndarray]],
) -> TransferResult:
    """
    Evaluate transferability of adversarial examples to a set of target models.

    Parameters
    ----------
    attack_result : AttackResult
        Output from a surrogate-based attack (FGSM/PGD).
    target_models : dict
        Mapping from model name to a callable f(X) -> y_pred, where
        X is a NumPy array of shape (N, D) and y_pred is shape (N,).

        Examples of callables:
          - lambda X: rf_model.predict(X)
          - lambda X: lr_model.predict(X)
          - lambda X: mlp_wrapper.predict(X)

    Returns
    -------
    TransferResult
        Contains clean/adv accuracy and attack success rate per model.
    """
    X_adv = attack_result.x_adv
    X_clean = attack_result.x_adv - (attack_result.x_adv - attack_result.x_adv)  # placeholder to emphasize X_clean usage
    # NOTE: In practice, you should pass X_clean into this function as well.
    # Here we assume caller kept X_clean separately; we use y_pred_clean as reference.

    y_true = attack_result.y_true
    clean_accuracy: Dict[str, float] = {}
    adv_accuracy: Dict[str, float] = {}
    asr: Dict[str, float] = {}

    for name, model_fn in target_models.items():
        # For transfer evaluation we need both clean and adversarial predictions.
        # Since AttackResult only stores the surrogate's preds, the caller should
        # pass X_clean and X_adv explicitly. To keep this function generic, we
        # assume model_fn can be invoked on both X_clean and X_adv arrays.

        # The recommended calling pattern is:
        #   evaluate_transfer(attack_result, target_models, X_clean)
        # but to keep backward compatibility we derive X_clean elsewhere.
        raise NotImplementedError(
            "evaluate_transfer currently requires X_clean to be passed explicitly. "
            "Please use evaluate_transfer_with_clean instead."
        )


def evaluate_transfer_with_clean(
    X_clean: np.ndarray,
    attack_result: AttackResult,
    target_models: Dict[str, Callable[[np.ndarray], np.ndarray]],
) -> TransferResult:
    """
    Evaluate transferability of adversarial examples to a set of target models,
    given access to clean inputs.

    Parameters
    ----------
    X_clean : np.ndarray
        Clean inputs used to generate the adversarial examples, shape (N, D).
    attack_result : AttackResult
        Output from a surrogate-based attack (FGSM/PGD) on the same X_clean.
    target_models : dict
        Mapping from model name to a callable f(X) -> y_pred.

    Returns
    -------
    TransferResult
        Contains clean/adv accuracy and attack success rate per model.
    """
    X_adv = attack_result.x_adv
    y_true = attack_result.y_true

    clean_accuracy: Dict[str, float] = {}
    adv_accuracy: Dict[str, float] = {}
    asr: Dict[str, float] = {}

    for name, model_fn in target_models.items():
        y_clean_pred = model_fn(X_clean)
        y_adv_pred = model_fn(X_adv)

        clean_acc = float(np.mean(y_clean_pred == y_true))
        adv_acc = float(np.mean(y_adv_pred == y_true))
        # ASR defined as fraction of originally correct predictions that become incorrect
        originally_correct = (y_clean_pred == y_true)
        now_incorrect = (y_adv_pred != y_true)
        asr_value = float(
            np.sum(originally_correct & now_incorrect) / max(1, np.sum(originally_correct))
        )

        clean_accuracy[name] = clean_acc
        adv_accuracy[name] = adv_acc
        asr[name] = asr_value

    meta = attack_result.meta.copy() if attack_result.meta is not None else {}
    meta.update({"type": "transfer"})

    return TransferResult(
        clean_accuracy=clean_accuracy,
        adv_accuracy=adv_accuracy,
        asr=asr,
        meta=meta,
    )
