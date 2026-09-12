import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

"""
metrics.py

This module provides functions to calculate evaluation metrics for both
intent classification and escalation decision tasks.
"""

def evaluate_classification(y_true, y_pred, labels):
    """
    Calculates accuracy, precision, recall, f1, and confusion matrix for intent classification.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return {
        "accuracy": acc,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist()
    }

def evaluate_escalation(y_true, y_pred):
    """
    Calculates metrics for the AUTO vs ESCALATE decision, with special focus
    on ESCALATE recall.
    """
    labels = ["AUTO", "ESCALATE"]
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="ESCALATE", average='binary', zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="ESCALATE", average='binary', zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="ESCALATE", average='binary', zero_division=0)
    
    return {
        "accuracy": acc,
        "precision_escalate": prec,
        "recall_escalate": rec,
        "f1_escalate": f1
    }
