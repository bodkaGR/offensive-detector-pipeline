from __future__ import annotations

import os


def print_results(metrics: dict, title: str = "Test Results") -> None:
    skip = {
        "confusion_matrix",
        "classification_report",
        "probs",
        "labels",
        "preds"
    }
    border = "=" * 62
    print(f"\n{border}\n  {title}\n{border}")
    for key, value in metrics.items():
        if key not in skip:
            print(f"  {key:<20}: {value}")
    print(border)
    print("\nClassification Report:")
    print(metrics["classification_report"])
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print(border)

def save_report(metrics: dict, reports_path: str) -> None:
    report_path = os.path.join(reports_path, "test_metrics.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("MPNet + TransformerEncoder - Test Metrics\n" + "=" * 50 + "\n")
        for key, value in metrics.items():
            if key not in ("probs", "labels", "preds"):
                f.write(f"{key}: {value}\n")