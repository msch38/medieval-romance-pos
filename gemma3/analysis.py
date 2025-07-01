# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:32:43 2025

@email: Matthias.Schoeffel@lmu.de
"""

# pos_evaluator.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score
)
import os
from collections import Counter

def evaluate_pos(reference_file, prediction_file, output_folder=None):
    # Load files
    df_ref = pd.read_excel(reference_file)
    df_pred = pd.read_excel(prediction_file)
    
    # Combine and clean
    combined_df = pd.concat([df_ref, df_pred], axis=1)
    
    # Identify valid UPOS tags based on the reference
    valid_tags = combined_df['POS'].unique().tolist()
    reference_tags = set(valid_tags)
    
    y_true = combined_df['POS'].tolist()
    y_pred = combined_df['upos'].tolist()
    
    # Filter out predictions with tags not in reference
    valid_indices = [i for i, tag in enumerate(y_pred) if tag in reference_tags]
    y_true_filtered = [y_true[i] for i in valid_indices]
    y_pred_filtered = [y_pred[i] for i in valid_indices]
    
    # Report unknowns
    unknown_tags = set(y_pred) - reference_tags
    print(f"\n⚠️ Ignored {len(unknown_tags)} unknown tags: {unknown_tags}")
    print(f"🔍 Evaluating on {len(valid_indices)} valid samples")

    # Accuracy
    accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    balanced_acc = balanced_accuracy_score(y_true_filtered, y_pred_filtered)

    # Macro/Micro scores
    macro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)
    macro_precision = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    micro_precision = precision_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)
    macro_recall = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    micro_recall = recall_score(y_true_filtered, y_pred_filtered, average='micro', zero_division=0)

    # Cohen's Kappa
    kappa = cohen_kappa_score(y_true_filtered, y_pred_filtered)

    # Per-class metrics
    report_dict = classification_report(y_true_filtered, y_pred_filtered, labels=valid_tags, output_dict=True, zero_division=0)
    report_text = classification_report(y_true_filtered, y_pred_filtered, digits=4, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "Label"})

    precision_arr, recall_arr, f1_arr, support_arr = precision_recall_fscore_support(
        y_true_filtered, y_pred_filtered, labels=valid_tags, zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=valid_tags)
    cm_df = pd.DataFrame(cm, index=valid_tags, columns=valid_tags)

    # Print summary
    print("\n=== Classification Report ===")
    print(report_text)
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Balanced Accuracy: {balanced_acc:.4f}")
    print(f"📐 Cohen's Kappa: {kappa:.4f}")
    print(f"🔁 Macro Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
    print(f"🔁 Micro Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}")

    # Most confused tag pairs
    most_confused = cm_df.stack().sort_values(ascending=False)
    most_confused = most_confused[most_confused.index.get_level_values(0) != most_confused.index.get_level_values(1)]
    print("\n🔄 Most confused tag pairs:")
    print(most_confused.head(5))

    # Output folder setup
    if output_folder is None:
        output_folder = os.path.splitext(os.path.basename(prediction_file))[0]
    os.makedirs(output_folder, exist_ok=True)

    # Save Excel with report and confusion matrix (in one file, multiple sheets)
    with pd.ExcelWriter(os.path.join(output_folder, "evaluation_summary.xlsx")) as writer:
        report_df.to_excel(writer, sheet_name="Classification Report", index=False)
        cm_df.to_excel(writer, sheet_name="Confusion Matrix")

    # Save plain report as .txt
    with open(os.path.join(output_folder, "classification_report.txt"), "w") as f:
        f.write(report_text)
        f.write(f"\nAccuracy: {accuracy:.4f}")
        f.write(f"\nBalanced Accuracy: {balanced_acc:.4f}")
        f.write(f"\nCohen's Kappa: {kappa:.4f}")
        f.write(f"\nMacro Precision: {macro_precision:.4f}, Recall: {macro_recall:.4f}, F1: {macro_f1:.4f}")
        f.write(f"\nMicro Precision: {micro_precision:.4f}, Recall: {micro_recall:.4f}, F1: {micro_f1:.4f}")

    # Save unknown tags
    if unknown_tags:
        with open(os.path.join(output_folder, "unknown_tags.txt"), "w") as f:
            f.write("\n".join(sorted(unknown_tags)))

    # Save confusion matrix plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "confusion_matrix.png"), dpi=300)
    plt.close()

    # Show top excluded tags
    excluded_indices = [i for i, tag in enumerate(y_pred) if tag not in reference_tags]
    excluded_tags = [y_pred[i] for i in excluded_indices]
    excluded_counts = Counter(excluded_tags)

    print(f"\n🔴 Total excluded tokens: {len(excluded_indices)}")
    print("🔍 Most frequently excluded tags:")
    for tag, count in excluded_counts.most_common(5):
        print(f"  {tag}: {count} times")

    print(f"\n✅ Results saved in '{output_folder}'")


# Usage example
if __name__ == "__main__":
    import glob

    reference_file = "../../../Llibre_reference.xlsx"
    prediction_folder = "."
    prediction_files = glob.glob(os.path.join(prediction_folder, "*.xlsx"))

    for prediction_file in prediction_files:
        # Skip the reference file itself if it's in the same folder
        if os.path.abspath(prediction_file) == os.path.abspath(reference_file):
            continue

        print(f"\n🚀 Processing: {prediction_file}")
        try:
            evaluate_pos(reference_file, prediction_file)
        except Exception as e:
            print(f"❌ Failed to evaluate {prediction_file}: {e}")
