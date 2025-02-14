import os
import numpy as np
import pandas as pd
from tabulate import tabulate

# Project imports
from config import FINAL_SAVE_DIR, MANUAL_ANNOTATIONS_PATH
from utils import compute_kappa_agreement, compute_kappa, plot_accuracy_bar_chart, plot_distribution_by_emotion, plot_distribution_of_utterance_length, plot_distribution_of_dialogue_frequency, plot_role_distribution_per_speaker, plot_accuracy_by_utterance_length

# Load manually annotated dataset
merged_results = pd.read_csv(MANUAL_ANNOTATIONS_PATH)

# List of models
models_file_names = ['approach1', 'approach2', 'approach3', 'approach2_hashed', 'approach3_hashed']

# Merge predictions from different models
for file_name in models_file_names:
    file_path = os.path.join(FINAL_SAVE_DIR, f"test_{file_name}.csv")
    role_mapping = pd.read_csv(file_path).set_index('Sr No.')['Role'].to_dict()
    merged_results[f'{file_name} Role'] = merged_results['Sr No.'].map(role_mapping)

# Annotators' role columns
role_columns = ['Amit Role', 'Noa Role', 'Guy Role', 'Omer Role']

def resolve_majority_vote(row):
    """
    Compute majority vote based on annotators' roles.
    If there's a tie, a random choice is made among the top candidates.
    """
    counts = row[role_columns].dropna().value_counts()
    if counts.empty:
        return np.nan

    max_count = counts.max()
    candidates = counts[counts == max_count].index.tolist()
    return candidates[0] if len(candidates) == 1 else np.random.choice(candidates)

# Compute Majority Role
merged_results['Majority Role'] = merged_results.apply(resolve_majority_vote, axis=1)

# Display sample rows
print("\n" + "=" * 40)
print("Merged Dataset Sample (First 5 Rows)")
print("=" * 40)
print(tabulate(merged_results.head(), headers="keys", tablefmt="grid"))

# Compute Cohen's Kappa agreement
kappa_results = compute_kappa_agreement(merged_results, role_columns)

print("\n" + "=" * 40)
print("Cohen's Kappa Agreement Scores")
print("=" * 40)
print(tabulate(kappa_results, headers=["Annotator 1", "Annotator 2", "Kappa Score"], tablefmt="grid"))

# Compute and print final Kappa score
final_kappa = compute_kappa(merged_results, role_columns)
print("\nFinal Kappa Score (Avg of Pairs): {:.3f}".format(final_kappa))

# Define classifiers and reference column for accuracy plot
approaches_columns = ['approach1 Role', 'approach2 Role', 'approach3 Role', 'approach2_hashed Role', 'approach3_hashed Role']

def analysis_main():

    plot_accuracy_bar_chart(merged_results, approaches_columns, reference_column='Majority Role')

    role_colors = {
        'Protagonist': '#1f77b4',  # Blue
        'Supporter': '#2ca02c',    # Green
        'Neutral': '#ff7f0e',      # Orange
        'Gatekeeper': '#9467bd',   # Purple
        'Attacker': '#d62728'      # Red
    }

    plot_distribution_by_emotion(merged_results, role_colors, reference_column='approach3 Role')

    plot_distribution_of_utterance_length(merged_results, reference_column='Utterance')

    plot_distribution_of_dialogue_frequency(merged_results)

    plot_role_distribution_per_speaker(merged_results, approaches_columns, role_colors)

    plot_accuracy_by_utterance_length(merged_results)