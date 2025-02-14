import seaborn as sns
import pandas as pd
import numpy as np
import torch
from config import TRAIN_PATH, TEST_PATH
from sklearn.metrics import cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_dataset():
    """Loads dataset CSV files into pandas DataFrames."""
    return pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)


def compute_kappa_agreement(df, columns):
    """
    Compute Cohen's Kappa agreement between each pair of columns in the given list.
    """
    kappa_scores = []
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            kappa = cohen_kappa_score(df[col1], df[col2])
            kappa_scores.append((col1, col2, f"{kappa:.3f}"))

    return kappa_scores


def compute_kappa(df, annotators):
    """
    Compute the average Cohen's Kappa agreement among multiple annotators.
    """
    valid_rows = df.dropna(subset=annotators)  # Ensure non-null comparisons
    labels = [valid_rows[col].astype(str).tolist() for col in annotators]

    # Compute pairwise kappa and average
    kappa_scores = []
    for i in range(len(annotators)):
        for j in range(i + 1, len(annotators)):
            kappa_scores.append(cohen_kappa_score(labels[i], labels[j]))

    return np.mean(kappa_scores) if kappa_scores else 0



def plot_accuracy_bar_chart(df, classifiers, reference_column):
    """
    Improved visualization for classifier accuracy compared to the reference column.
    """
    accuracies = {col: accuracy_score(df[reference_column], df[col]) for col in classifiers}

    # Sorting classifiers by accuracy (Descending Order)
    sorted_accuracies = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    classifiers_sorted, accuracy_values = zip(*sorted_accuracies)

    # Dynamic color mapping
    colors = plt.cm.Blues(np.linspace(0.4, 0.8, len(classifiers_sorted)))

    # Plot setup
    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.bar(classifiers_sorted, accuracy_values, color=colors, alpha=0.85, edgecolor='black')

    # Set labels and title
    ax.set_xlabel("Classifier", fontsize=13, fontweight='bold')
    ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold')
    ax.set_title(f"Classifier Accuracy Compared to '{reference_column}'", fontsize=14, fontweight='bold', pad=15)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=11)

    # Add space below the x-axis labels
    plt.subplots_adjust(bottom=0.28)

    # Add value labels on top of bars
    for bar, accuracy in zip(bars, accuracy_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                f"{accuracy:.3f}", ha='center', fontsize=12, fontweight='bold', color='black')

    # Grid styling
    ax.yaxis.grid(True, linestyle='dashed', alpha=0.5)
    ax.set_axisbelow(True)  # Ensure grid lines are behind bars

    # Adjust y-limit dynamically
    ax.set_ylim(0, max(accuracy_values) + 0.05)

    plt.show()



def plot_distribution_by_emotion(df, role_colors, reference_column):
    """
    Plots the distribution of roles by emotion.
    """
    # Compute the percentage distribution for 'Majority Role' by 'Emotion'
    grouped_roles_by_emotion = (
        df.groupby('Emotion')[reference_column]
        .value_counts(normalize=True)
        .mul(100)
        .reset_index(name='Percentage')
    )

    # Sort roles within each emotion separately by percentage (descending)
    grouped_roles_by_emotion = grouped_roles_by_emotion.sort_values(by=['Emotion', 'Percentage'], ascending=[True, False])

    # Function to preserve role order within each emotion
    def order_roles_per_emotion(df):
        df[reference_column] = pd.Categorical(df[reference_column],
                                            categories=df.sort_values('Percentage', ascending=False)[reference_column].tolist(),
                                            ordered=True)
        return df

    # Apply the function for per-emotion ordering
    grouped_roles_by_emotion = grouped_roles_by_emotion.groupby('Emotion', group_keys=False).apply(order_roles_per_emotion)

    # Plot each emotion separately to maintain independent role order
    plt.figure(figsize=(18, 6))
    sns.set_style("whitegrid")

    for i, emotion in enumerate(grouped_roles_by_emotion['Emotion'].unique()):
        emotion_df = grouped_roles_by_emotion[grouped_roles_by_emotion['Emotion'] == emotion]

        plt.subplot(1, len(grouped_roles_by_emotion['Emotion'].unique()), i + 1)
        ax = sns.barplot(
            x=reference_column,
            hue=reference_column,
            y='Percentage',
            data=emotion_df,
            palette=role_colors,
            legend=False
        )

        plt.title(emotion, fontsize=14, fontweight='bold')
        plt.xlabel('')
        plt.ylabel('Percentage (%)' if i == 0 else '')
        plt.xticks(rotation=45)

        # Add percentage labels on bars with slight vertical offset
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f"{p.get_height():.1f}%",
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', fontsize=8, color='black')

    # Adjust layout to prevent title from getting cut off
    plt.subplots_adjust(top=0.85)  # Reserves space for title

    # Add title with correct positioning
    plt.suptitle(f"Role Distribution by Emotion of '{reference_column}'", 
                 fontsize=16, fontweight='bold', y=0.98)  # Moves title higher

    plt.tight_layout(rect=[0, 0, 1, 0.90])  # Adjusts layout to keep space

    plt.show()



def plot_distribution_of_utterance_length(df, reference_column):
    """
    Plots the distribution of utterance lengths (word count per utterance).
    """
    # Calculate utterance lengths
    df['Utterance_Length'] = df[reference_column].apply(lambda x: len(x.split()))

    # Create the distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Utterance_Length'], kde=True, color="#4C72B0", edgecolor="black", alpha=0.8)

    # Improve aesthetics
    plt.xlabel('Number of Words per Utterance', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Adjust layout to reserve space for the title
    plt.subplots_adjust(top=0.85)

    # Add title with consistent formatting
    plt.suptitle(f"Distribution of Utterance Lengths", 
                 fontsize=16, fontweight='bold', y=0.98)  # Ensures title is not cut off

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()




def plot_distribution_of_dialogue_frequency(df):
    """
    Plots the distribution of dialogue lengths (number of utterances per dialogue).
    """
    # Count utterances per dialogue
    dialogue_lengths = df.groupby("Dialogue_ID")["Utterance_ID"].count()

    # Plot distribution
    plt.figure(figsize=(10, 6))
    plt.hist(dialogue_lengths, bins=25, edgecolor="black", alpha=0.8, color="#E0AC4F")

    # Improve aesthetics
    plt.xlabel("Number of Utterances per Dialogue", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)

    # Adjust layout to reserve space for the title
    plt.subplots_adjust(top=0.85)

    # Add title with consistent formatting
    plt.suptitle("Distribution of Dialogue Lengths", fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.show()



def plot_role_distribution_per_speaker(df, role_columns, role_colors):
    """
    Plots the role distribution for the top 6 speakers:
    1. A single plot for 'approach1 Role' with the title above the legend.
    2. A 2x2 subplot grid for the remaining role assignment approaches with a horizontal legend above.
    """
    # Identify the top 6 most frequent speakers
    top_speakers = df['Speaker'].value_counts().nlargest(6).index

    # Filter the DataFrame to only include the top speakers
    df_top_speakers = df[df['Speaker'].isin(top_speakers)].copy()

    # Define a fixed role order to maintain consistent color mapping across plots
    fixed_role_order = ['Protagonist', 'Supporter', 'Neutral', 'Gatekeeper', 'Attacker']

    ### PLOT 1: Single plot for 'approach1 Role' ###
    fig1, ax1 = plt.subplots(figsize=(10, 7))  # Adjusted figure size

    role_col = 'approach1 Role'
    role_distribution = df_top_speakers.groupby(['Speaker', role_col], observed=True).size().unstack(fill_value=0)

    # Reorder columns to match the fixed role order (ensuring consistency)
    role_distribution = role_distribution.reindex(columns=fixed_role_order, fill_value=0)

    # Convert to percentage
    role_distribution = role_distribution.div(role_distribution.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    role_distribution.plot(
        kind='bar',
        stacked=True,
        ax=ax1,
        color=[role_colors.get(role, "#999999") for role in fixed_role_order]  # Assign colors in fixed order
    )

    # Formatting
    ax1.set_ylabel("Percentage (%)", fontsize=12)
    ax1.set_xticklabels(role_distribution.index, rotation=45, fontsize=12)
    ax1.set_yticks(ax1.get_yticks())
    ax1.set_yticklabels([f"{tick:.0f}" for tick in ax1.get_yticks()], fontsize=12)

    # Remove legend inside the plot
    ax1.get_legend().remove()

    # **Add title ABOVE the legend, but lower it so it stays visible**
    fig1.suptitle(f"Distribution of '{role_col}' for Top 6 Speakers", fontsize=16, fontweight="bold", y=0.98)

    # **Move legend BELOW the title (but not too high)**
    handles, labels = ax1.get_legend_handles_labels()
    fig1.legend(
        handles, labels, title="Role", loc="upper center", ncol=len(fixed_role_order), frameon=False,
        bbox_to_anchor=(0.5, 0.95), fontsize=12, title_fontsize=14
    )

    # **Adjust layout to create space for title and legend**
    fig1.subplots_adjust(top=0.80)  # Pushes plot down but keeps everything visible

    plt.show()

    ### PLOT 2: 2x2 grid for the other four role assignment approaches ###
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 11), sharey=True)
    axes2 = axes2.flatten()  # Flatten for easy iteration

    for i, role_col in enumerate(['approach2 Role', 'approach2_hashed Role', 'approach3 Role', 'approach3_hashed Role']):
        # Compute the percentage distribution of roles for each speaker
        role_distribution = df_top_speakers.groupby(['Speaker', role_col], observed=True).size().unstack(fill_value=0)

        # Reorder columns to match the fixed role order (ensuring consistency)
        role_distribution = role_distribution.reindex(columns=fixed_role_order, fill_value=0)

        # Convert to percentage
        role_distribution = role_distribution.div(role_distribution.sum(axis=1), axis=0) * 100

        # Plot stacked bar chart
        role_distribution.plot(
            kind='bar',
            stacked=True,
            ax=axes2[i],
            color=[role_colors.get(role, "#999999") for role in fixed_role_order]  # Assign colors in fixed order
        )

        # Formatting
        axes2[i].set_ylabel("Percentage (%)", fontsize=12)
        axes2[i].set_title(f"Distribution of '{role_col}' for Top 6 Speakers", fontsize=12, fontweight="bold", pad=20)
        axes2[i].set_xticklabels(role_distribution.index, rotation=45, fontsize=12)
        axes2[i].set_yticks(axes2[i].get_yticks())
        axes2[i].set_yticklabels([f"{tick:.0f}" for tick in axes2[i].get_yticks()], fontsize=12)

        # Remove individual legends
        axes2[i].get_legend().remove()

    # Create a single legend for all subplots
    handles, labels = axes2[0].get_legend_handles_labels()
    legend = fig2.legend(
        handles, labels, title="Role", loc="upper center", ncol=len(fixed_role_order), frameon=False,
        bbox_to_anchor=(0.5, 0.95), fontsize=12, title_fontsize=14
    )

    # Make the legend background slightly transparent
    legend.get_frame().set_alpha(0.5)

    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.9])

    # Add overall title to the figure
    plt.suptitle("Role Distribution Across Different Approaches for Top 6 Speakers", 
                 fontsize=16, fontweight="bold", y=0.98)

    plt.show()



def plot_accuracy_by_utterance_length(df):
    """
    Improved visualization for accuracy trends across utterance length buckets.
    """
    # Add a column for utterance length (word count)
    df['Utterance_Length'] = df['Utterance'].str.split().str.len()

    # Define new sentence length bins and labels
    bins = [0, 3, 6, 10, 15, np.inf]
    labels = ['1-3', '4-6', '7-10', '11-15', '16+']
    df['Length_Bucket'] = pd.cut(df['Utterance_Length'], bins=bins, labels=labels)

    # Calculate accuracy for each approach per length bucket
    length_performance = {}
    for approach in ['approach1 Role', 'approach2 Role', 'approach3 Role']:
        length_performance[approach] = (
            df.groupby('Length_Bucket', observed=False).apply(  # Explicitly set observed=False
                lambda x: (x[approach] == x['Majority Role']).mean() * 100
            )
        )

    # Define colors and markers similar to the uploaded image
    colors = ['orange', 'red', 'blue']
    markers = ['o', 's', 'x']
    labels = ['Naive Baseline', 'Full-Dialogue Baseline', 'Contextual Baseline']

    # Plot setup
    plt.figure(figsize=(10, 6))
    for approach, color, marker, label in zip(length_performance.keys(), colors, markers, labels):
        plt.plot(length_performance[approach].index, length_performance[approach].values,
                 marker=marker, linestyle="-", color=color, label=label)

    # Set labels and title with improved formatting
    plt.xlabel("Utterance Length Bucket", fontsize=13, fontweight='bold')
    plt.ylabel("Accuracy Against Our Annotations", fontsize=13, fontweight='bold')
    plt.title("Accuracy Trends Across Utterance Length Buckets", fontsize=14, fontweight='bold', pad=15)

    # Customize legend placement and styling
    plt.legend(title='Approach', loc='lower center', fontsize=11, frameon=True, edgecolor='black')

    # Improve grid styling
    plt.grid(True, linestyle='dashed', alpha=0.5)

    # Show plot
    plt.show()