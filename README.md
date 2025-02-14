# EMELD: Enhanced Multimodal Emotion Lines Dataset

## Introduction

Understanding group conversation dynamics is essential in computational linguistics. Existing datasets like MELD primarily focus on emotions and sentiments but lack speaker role annotations. EMELD enhances MELD by introducing speaker roles, enabling a more nuanced analysis of group conversations.

This project implements multiple approaches for speaker role classification, incorporating contextual and social cues to improve accuracy. Additionally, it provides statistical analysis and visualization tools to evaluate model performance.

## Features

- Implements three speaker role classification approaches.
- Uses Mistral LLM via Ollama for role assignment.
- Generates analysis reports including Cohen's Kappa agreement and accuracy metrics.
- Provides visualization tools for result comparison.
- Saves output to temporary files to prevent overriding existing results.

## Requirements

Ensure you have the following dependencies installed before running the project:

```bash
pip install -r requirements.txt
```

Required packages:
```
pandas
numpy
torch
matplotlib
seaborn
scikit-learn
tqdm
huggingface_hub
plotly
networkx
langchain-ollama
ollama
tabulate
```

### Additional Requirements

- **CUDA:** Required for running models efficiently on GPU.
- **Ollama:** The Ollama framework is used for local LLM inference. Follow the steps below to install and configure it.

### Setting Up Ollama

1. Install Ollama by following the official guide: [Ollama Installation](https://ollama.com/docs)
2. Ensure the Mistral model is available within Ollama:
   ```bash
   ollama pull mistral
   ```
3. Confirm that Ollama is running correctly:
   ```bash
   ollama list
   ```

## Project Structure

```
EMELD/
│── main.py                         # Entry point to run the approaches and analysis
│── ollama_setup.py                 # LLM prompt setup using Ollama
│── requirements.txt                # List of dependencies
│── config.py                       # Configuration file with paths and constants
│── utils.py                        # Helper functions for processing and plots
│── base_role_approach.py           # Implementation base class that is used for approaches 2 and 3
│── approach1.py                    # Implementation of Approach 1
│── approach2.py                    # Implementation of Approach 2 (including hash speaker names option)
│── approach3.py                    # Implementation of Approach 3 (including hash speaker names option)
│── analysis.py                     # Implementation of all analysis functions
│── Manual annotation tool.ipynb    # Interactive tool for manual annotating test data
│── data/                           # Contains train and test datasets
│   ├── train_sent_emo.csv    
│   ├── test_sent_emo.csv     
│── results/                        # Stores results of model runs and annotations (elaborated below)
│   ├── Final Approaches Annotations/
│   │   ├── test_approach1.csv              # Final annotation file for Approach 1
│   │   ├── test_approach2.csv              # Final annotation file for Approach 2
│   │   ├── test_approach3.csv              # Final annotation file for Approach 3
│   │   ├── test_approach2_hashed.csv       # Final annotation file for Approach 2 (hashed speaker names)
│   │   ├── test_approach3_hashed.csv       # Final annotation file for Approach 3 (hashed speaker names)
│   ├── Temp Approaches Annotations for Testing/
│   │   ├── (Temporary annotation files for local runs, avoiding overwriting final results)
│   ├── Manual Annotations/
│   │   ├── annotated_test_sent_emo.csv # Manual annotations with assigned roles per utterance
```
## Results Folder Structure

The `results/` directory contains multiple subdirectories that store the outputs of the model approaches and manual annotations:

- **Final Approaches Annotations/**
  - Stores the final annotation files for each approach, including:
    - `test_approach1.csv`
    - `test_approach2.csv`
    - `test_approach3.csv`
    - `test_approach2_hashed.csv`
    - `test_approach3_hashed.csv`

- **Temp Approaches Annotations for Testing/**
  - This directory saves the temporary annotation results when running `main.py` locally, ensuring final results remain unaltered.

- **Manual Annotations/**
  - Contains `annotated_test_sent_emo.csv`, where manual annotations are recorded. Each project member has a dedicated annotation column:
    - `Amit Role`
    - `Noa Role`
    - `Guy Role`
    - `Omer Role`
  - Other columns include features from the test dataset, aiding in deeper analysis.

## Manual Annotation Tool

The project includes a Jupyter Notebook, `Manual annotation tool.ipynb`, designed to streamline the annotation process. This interactive tool enables users to annotate speaker roles efficiently by looping over the shared `annotated_test_sent_emo.csv` file and allowing users to update their respective columns directly. This ensures consistency and reduces manual overhead in labeling utterances.

## Approaches

We tested three different approaches to classify speaker roles, each leveraging different levels of contextual information and metadata.

### Approach 1: Sentence-Only Approach (Naive Baseline)

This approach classifies speaker roles based solely on individual sentences, without incorporating dialogue context or metadata. It serves as a baseline model, providing a simple yet limited view of speaker roles.

### Approach 2: Dialogue-Aware Approach (Full-Dialogue Baseline)

This approach takes into account the entire conversation, allowing the model to consider how different utterances interact. By leveraging full-dialogue context, this method improves role classification compared to the naive baseline.

### Approach 3: Context-Enriched Dialogue-Aware Approach (Contextual Baseline)

Building upon the dialogue-aware method, this approach integrates additional metadata such as response duration, word and letter counts, and expressed sentiments/emotions. This added context enhances the model’s ability to recognize subtle speaker role distinctions, making it the most advanced of the three methods.


## Running the Project

The project allows you to run different role classification approaches and analyze the results.

### Running Approaches

To execute the three approaches, run:
```bash
python main.py
```
By default, this will run all three approaches and analyze the results. If you want to disable specific approaches, modify the flags in `main.py`:
```python
run_approach1 = True  # Set to False to disable Approach 1
run_approach2 = True  # Set to False to disable Approach 2
run_approach3 = True  # Set to False to disable Approach 3
analyse_results = True  # Set to False to skip result analysis
```

### Analysis and Visualization

After running the approaches, the project provides:

- **Cohen's Kappa agreement analysis** to measure annotator consistency.
- **Accuracy comparisons** between models.
- **Visualization of role classification performance** using bar charts.

## Ollama LLM Integration

The project uses **Mistral LLM** via Ollama for role classification. The relevant code is in `ollama_setup.py`:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from config import OLLAMA_MODEL

def run_llm(question, template):
    """Runs a question through the Ollama model."""
    prompt = ChatPromptTemplate.from_template(template)
    model = OllamaLLM(model=OLLAMA_MODEL)
    chain = prompt | model
    response = chain.invoke({"question": question})
    return response
```

## Notes

- The project saves results in temporary files to prevent overriding achieved results.
- Requires **CUDA support** for efficient model execution.
- **Mistral model** must be loaded in Ollama before execution.

## References
For more details on the dataset and methodologies, refer to our [paper](./Enhanced%20Multimodal%20Emotion%20Lines%20Dataset%20(E-MELD).pdf).
