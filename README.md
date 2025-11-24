# On the Detectability of LLM-generated Judgments

This project implements a machine learning pipeline to detect LLM-generated judgments across multiple datasets, exploring how detection performance varies with different features, group sizes, rating scales, and judgment dimensions.

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Dataset Structure](#dataset-structure)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Tasks Implemented](#tasks-implemented)
- [Usage](#usage)
- [Results](#results)
- [References](#references)

## 🎯 Overview

This project investigates the detectability of LLM-generated judgments by:
- Building classifiers to distinguish between human and LLM-generated judgments
- Augmenting base models with linguistic and LLM-enhanced features
- Analyzing detection performance across different group sizes
- Examining the impact of rating scales and judgment dimensions on detectability

## 📚 Prerequisites

### Required Papers
Before running the code, read these papers for background:
1. **From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge** (Sections 1-2)
2. **Who's Your Judge? On the Detectability of LLM-Generated Judgments**

### Python Packages
```bash
pip install scikit-learn pandas numpy matplotlib scipy
```

Required packages:
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical operations
- `matplotlib` - Visualization
- `scipy` - Statistical functions

## 📊 Dataset Structure

### Required Data
Download all data from the provided Google Drive folder. The data should be organized as:

```
data/
├── dataset_detection/
│   ├── gpt-4o-2024-08-06_helpsteer2_train_1_grouped/
│   ├── gpt-4o-2024-08-06_helpsteer2_test_1_grouped/
│   ├── gpt-4o-2024-08-06_helpsteer3_train_1_grouped/
│   ├── gpt-4o-2024-08-06_helpsteer3_test_1_grouped/
│   ├── gpt-4o-2024-08-06_neurips_train_1_grouped/
│   ├── gpt-4o-2024-08-06_neurips_test_1_grouped/
│   ├── gpt-4o-2024-08-06_antique_train_1_grouped/
│   ├── gpt-4o-2024-08-06_antique_test_1_grouped/
│   └── [similar folders for group_size 2, 4, 8, 16]
└── features/
    ├── linguistic/
    │   ├── helpsteer2_train.csv
    │   ├── helpsteer2_test.csv
    │   └── [similar files for other datasets]
    └── llm_enhanced/
        ├── helpsteer2_train.json
        ├── helpsteer2_test.json
        └── [similar files for other datasets]
```

### Datasets

| Dataset | Judgment Dimensions |
|---------|-------------------|
| **Helpsteer2** | helpfulness, correctness, coherence, complexity, verbosity |
| **Helpsteer3** | score |
| **Neurips** | rating, confidence, soundness, presentation, contribution |
| **ANTIQUE** | ranking |

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Judgment-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download and organize data as described in [Dataset Structure](#dataset-structure)

4. **Important**: Remove columns containing "noun_verb_ratio" from feature files as specified in the project requirements.

## 📁 Project Structure

```
.
├── main.py              # Main execution script
├── data.py              # Data loading and preprocessing
├── mlmodel.py           # Model training and evaluation
├── plot.py              # Visualization functions
├── metrics.csv          # Generated metrics output
├── README.md            # This file
└── data/                # Data directory (not included)
```

### File Descriptions

- **`main.py`**: Orchestrates the entire pipeline, from data loading to model training and evaluation
- **`data.py`**: Handles data loading, preprocessing, feature extraction, and dataset transformations
- **`mlmodel.py`**: Implements classifier training, prediction, and evaluation metrics
- **`plot.py`**: Generates visualization plots for group detection analysis

## ✅ Tasks Implemented

### Task 1: Base Detector Implementation
- ✅ Loads training/test datasets
- ✅ Extracts judgment dimension fields
- ✅ Trains Logistic Regression and Random Forest classifiers
- ✅ Reports accuracy and F1 score

### Task 2: Augmented Feature Detector
- ✅ Integrates linguistic features
- ✅ Integrates LLM-enhanced features
- ✅ Trains models on combined feature sets
- ✅ Compares performance with base detector

### Task 3: Group-Level Detection
- ✅ Implements instance-level predictions
- ✅ Aggregates logits across groups (sum method)
- ✅ Evaluates group-level accuracy and F1
- ✅ Tests across group sizes: 1, 2, 4, 8, 16

### Task 4: Detectability Analysis
- ✅ **Group Size Analysis**: Varies k = 1, 2, 4, 8, 16
- ✅ **Rating Scale Analysis**: 
  - Helpsteer2: Maps 1/2→0, 3/4/5→1
  - Helpsteer3: Merges -3/-2/-1→-1, 1/2/3→1
- ✅ **Dimension Number Analysis**: Tests 1, 3, 5 dimensions for Helpsteer2 and Neurips
- ✅ **Visualization**: Generates plots similar to Figure 6 in the paper

## 💻 Usage

### Basic Execution

Run the complete pipeline:
```bash
python main.py
```

This will:
1. Load and preprocess all datasets
2. Train base classifiers (judgment features only)
3. Train augmented classifiers (with linguistic + LLM features)
4. Perform group-level detection across different group sizes
5. Conduct rating scale and dimension analyses
6. Generate visualization plots
7. Save all metrics to `metrics.csv`

### Output Files

After execution, the following files are generated:

**Metrics:**
- `metrics.csv` - Complete metrics for all experiments

**Visualizations:**
- `accuracy_logistic_regression.png`
- `accuracy_random_forest.png`
- `f1_logistic_regression.png`
- `f1_random_forest.png`
- `accuracy_logistic_regression_aug.png`
- `accuracy_random_forest_aug.png`
- `f1_logistic_regression_aug.png`
- `f1_random_forest_aug.png`

**Debug CSVs:**
- `helpsteer2.csv`, `helpsteer3.csv`, `antique.csv`, `neurips.csv`
- `helpsteer2Combine.csv`, `helpsteer3Combine.csv`, `antique Combine.csv`, `neuripsCombine.csv`

## 📈 Results

Results are automatically saved to `metrics.csv` with the following structure:

| Task | Model | Dataset | Accuracy | F1 |
|------|-------|---------|----------|-----|
| Judgement Features | Logistic Regression | helpsteer2 | 0.XX | 0.XX |
| Augmented features | Random Forest | neurips | 0.XX | 0.XX |
| GroupSize_8_Aug:True | Logistic Regression | antique | 0.XX | 0.XX |

### Key Findings

The implementation evaluates:
- **Base vs Augmented Performance**: How linguistic and LLM-enhanced features improve detection
- **Group Size Effect**: Detection accuracy improvement with larger judgment groups
- **Rating Scale Impact**: How rating granularity affects detectability
- **Dimension Importance**: Which judgment dimensions are most informative

## 🔧 Customization

### Changing Classifiers
Edit `mlmodel.py` to add new models:
```python
def get_mlmodel(type):
    if type == "YOUR_MODEL_NAME":
        model = YourModelClass()
    return model
```

### Modifying Group Sizes
Edit `main.py`:
```python
k = [1, 2, 4, 8, 16]  # Change to your desired group sizes
```

### Adjusting Features
Modify feature selection in `data.py`:
```python
# Remove specific features
data = data.loc[:, ~data.columns.str.contains("feature_name")]
```

## 📊 Interpretation Guide

### Metrics
- **Accuracy**: Overall correct classification rate (Human=0, LLM=1)
- **F1 Score**: Harmonic mean of precision and recall, useful for imbalanced datasets

### Group Detection
- Logit aggregation: `sum(logit(P(LLM|x_i)))` for group examples
- Classification: Group labeled as LLM if aggregated logit > 0

## ⚠️ Important Notes

1. **Noun-Verb Ratio**: Must be removed from feature files before running
2. **Group Size**: Use `group_size=1` for Tasks 1-2, vary for Task 3
3. **Data Matching**: Feature CSVs must match judgment data by key columns
4. **Memory**: Large group sizes may require significant RAM

## 🐛 Troubleshooting

**Issue**: "No such file or directory"
- Solution: Ensure data is downloaded and organized correctly

**Issue**: "KeyError" during feature merging
- Solution: Verify key columns (prompt, response, content, etc.) exist in both datasets

**Issue**: "NaN values in predictions"
- Solution: Check for missing data; the pipeline removes NaN rows automatically

## 📝 References

1. From Generation to Judgment: Opportunities and Challenges of LLM-as-a-judge
2. Who's Your Judge? On the Detectability of LLM-Generated Judgments

## 👥 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request
---

**Last Updated**: November 2024