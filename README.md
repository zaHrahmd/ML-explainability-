In this repository, you can find the data, code, and figures related to the paper: AUTOLYCUS: Exploiting Explainable Artificial Intelligence (XAI) for Model Extraction Attacks against Interpretable Models (to be published in PETs 2024).

1. You can experiment on the proposed attack using ipynb files.
2. Datasets can be included in the code directly or manually added to the 'data' folder. Make sure they follow a similar format to other datasets for explainer compatibility.
3. Some experimental result plots from the development stage can be found in 'LIME/plots' and 'SHAP/plots'.
4. 'requirements.txt' provides the libraries and their version numbers as used in our work.
5. 'packages_full.txt' gives the list of all packages in the environment in which we conducted our experiments.

We welcome suggested improvements for streamlining the code and enhancing the attack. During the standardization, we realized there are many bugs, runtime issues and we hope to resolve them in time. For further questions and improvements about the code, email abdullahcaglar.oksuz@case.edu.


# Defending Against Explanation-Based Model Extraction Attacks  
### An Empirical Study of the AUTOLYCUS Framework

---

## Overview

Machine learning models exposed through APIs are vulnerable to **model extraction attacks**, where an adversary reconstructs a target model by querying it.

This project investigates the **AUTOLYCUS framework** and proposes a **defense strategy** that combines:

- Query-pattern detection  
- Output distortion (label + explanation manipulation)

The goal is to **detect malicious behavior early** and **degrade the attacker's ability to reconstruct the model**, while maintaining usability for legitimate users.

---

## Methodology

### 🔹 Attack Model
We simulate **explanation-based model extraction attacks** using:

- SHAP explanations  
- Targeted query generation (feature-space traversal)

---

### 🔹 Defense Strategy

The proposed defense consists of two stages:

#### 1. Query Pattern Detection
- Detects suspicious query clustering behavior  
- Uses feature-space distance and similarity thresholds  

#### 2. Output Distortion (Poisoning)
- Activated after detection  
- Applies:
  - Label manipulation (second-best or random class)  
  - SHAP explanation distortion  
- Contaminates attacker training data  

---

## Experiments

We evaluate the attack and defense across multiple datasets:

- Iris  
- Breast Cancer  
- Adult Income  
- Crop  
- Nursery  
- Mushroom  

### Evaluation Metric

- **Extraction similarity (rtest_sim)**  
  Measures how closely the attacker’s surrogate model matches the target model  

- **Detection rate**  
  Percentage of runs where malicious behavior is detected  

- **Detection delay**  
  Number of queries before detection is triggered  

---

## Project Structure
.
├── step2_experiment.py     # Main experiment script (FINAL version used in report)
├── utils.py                # Core pipeline: attack + defense implementation
├── data/                   # Datasets
├── plots/                  # Generated figures
├── requirements.txt        # Dependencies
└── README.md

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt

python step2_experiment.py
