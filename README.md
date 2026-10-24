# Lung Cancer Risk Screening: Data Modeling with Logistic Regression

> A data-driven early screening and diagnostic system that analyzes risk factors (smoking habits, air pollution, clinical symptoms) to classify patients, so doctors can intervene earlier.

**Course:** ADY201m, Data Modeling | **Class:** AI2002 | **Institution:** FPT University | **Team:** Group 4 (5 members)

---

## Overview

Lung cancer is far easier to treat when it is caught early. This project builds an interpretable screening model on a patient dataset of **5,000 records and 29 features** (demographics, smoking behavior, environmental exposure, medical history, symptoms, and lab/clinical measurements).

Instead of relying on a single global model, we asked: **does splitting patients into clinically meaningful subgroups and training a dedicated model per group improve accuracy?** The result is a rule-based routing system that sends each patient to the model best suited to them.

## Key Results

| Approach | Avg. Accuracy |
|---|---|
| Task 1: Global Logistic Regression (29 features) | **~89.3%** |
| Task 4: Reduced model (8 features only) | **88.13%** |
| Task 2: Subgroup models (best groups) | up to **91.0%** (Low CRP) |

- A reduced **8-feature model** loses only about 1 percentage point of accuracy compared with the full 29-feature model, which makes it far simpler to use in practice.
- Subgroup models **outperformed the global model for some groups** (e.g. Low CRP, Non-Smoker, Symptoms + abnormal X-ray) and underperformed for others (e.g. Mid CRP, Light Smoker).
- Accuracy is averaged over 3 runs, each with a different random shuffle (`random_state = 24, 48, 72`) and a 90/10 train/test split (4,500 train / 500 test).

## Approach

### Task 1: Baseline model
- Shuffled the data with 3 different seeds and split 90/10.
- Trained a `LogisticRegression` model on all 29 features and evaluated with `accuracy_score`.
- Exported the learned coefficients as an explicit equation `f(x)` for each run, so the model stays fully interpretable.

### Task 2: Subgroup (cluster) modeling
We split patients into 3 groups along three clinical dimensions, trained a separate model for each group, and compared it against the baseline.

| Field | Grouping | Group avg. accuracy |
|---|---|---|
| **1. Smoking Behavioral Cluster** | Non-smoker / Light (pack-years < 13) / Heavy (>= 13) | 89.3% / 84.7% / 87.7% |
| **2. CRP Level** | Low (<= 2) / Mid (3-9) / High (>= 10) | 91.0% / 84.2% / 89.3% |
| **3. Clinical Symptom Cluster** | No symptoms & normal X-ray / Symptoms, normal X-ray / Symptoms & abnormal X-ray | 87.1% / 82.4% / 89.5% |

Features used to define each grouping were dropped from that group's model to avoid leakage. Each field also includes a **data routing** step: a new patient is assigned to a group, the system compares that group model's average accuracy against the global model, and applies whichever is better.

### Task 3: Comparison and final model
Combining the winning subgroup models gives a piecewise final classifier:

```
f_final(X) =
    f_LowCRP(X)      if crp_level <= 2
    f_HighCRP(X)     else if crp_level >= 10
    f_SympXray(X)    else if xray_abnormal == 1 AND (chronic_cough == 1 OR shortness_of_breath == 1)
    f_NonSmoker(X)   else if smoker == 0
    f_Global(X)      otherwise
```

Findings: the Non-Smoker, Low-CRP, High-CRP, and Symptoms + abnormal X-ray groups usually beat the global model, while the other groups are usually equal or 3-7% lower.

### Task 4: Feature reduction
- Computed the correlation of every feature with `lung_cancer_risk`.
- Dropped features with |correlation| < 0.45, plus redundant smoking features (`smoker`, `smoking_years`, `cigarettes_per_day`) that are highly collinear with `pack_years` (r up to 0.94).
- Retained 8 features: `pack_years`, `copd`, `chronic_cough`, `shortness_of_breath`, `oxygen_saturation`, `fev1_x10`, `crp_level`, `xray_abnormal`.
- Result: **88.13%** average accuracy, close to the full model while using under a third of the features.

## Strongest Risk Factors (correlation with outcome)

`pack_years` (0.63), `crp_level` (0.60), `xray_abnormal` (0.58), `cigarettes_per_day` (0.57), `smoking_years` (0.55), plus `fev1_x10` (-0.60) and `oxygen_saturation` (-0.59) as the strongest protective/inverse signals.

## Tech Stack

- **Language:** Python
- **Libraries:** pandas, NumPy, scikit-learn (`LogisticRegression`, `train_test_split`, `accuracy_score`), Matplotlib / Seaborn

## Web Application

The team also built a web application that wraps the trained model so a user can enter patient information and get a lung cancer risk prediction. Its source code is in the `uiwebsource/` folder.

## Repository Structure

> Adjust to match your actual files.

```
.
├── data/                  # dataset
├── task1_baseline.ipynb   # global logistic regression
├── task2_subgroups.ipynb  # smoking / CRP / clinical-symptom models + routing
├── task3_comparison.ipynb # comparison + final piecewise model
├── task4_reduction.ipynb  # correlation-based feature reduction
├── uiwebsource/           # web app that serves the prediction model
├── slides/05_DataModeling.pdf
└── README.md
```

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
jupyter notebook
```

Open the notebooks in order (Task 1 to Task 4).



## Team (Group 4)

| Name | Student ID |
|---|---|
| Nguyen Phuoc Thinh (Leader) | CE200373 |
| Nguyen Viet Linh | CE200222 |
| Nguyen Phuoc Xiem | CE190249 |
| Ho Vinh Hung | CE202073 |
| Lam Thien Bao | CE201661 |

Lecturer: HieuNT

## Disclaimer

This is an academic project for educational purposes. It is **not** a validated medical device and must not be used for real clinical decisions.
