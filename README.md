# EasyVisa: Visa Case Certification Prediction

Classification project that predicts whether a US employer's foreign-worker visa application will be **certified or denied**, and which applicant and employer attributes drive that outcome. It is meant to help the Office of Foreign Labor Certification (OFLC) prioritise the review of a growing number of cases.

## Business Problem

OFLC processed more than 775,000 employer applications in FY 2016, and reviewing every case by hand is slow. EasyVisa was hired to:

1. Build a model that helps shortlist cases with a higher chance of approval.
2. Describe the applicant profiles that are more likely to be certified or denied.

## Dataset

`EasyVisa.csv`: 25,480 visa cases, 12 columns, no missing values.

| Column | Description |
|---|---|
| `case_id` | Case ID (dropped, identifier only) |
| `continent` | Continent of the employee |
| `education_of_employee` | High School, Bachelor's, Master's or Doctorate |
| `has_job_experience` | Y/N |
| `requires_job_training` | Y/N |
| `no_of_employees` | Employees at the employer's company |
| `yr_of_estab` | Year the employer was established |
| `region_of_employment` | US region of the intended job |
| `prevailing_wage` | Average wage for similar workers in that occupation and area |
| `unit_of_wage` | Unit of the prevailing wage (Hour, Week, Month, Year) |
| `full_time_position` | Y/N |
| `case_status` | **Target.** Certified or Denied |

**Class balance:** 66.8% certified, 33.2% denied. A "model" that certifies everything is 66.8% accurate, so the notebook compares every model with that reference.

**Data issues handled in the notebook**

- 33 rows (0.13%) have a negative `no_of_employees` (-11 to -26). They look like sign errors and are dropped, leaving 25,447 rows.
- `prevailing_wage` is quoted in different units, and the labels look unreliable: "weekly" and "monthly" wages are on a yearly scale (about $80k), and only the hourly rows look like hourly rates. The wage is therefore **not converted**; `unit_of_wage` is kept as a feature instead.

## Approach

1. **EDA:** distributions, and certification rate by education, continent, experience, region, wage and wage unit.
2. **Preprocessing:** 0/1 encoding for yes/no columns, ordinal encoding for education, one-hot encoding for continent, region and wage unit, then a stratified 60/20/20 train/validation/test split.
3. **Baselines:** seven model types (Bagging, Random Forest, GBM, AdaBoost, Decision Tree, XGBoost, Logistic Regression) on the original, SMOTE-oversampled and undersampled training data, scored on the validation set.
4. **Tuning:** randomised search for AdaBoost, Gradient Boosting and XGBoost, scored on cross-validated ROC-AUC. The resampling method is part of the search and lives inside an `imblearn` pipeline, so resampling only touches the training part of each fold.
5. **Selection:** the tuned model with the best validation ROC-AUC is scored once on the test set.
6. **Interpretation:** permutation importance plus the EDA.

ROC-AUC is the primary metric because accuracy, recall and F1 for the majority class are easy to inflate (certifying everyone gives recall 1.0 and F1 0.80).

## Results

Final model: **XGBoost** (tuned, no resampling selected).

| Test set (5,090 cases) | Accuracy | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Final model | 0.749 | 0.874 | 0.778 | 0.823 | 0.779 |
| Certify everyone | 0.668 | 1.000 | 0.668 | 0.801 | 0.500 |

- The three tuned boosting models are within about 0.003 AUC of each other on validation, so the choice among them is not decisive.
- Resampling did not improve ROC-AUC for the best models.
- The model finds most certified cases, but it also certifies about half of the cases that were actually denied (847 of 1,689 in the test set). It is useful for **ranking cases for review**, not for automatic decisions.

## Key Findings

| Factor | Certification rate |
|---|---|
| Doctorate / Master's / Bachelor's / High School | 87% / 79% / 62% / 34% |
| Job experience: yes / no | 75% / 56% |
| Wage unit: yearly / hourly | 70% / 35% |
| Region: Midwest / South / Northeast / West | 76% / 70% / 63% / 62% |
| Continent: Europe / Asia / South America | 79% / 65% / 58% |

- **Education** is the most important feature by a wide margin, followed by **job experience**.
- Experience helps most for Master's holders (64% to 89%). It barely helps High School applicants (32% to 36%).
- Within the same wage unit, the **level of the prevailing wage has almost no relationship** with certification, and neither do full-time status, job training, company size or company age.

## Repository Contents

```
.
├── EasyVisa_Full_Code_Notebook.ipynb   # Full analysis and modelling (run end to end)
├── EasyVisa.csv                        # Dataset
├── requirements.txt
└── README.md
```

## Running the Notebook

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook EasyVisa_Full_Code_Notebook.ipynb
```

Run it from the folder that contains `EasyVisa.csv`. The tuning cells take several minutes on a laptop.

## Limitations

- The data has no occupation, job title or offered wage, so a large part of the outcome is unexplained (ROC-AUC is about 0.78).
- The patterns are associations, not causes.
- `continent` and `region` can act as proxies for nationality. Check any operational model for disparate impact before using it.
- Results come from one train/validation/test split; differences of about 0.01 AUC between models are within noise.

## Tech Stack

Python, pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn.
