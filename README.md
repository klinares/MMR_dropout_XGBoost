# 💉 MMR_dropout_XGBoost

> *"First dose: ✅. Second dose: 📭. Machine learning: on it."*

![](images/clipboard-4050432742.png)

A supervised machine learning classification study predicting measles
vaccination dropout among Afghan children using the 2023 UNICEF Multiple
Indicator Cluster Survey (MICS) and World Bank provincial indicators.
Five algorithms — logistic regression, bagging, random forest, neural
network, and XGBoost — are compared under a rigorous tidymodels pipeline
with 10-fold cross-validation.

**Author:** Kevin Linares · University of Maryland, JPSM

Paper -> [MMR Dropout: A Predictive Approach](https://klinares.github.io/MMR_dropout_XGBoost/paper/mmr_dropout_ml_paper.html)

------------------------------------------------------------------------

Background

Afghanistan reported nearly 13,000 measles cases among children under
five in 2024, with national two-dose vaccination coverage stalled at
approximately 83%. A critical driver of this gap is **vaccination
dropout** — children who receive a first MMR dose but never return for
the essential second. This project builds a predictive model to identify
children at high risk of dropout, providing actionable targeting
information for public health interventions under resource-constrained
conditions.

------------------------------------------------------------------------

Data

| Source | Description |
|------------------------------------|------------------------------------|
| [UNICEF MICS 2023 — Afghanistan](https://mics.unicef.org/surveys) | Child and household surveys; multi-stage cluster sample design |
| [World Bank Province Indicators](https://www.worldbank.org/en/data/interactive/2019/08/01/afghanistan-interactive-province-level-visualization) | Conflict index, female literacy rate, health infrastructure score, population density, poverty rate |
| Afghanistan provincial shapefile | Used for choropleth mapping of dropout rates |

**Analytic sample:** 971 children aged 24+ months who received at least
one measles vaccine dose (8% dropped for missing data from an initial n
= 1,067).

**Outcome variable:** - `dropped` (1): Received dose 1 only — 39% of
sample - `completed` (0): Received both doses — 61% of sample

**Features (22 total):** Household characteristics (electricity, water
source, floor material, toilet type, rooms, handwashing), mother's
education level, family wealth score, national immunization campaign
participation, and five World Bank province-level indicators.

------------------------------------------------------------------------
Methods

All modeling uses the `tidymodels` framework in R with parallel
processing via `future`.

| Algorithm | Engine | Key Tuning Parameters |
|------------------------|------------------------|------------------------|
| Logistic Regression | `glmnet` | penalty |
| Bagging | `rpart` (×100) | cost_complexity, tree_depth, min_n |
| Random Forest | `ranger` | mtry, trees, min_n |
| Neural Network | `nnet` | penalty, epochs, hidden_units |
| XGBoost | `xgboost` | mtry, trees, tree_depth, learn_rate, loss_reduction, sample_size, min_n |

**Tuning strategy:** - 80/20 train/test split (stratified) - 10-fold
cross-validation on training set - `grid_space_filling()` for neural
network (300 combinations) - `grid_latin_hypercube()` for XGBoost (500
combinations) - Model selection criterion: F1-score (prioritizes
minimizing false positives given resource constraints)

**Evaluation metrics:** AUC-ROC and F1-score on held-out test set.

------------------------------------------------------------------------

Results

The neural network achieved the best overall performance on the test
set:

| Model               | F1-Score  | AUC-ROC |
|---------------------|-----------|---------|
| Logistic Regression | —         | —       |
| Bagging             | —         | —       |
| Random Forest       | —         | 0.633   |
| **Neural Network**  | **0.722** | —       |
| XGBoost             | —         | —       |

All models outperformed random chance. The neural network was selected
as the final model based on its leading F1-score.

**Top predictors (neural network VIP):** Provincial population density,
number of rooms in house, access to septic tank, and housing tenure were
among the most important features driving predictions.

------------------------------------------------------------------------

Repository Structure

```         
MMR_dropout_XGBoost/
│
├── inputs/
│   ├── mics/
│   │   └── Afghanistan/spss_dataset/
│   │       ├── ch.SAV               # MICS child survey
│   │       └── hh.SAV               # MICS household survey
│   └── world_bank/
│       └── Afghanistan_province_indicators.csv
│
├── shape_files/
│   ├── afghan_province/             # Province shapefile
│   └── afghanistan_provincial_map.RDS
│
├── outputs/
│   ├── lr_res.RDS                   # Saved tuning results
│   ├── bag_res.RDS
│   ├── rf_res.RDS
│   ├── nnet_res.RDS
│   └── xgb_res.RDS
│
├── workflow.png                     # ML workflow diagram (Figure 1)
├── mmr_dropout_ml_paper.qmd         # Main Quarto source
│
└── README.md
```

------------------------------------------------------------------------
Reproducing the Analysis

Prerequisites

R 4.5 with the following packages:

``` r
pacman::p_load(
  knitr, sjlabelled, haven,
  vip, kableExtra,
  sf, baguette, bonsai, brulee,
  stacks, future, tictoc,
  patchwork, viridis, ggthemes,
  janitor, tidymodels, tidyverse
)
```

Notes on reproducibility

Model tuning is computationally expensive. Pre-tuned model objects are
saved as `.RDS` files in `outputs/` and loaded directly during rendering
— the tuning code blocks are commented out. To re-tune from scratch,
uncomment the relevant blocks and expect significant runtime (parallel
processing via `plan(multisession)` is enabled).

Update all `file_path` variables at the top of the `.qmd` to point to
your local directory structure.

Render

``` bash
quarto render mmr_dropout_ml_paper.qmd --to jasa-pdf
```

------------------------------------------------------------------------

Limitations

-   Small analytic sample (\~971 children) limits model generalizability
-   Feature engineering required collapsing sparse factor levels, which
    may obscure signal
-   Province-level World Bank indicators are ecological proxies, not
    individual-level measurements
-   Model performance is moderate; results should be interpreted as
    targeting guidance, not clinical prediction

------------------------------------------------------------------------
