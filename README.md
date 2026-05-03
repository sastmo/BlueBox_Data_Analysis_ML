# Decoding Canada's Blue Box Program with Machine Learning

[![CI](https://github.com/sastmo/BlueBox_Data_Analysis_ML/actions/workflows/ci.yml/badge.svg)](https://github.com/sastmo/BlueBox_Data_Analysis_ML/actions/workflows/ci.yml)
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Clustering-blue)](https://hdbscan.readthedocs.io/en/latest/)
[![Feature Engineering](https://img.shields.io/badge/Feature-Engineering-blueviolet.svg)](https://en.wikipedia.org/wiki/Feature_engineering)
[![Feature Selection](https://img.shields.io/badge/Feature%20Selection-Global%20%26%20Local-teal.svg)](https://en.wikipedia.org/wiki/Feature_selection)
[![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter-Tuning-orange.svg)](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
[![Gradient Boost](https://img.shields.io/badge/Gradient%20Boost-Regression-yellowgreen)](https://en.wikipedia.org/wiki/Gradient_boosting)
[![Explainability](https://img.shields.io/badge/Explainability-LIME%20%26%20TreeInterpreter-red.svg)](https://en.wikipedia.org/wiki/Explainable_artificial_intelligence)
[![Power BI](https://img.shields.io/badge/Power%20BI-Visualization-orange)](https://powerbi.microsoft.com/)

Canada's Blue Box Program collects recyclable material from 5.4 million households across 70+ locations in 9 provinces -- roughly 740,000 tonnes per year. The raw numbers are well documented. What isn't is the question underneath them: why do some municipalities collect three times more material per household than others, using similar infrastructure and comparable budgets?

Answering that requires more than descriptive statistics. This project builds a full ML pipeline on three years of program data (2019-2021) -- one designed to move from observation to explanation. Clustering first establishes behavioral archetypes without assuming their shape. Feature selection then runs in two layers, filtering noise before confirming only genuinely predictive variables survive. Local interpretability closes the loop, explaining not just what drives performance across programs on average, but why a specific municipality is underperforming and what levers it has.

The pipeline is modular by design. The stages -- cluster, tune, select globally, select locally, model, explain -- transfer to any dataset where the goal is to diagnose performance differences across a population of programs, regions, or entities. Swap the data, and the structure holds.

---

## Pipeline

### 1. Data Model and Data Manipulation

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/028f30ab-bca8-4b47-a6b9-7263c289c346)

Three Excel datasets (market volumes, costs/revenue, finance program) covering 2019-2021 were cleaned and merged using Power Query into a Star Schema. Geographic and demographic context was pulled from CSV files, with Python fuzzy matching resolving inconsistent municipality names across sources.

**Script:** `src/data_model.py` | **Power BI file:** `PowerBI_Visualization_Blue_Box/` | [Details](https://tasteful-background-b30.notion.site/1-Data-Model-ce06ad4af58346cb98e853fe997345cb?pvs=4)

---

### 2. Data Visualization with Power BI

Explore Material Collections: 2019 vs. 2021 - Changes by Category

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/6f4d9cab-5b91-4927-a270-7334cb0331e8)

Power BI dashboards answer the core question: where is material collection growing, where is it shrinking, and what policies correlate with better outcomes? DAX measures drive the visuals across 70+ programs in 9 provinces.

**Power BI file:** `PowerBI_Visualization_Blue_Box/` | [Details](https://tasteful-background-b30.notion.site/2-Data-Visualization-using-Power-BI-9c9c024c4ddc4fceb1f7ce2d1ababad5?pvs=4)

---

### 3. Exploratory Data Analysis

Quadrant-Based Classification: Waste Material Change vs. Household Service Coverage Change (2020-2021)

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/a323cabd-acae-4d5d-8be3-e2049560b020)

Three findings stood out. Resource allocation is roughly uniform across regions regardless of program size -- an equity-looking pattern that actually breeds inefficiency in high-volume areas. The cost-to-volume relationship follows a power function, suggesting diminishing returns beyond a threshold. And the highest-volume regions -- Toronto, Ottawa, Vancouver, Montreal -- show the lowest efficiency per household.

**Script:** `src/exploratory_data_analysis.py` | **Plots:** `outputs/eda_plots/` | [Details](https://tasteful-background-b30.notion.site/3-Exploratory-Data-Analysis-EDA-a802f6e292ad426e8a17a65fdd6e4bbb?pvs=4)

---

### 4. Data Clustering (HDBSCAN + t-SNE)

Clustered Programs in 2021: t-SNE with HDBSCAN and Program Efficiency

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/e424c2ba-5dc6-4851-88e8-3b7972f768d5)

HDBSCAN groups programs by behavioral similarity rather than geography, without requiring a preset number of clusters. t-SNE reduces the feature space to 2D for visualization. The result is a set of distinct program archetypes that form the basis for all subsequent feature analysis.

**Script:** `src/clustering_hdbscan_tsne.py` | **Plots:** `outputs/hdbscan_plots/` | [Details](https://tasteful-background-b30.notion.site/4-Data-Clustering-1f3fc49ed986428c806098c6555dda78?pvs=4)

---

### 5. Hyperparameter Tuning

Stability Assessment: Out-of-Bag Error Rate for Random Forest

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/f887f8b1-eb09-4fda-8817-b3bd52bb826c)

Random Forest drives the feature analysis in the next stage. Before relying on its importances, multicollinearity was addressed with VIF and hyperparameters were tuned with Randomized Search followed by Grid Search. The OOB error curve confirms the final configuration is stable.

**Script:** `src/hyperparameter_tuning.py` | [Details](https://tasteful-background-b30.notion.site/5-Hyperparameter-Tuning-89e48769a75d45f69d560d64a2787596?pvs=4)

---

### 6. Feature Analysis

Feature Selection: Prioritizing Important Features by Frequency

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/8e568535-6297-4aa4-9c27-5a29b7334368)

Two-layer selection. Layer one introduces a random shadow feature for each real feature -- any feature that can't beat its random twin is dropped. Layer two applies Boruta, using shuffled shadow twins with statistical testing across multiple iterations. Only features that survive both rounds make it to the model.

**Scripts:** `src/feature_selection_random.py`, `src/feature_selection_boruta.py` | [Details](https://tasteful-background-b30.notion.site/6-Feature-Analysis-cb998fe7a17c4446b72d206fad7d03ec?pvs=4)

---

### 7. Local Feature Selection

Local Feature Selection for Noise Cluster (-1)

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/36031e43-dc77-431c-a7da-ad3cba48e279)

Global feature importance shows what matters on average. Local selection goes further -- at the group level, binary classification identifies which features separate one cluster from the rest. At the observation level, TreeInterpreter and LIME explain individual predictions, making it possible to diagnose why a specific municipality is underperforming and what levers are available.

**Scripts:** `src/local_feature_selection_group.py`, `src/exploratory_data_analysis_clusters_v1.py`, `src/local_feature_selection_observation.py` | [Details](https://www.notion.so/7-Local-Feature-Selection-599e6ea05a9e43eab629f7e9192d003a)

---

### 8. Gradient Boosting Regressor

Partial Dependency Analysis for 2021

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/471660b2-0b46-4502-8d95-55d6d6661de3)

The final model is a Gradient Boosting Regressor trained on the features that cleared both selection stages. It forecasts next-year collection volumes per municipality. Partial dependence plots show how each feature influences predictions across its range, and early stopping controls overfitting.

**Scripts:** `src/gradient_boosting_class.py`, `src/gradient_boost_regression.py` | [Details](https://www.notion.so/8-GradientBoostingRegressor-ca74e6e8c17d4ea3b2972d4c57649c84?pvs=4)

---

[Full project documentation](https://tasteful-background-b30.notion.site/Exploratory-Data-Analysis-EDA-of-the-Canada-Blue-Box-Program-d5ff354b53664c3b8d0d4091f65a336c?pvs=4)

---

<details>
<summary><strong>Installation & Setup</strong></summary>

**Prerequisites:** Python 3.9+, Git

**macOS / Linux**
```bash
git clone https://github.com/sastmo/BlueBox_Data_Analysis_ML.git
cd BlueBox_Data_Analysis_ML

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

**Windows (Command Prompt or PowerShell)**
```bash
git clone https://github.com/sastmo/BlueBox_Data_Analysis_ML.git
cd BlueBox_Data_Analysis_ML

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

> Some packages (`hdbscan`, `treeinterpreter`) require a C compiler. On Windows, install [Microsoft C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) first. On macOS, run `xcode-select --install` if prompted.

**Running the pipeline**

Scripts run from the `src/` directory. Each step produces outputs that the next step depends on.

```bash
cd src
python data_model.py                          # produces data/market_agg.csv, data/quadrant_data.csv
python exploratory_data_analysis.py           # EDA plots, produces data/quadrant_data.csv
python clustering_hdbscan_tsne.py             # clustering, produces data/result_classification.csv
python feature_selection_random.py            # layer 1 feature selection
python feature_selection_boruta.py            # layer 2 feature selection
python hyperparameter_tuning.py               # tune Random Forest
python local_feature_selection_group.py       # group-level local importance
python local_feature_selection_observation.py # observation-level importance
python gradient_boost_regression.py           # final model
```

Data files are in `data/`, plots and outputs land in `outputs/`.

</details>
