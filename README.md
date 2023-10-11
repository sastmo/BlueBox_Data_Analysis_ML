# Diving into Canada's Blue Box Program with EDA and ML 👋
[![EDA](https://img.shields.io/badge/EDA-Data%20Analysis-blue.svg)](https://en.wikipedia.org/wiki/Exploratory_data_analysis)
[![Python](https://img.shields.io/badge/Python-3.7-blue.svg)](https://www.python.org/)
[![Power BI](https://img.shields.io/badge/Power%20BI-Data%20Visualization-orange)](https://powerbi.microsoft.com/)
[![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Enabled-blue.svg)](https://en.wikipedia.org/wiki/Machine_learning)
[![Data Clustering](https://img.shields.io/badge/Data%20Clustering-Clustering-lightgrey)](https://en.wikipedia.org/wiki/Cluster_analysis)
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Clustering-blue)](https://hdbscan.readthedocs.io/en/latest/)
[![Hyperparameter Tuning](https://img.shields.io/badge/Hyperparameter%20Tuning-Method-blue.svg)](https://en.wikipedia.org/wiki/Hyperparameter_tuning)
[![Feature Analysis](https://img.shields.io/badge/Feature%20Analysis-Method-blue.svg)](https://en.wikipedia.org/wiki/Feature_analysis_in_machine_learning)
[![Random Forest](https://img.shields.io/badge/Random%20Forest-Modeling-brightgreen)](https://en.wikipedia.org/wiki/Random_forest)
[![Gradient Boost](https://img.shields.io/badge/Gradient%20Boost-Modeling-yellowgreen)](https://en.wikipedia.org/wiki/Gradient_boosting)
[![Regression](https://img.shields.io/badge/Regression-Modeling-green)](https://en.wikipedia.org/wiki/Regression_analysis)

# Welcome to the Blue Box Program Data Analysis 🌱♻️
Join me as we embark on an eco-conscious journey into the heart of data analysis, where sustainability meets insight. 

In Canada, the renowned Blue Box Program takes center stage in the realm of waste management and recycling. It's a nationwide effort, orchestrated by diverse municipalities and organizations, with a shared vision of promoting recycling and responsible material disposal in our homes and communities. The program equips residents with blue boxes or bins designed to cradle recyclables like paper, cardboard, glass, plastic, and metal. These materials are tenderly plucked from the regular trash stream and sent on a path of renewal through recycling.

Operated by a nationwide waste management organization, this program spans over 70 locations, reaching 9 provinces. With a whopping 5.4 million active participants, its commitment to boosting recycling knows no bounds. Year after year, it welcomes a 5% rise in fresh contributors, contributing to the successful recycling of an impressive 740,000 tons of waste materials on average.

At the heart of this project, you'll find a deep-seated passion for the environment and an unwavering dedication to effective waste management.

## Project Goals 🎯

My mission? To craft an all-encompassing framework for data analysis, guiding us through data cleaning, preparation, feature analysis, and regression. It's a roadmap to unearth the hidden gems buried within the data, one step at a time.

Together, let's dive in and unlock the potential of this data-driven exploration! 🚀📊

## Table of Contents

1. 🔧 **Data Model and Data Manipulation**
2. 📊 **Data Visualization with Power BI**
3. 🔍 **Exploratory Data Analysis (EDA)**
4. 🧩 **Data Clustering**
5. 👨‍🔧 **Hyperparameter Tuning**
6. 📊 **Feature Analysis**
7. 💫 **Local Feature Selection**
8. 🌱🌱 **GradientBoostingRegressor**


## 🔧 **1. Data Model and Data Manipulation**
![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/028f30ab-bca8-4b47-a6b9-7263c289c346)

In this section, we delve into the world of data modeling and manipulation to lay the foundation for our comprehensive data analysis framework.

Our mission is clear: create a comprehensive data analysis framework from start to finish, covering data cleaning, preparation, feature analysis, and regression modeling.

📈 It all begins here with data manipulation. Using Power Query, we carefully cleaned and merged datasets from 2019, 2020, and 2021, ensuring our data was ready for analysis. A Star Schema data model was crafted by defining primary and foreign keys within the datasets.

🗺️ To enrich our analysis, we incorporated geolocation and demographic data from CSV files, adding a valuable layer of context. Python played a role in data refinement and matching, maintaining accuracy through the use of fuzzy logic.

🚀 Join me as we move forward to uncover thrilling Power BI data visualizations and unveil the mesmerizing tapestry of data patterns. Your journey into the world of data-driven insights is about to get even more exciting!

**Related Scripts:** The scripts used for this section can be found in the file `Data_Model.py` in this repository.

**Data Model in Power BI:** The data model is available as a Power BI (pbix) file in the `PowerBI_Visualization_Blue_Box` folder within this repository.

**Extended Details:** For more in-depth information about this section, please visit my portfolio pages: [Data Model and Data Manipulation](https://tasteful-background-b30.notion.site/1-Data-Model-ce06ad4af58346cb98e853fe997345cb?pvs=4)


## 📊 **2. Data Visualization with Power BI**

Explore Material Collections: 2019 vs. 2021 - Changes by Category

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/6f4d9cab-5b91-4927-a270-7334cb0331e8)

Welcome to the captivating world of **Data Visualization with Power BI**!

In this section, we leverage the power of Power BI to transform our meticulously prepared data into stunning visuals that vividly convey our discoveries. We pose essential questions at the heart of our exploration and use Data Analysis Expressions (DAX) to craft responses within Power BI visuals.

🌍 These visuals provide a high-level overview of waste management and recyclable materials data, painting a vivid picture of the program's scope across the nation. Throughout this process, we've uncovered valuable insights that have the potential to shape decisions and influence the future of this essential work.

💼 As you reach this point, I'm thrilled to see you progressing on this enlightening journey. Together, we've gained a broader understanding of the program's reach and the materials it handles. Notably, we've identified policies that are making a positive impact, and we've unveiled strong relationships between key parameters. 📈🤝

**Power BI Dashboard:** The data model is available as a Power BI (pbix) file in the `PowerBI_Visualization_Blue_Box` folder within this repository.

**Extended Details:** For more in-depth information about this section, please visit my portfolio pages: [Data Visualization using Power BI](https://tasteful-background-b30.notion.site/2-Data-Visualization-using-Power-BI-9c9c024c4ddc4fceb1f7ce2d1ababad5?pvs=4)


## 🔍**3. Exploratory Data Analysis (EDA)**

Analyzing Programs: Quadrant-Based Classification of Waste Material Change vs. Household Service Coverage Change (2020-2021)

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/a323cabd-acae-4d5d-8be3-e2049560b020)

In this captivating expedition, we've delved deep into the intricacies of our dataset, seeking to unearth the hidden gems of knowledge within. As we navigated through our data-driven adventure, we uncovered three pivotal insights:

1. 🌎 **Uniform Distribution Across Regions:** Our data revealed a striking uniformity in resource allocation across regions, encompassing vibrant cities like Toronto, Ottawa, Vancouver, and Montreal. While this might seem equitable, it has inadvertently bred inefficiencies within these areas.
2. 💰 **High Operations, Low Efficiency:** Curiously, these very regions boasting the highest program operations and waste collection rates demonstrated lower efficiency. This imbalance creates a significant financial burden that could potentially jeopardize the program's sustainability.
3. 📈 **Cost vs. Volume Relationship:** Our exploration ventured into the intriguing relationship between program costs and collected materials. It exhibited a fascinating power function pattern, characterized by a power value between 0 and 1. This suggests a potential volume shrinkage phenomenon beyond a certain point. To ensure program success, we must pinpoint the optimal values for each region.

With these valuable insights in our arsenal, we find ourselves at the thrilling culmination of our EDA journey. 🌟🔍 For more exciting insights, I invite you to explore this section in detail.

**Related Scripts:** `Exploratory_Data_Analysis.py`

**Additional Plots:** More plots related to this section can be found in the `Exploratory_Data_Analysis_plots` folder in my repository.

**Extended Details:** For detailed information about this section, please visit this page: [Exploratory Data Analysis (EDA)](https://tasteful-background-b30.notion.site/3-Exploratory-Data-Analysis-EDA-a802f6e292ad426e8a17a65fdd6e4bbb?pvs=4)

Certainly, here's the revised section for your README:

## 🧩 **4. Data Clustering**

Visualizing Clustered Programs in 2021: t-SNE with HDBSCAN Clustering and Program Efficiency

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/e424c2ba-5dc6-4851-88e8-3b7972f768d5)

In our pursuit of understanding what distinguishes regions with efficient material collection and household services from those that face challenges, we're making a critical pit stop: **Data Clustering**. Our aim is to group similar programs together, enhancing the precision of our feature analysis and unveiling hidden patterns.

Buckle up as we explore Data Clustering with the power of HDBSCAN, a highly accurate clustering method, and visualize these clusters in two dimensions using t-SNE. This promises to be an electrifying session that will reveal insights like never before! 🚀📊

**Related Scripts:** `Clustering(HDBSCAN)_Visualization(t_SNE).py`

**For more information about this section, please visit this page**: [Data Clustering](https://tasteful-background-b30.notion.site/4-Data-Clustering-1f3fc49ed986428c806098c6555dda78?pvs=4)


### 👨‍🔧 **5. Hyperparameter Tuning**

Stability Assessment: Out-of-Bag Error Rate for Random Forest

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/f887f8b1-eb09-4fda-8817-b3bd52bb826c)

In this section, we embark on the vital journey of fine-tuning machine learning models—an indispensable step to ensure the precision and accuracy of our analysis results. 🛠️

As you delve into this chapter, you'll uncover the profound importance of hyperparameter tuning. Originally conceived as an integral part of our Global Feature Analysis, its significance magnified as our analysis evolved. Observe how we meticulously adjust parameters to optimize our model's performance.

Our framework adeptly navigates complex challenges like multicollinearity, employing Randomized Search and Grid Search in tandem to fine-tune hyperparameters. The ultimate result is a rock-solid foundation for our analysis, substantiated by the insightful OOB Error vs. Number of Estimators graph. 📈

I warmly invite you to join me on this journey, whether you are already well-versed in hyperparameter tuning or simply seeking valuable data analysis insights. Your presence has been invaluable, and I assure you that this exploration promises significant returns. 🚀

Buckle up, as we prepare to venture into the next thrilling chapter—the captivating world of data analysis! 🚀 In the upcoming leg of our expedition, we will delve deeply into feature analysis, both on a global scale and with a local focus. Our goal is to discern the differences between high-performing programs and those in need of improvement.

**Related Scripts:** `Hyperparameter-tuning.py`

**For more in-depth information about this section, please visit this page:** [Hyperparameter Tuning](https://tasteful-background-b30.notion.site/5-Hyperparameter-Tuning-89e48769a75d45f69d560d64a2787596?pvs=4)


### 📊 **6. Feature Analysis**

Feature Selection Analysis: Prioritizing Important Features by Frequency

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/8e568535-6297-4aa4-9c27-5a29b7334368)

In the previous chapters, we've meticulously fine-tuned our machine-learning model, navigated the intricate landscape of hyperparameter tuning, and laid a solid foundation for our analysis. Now, it's time to turn our attention to **Feature Analysis**. 🧐 This is where we dive deep to comprehend the factors that truly matter, the features that can make or break a program's performance.

But this isn't just any feature analysis; it's a journey that involves a **two-layer feature selection** process. First, we'll employ **Feature Importance + Random Features**, introducing random twins for each feature to evaluate their importance. Then, we'll raise the stakes with **Boruta Feature Importance**, using shuffling and shadow twins to unlock even more insights into feature selection. Get ready, because we're embarking on a data-driven expedition that will transform your understanding of the Blue Box Program. 💼🔍

Our journey has been a thrilling rollercoaster, filled with discoveries and insights. From hyperparameter tuning to global feature analysis, we've built a robust framework to guide our data analysis journey. Now, the moment we've all been waiting for has arrived: Local Feature Analysis. 🌟🔎

**Related Scripts:** `Feature_Selection_V1(Random Features).py`, `Feature_Selection_V2(Boruta).py`

**For more in-depth information about this section, please visit this page:** [Feature Analysis](https://tasteful-background-b30.notion.site/6-Feature-Analysis-cb998fe7a17c4446b72d206fad7d03ec?pvs=4)


### 💫 **7. Local Feature Selection**

Local Feature Selection for Noise Cluster (-1)

![image](https://github.com/sastmo/BlueBox_Data_Analysis_ML/assets/116411251/36031e43-dc77-431c-a7da-ad3cba48e279)


Welcome to Section 7 of our exhilarating journey! From the inception of this data analysis project to the in-depth exploration of various facets of the Blue Box Program, it's incredible to see how far we've come. As we delve into Section 7, we're entering a pivotal phase that promises to bring us even closer to unraveling the program's mysteries.

In this section, we'll immerse ourselves in the captivating world of **Local Feature Selection** – a critical element in our quest for data-driven insights. Get ready to embark on a voyage filled with valuable revelations and discoveries that could shape the course of our analysis. 🌟📊

1️⃣ **At the group level:** We'll kick off by employing a systematic binary classification approach to identify the features that play a significant role in distinguishing clusters within our dataset. This involves the creation of a unique 'Target' column, transforming our challenge into a classification task. By doing so, we'll pinpoint the features that set a specific cluster apart, assigning a target value of 1 to it and 0 to the others. This transformation opens up exciting possibilities for feature analysis within each category. 🧩

2️⃣ **At the point (Observation) level:** But our journey doesn't stop there. We'll venture deeper into **point-level** feature selection, dissecting the importance of features in predicting the 'Target' value for each data point or observation. This meticulous analysis will provide profound insights into the significance of each feature in our predictive models and offer invaluable guidance for areas that can be improved. 📈🔍

As we embark on this captivating expedition into local feature selection, we extend a warm invitation for you to join us in unveiling the intricate tapestry of data analysis. Together, we'll uncover the gems hidden within the Blue Box Program's dataset and gain a deeper understanding of its inner workings. 🌐🚀

**Related Scripts for Group Level:** `Local_Feature_Selection_Group_Level.py`, `Exploratory_Data_Analysis_CLusters.py`

**Related Scripts for Observation Level:** `Local_Feature_Selection_Observation_Level.py`

**For more in-depth information about this section, please visit this page:** [Local Feature Selection](https://www.notion.so/7-Local-Feature-Selection-599e6ea05a9e43eab629f7e9192d003a)
