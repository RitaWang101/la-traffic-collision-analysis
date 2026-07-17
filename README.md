# Los Angeles Traffic Collision Severity Analysis

## Project Overview

This project analyzes traffic collision data in Los Angeles to understand patterns related to collision severity, high-risk locations, demographic factors, and time-based risk trends. The project combines exploratory data analysis, spatial analysis, feature engineering, and machine learning to classify collision severity and identify factors associated with fatal or severe injuries.

The analysis was completed in a Jupyter Notebook using Python, with a focus on building an end-to-end analytical workflow from raw data cleaning to model evaluation and insight generation.

## Objectives

* Clean and preprocess Los Angeles traffic collision records.
* Extract temporal, demographic, spatial, and collision-type features.
* Map traffic collision MO codes into severity labels.
* Analyze fatal and severe injury patterns across time, location, and victim demographics.
* Use DBSCAN and spatial risk scoring to identify high-risk geographic areas.
* Train a Random Forest classifier to predict collision severity.
* Evaluate model performance with special attention to rare but critical classes such as fatal and severe injury collisions.
* Analyze predicted severity probabilities for young female victims aged 15–25.

## Dataset

The project uses Los Angeles traffic collision data stored as:

```text
Traffic_Collision_Data.csv
```

The raw dataset contains over 620,000 records. After cleaning and filtering invalid or incomplete records, the final modeling dataset contains approximately 360,000 severity-labeled collision records.

Key fields used in the analysis include:

* Date Occurred
* Time Occurred
* Area Name
* MO Codes
* Victim Age
* Victim Sex
* Victim Descent
* Location
* Latitude
* Longitude

## Methodology

### 1. Data Cleaning

The notebook first loads the raw traffic collision dataset and keeps the columns needed for analysis. Missing values, invalid coordinates, duplicate records, and records outside the Los Angeles bounding box are removed.

Main cleaning steps include:

* Parsing date and time into a unified datetime field.
* Extracting latitude and longitude from the location field.
* Removing records with missing MO codes, victim age, sex, or descent.
* Filtering invalid zero coordinates.
* Dropping duplicate collision records.

### 2. Feature Engineering

Additional features are created to support exploratory analysis and modeling:

* Hour
* Day of week
* Weekend indicator
* Month
* Time period
* Age group
* Sex category
* Descent category
* Collision type flags
* Hit-and-run indicator
* Intersection indicator

### 3. Severity Label Extraction

Collision severity is extracted from MO codes and mapped into five classes:

| MO Code | Severity Label    |
| ------- | ----------------- |
| 3027    | Fatal             |
| 3024    | Severe Injury     |
| 3025    | Visible Injury    |
| 3026    | Complaint of Pain |
| 3028    | Non-Injury        |

The final labeled dataset includes the following severity distribution:

| Severity          |   Count | Percentage |
| ----------------- | ------: | ---------: |
| Fatal             |   3,059 |       0.8% |
| Severe Injury     |  14,797 |       4.1% |
| Visible Injury    |  58,646 |      16.3% |
| Complaint of Pain | 135,877 |      37.7% |
| Non-Injury        | 148,308 |      41.1% |

### 4. Exploratory Data Analysis

The EDA section examines:

* Severity distribution
* Collision count by hour
* Fatal rate by hour
* Severity by time period
* Fatal rate by age group
* Fatal rate by sex
* Fatal rate by collision type
* Spatial distribution of fatal and severe collisions
* Class imbalance in severity prediction

Several visualizations are generated, including severity distribution charts, temporal pattern charts, demographic comparison charts, and spatial collision maps.

### 5. Spatial Risk Analysis

The project uses DBSCAN to identify clusters of fatal and severe injury collisions. A KDE heatmap is also generated to visualize high-density areas.

Because fatal collisions are spread across the city, the notebook also creates an area-level spatial risk score based on fatal collision rate by LAPD division. This score is normalized between 0 and 1 and used as a modeling feature.

The highest-risk LAPD areas by fatal rate include:

* Newton
* Hollenbeck
* Foothill
* Southeast
* Rampart
* Northeast

### 6. Machine Learning Model

A Random Forest classifier is trained to predict collision severity using engineered features.

The model uses:

* 70% training set
* 15% validation set
* 15% test set
* Stratified splitting by severity class
* `class_weight='balanced'` to address severe class imbalance

Model features include:

* Hour
* DayOfWeek
* IsWeekend
* Month
* TimePeriod
* AgeGroup
* Sex
* Descent
* Spatial risk score
* Vehicle vs pedestrian flag
* Vehicle vs bike flag
* Vehicle vs vehicle flag
* Vehicle vs motorcycle flag
* Hit-and-run flag
* Intersection flag
* DUI-related flag
* Street racing flag

## Model Performance

The final Random Forest model achieved the following test set performance:

| Metric                |  Value |
| --------------------- | -----: |
| Accuracy              | 58.35% |
| Macro Avg Recall      | 42.37% |
| Weighted Avg F1-score | 57.45% |

Because fatal collisions account for less than 1% of the dataset, the project prioritizes recall and interpretability over accuracy alone.

Selected test set results:

| Severity      | Precision | Recall | F1-score |
| ------------- | --------: | -----: | -------: |
| Fatal         |    0.1195 | 0.2026 |   0.1504 |
| Severe Injury |    0.2024 | 0.3119 |   0.2455 |
| Non-Injury    |    0.6776 | 0.8459 |   0.7524 |

## Key Findings

* Fatal collisions are highly imbalanced, representing only about 0.8% of labeled records.
* Spatial risk score is the most important feature in the Random Forest model.
* Month, hour, vehicle-vs-pedestrian collisions, and day of week are also important predictors.
* Some LAPD areas show higher fatal collision rates, especially Newton, Hollenbeck, and Foothill.
* Young female victims aged 15–25 have the highest predicted probabilities for complaint of pain and non-injury outcomes, while severe and fatal outcomes remain lower-probability but still important to monitor.

Top Random Forest feature importances:

| Feature            | Importance |
| ------------------ | ---------: |
| spatial_risk_score |     0.1411 |
| Month              |     0.1178 |
| Hour               |     0.1165 |
| TC_VehVsPed        |     0.0999 |
| DayOfWeek          |     0.0792 |
| AgeGroup           |     0.0751 |

## Generated Outputs

The notebook generates multiple intermediate datasets and visualization files, including:

```text
collisions_clean.csv
collisions_eda.csv
collisions_with_severity.csv
collisions_high_severity.csv
collisions_final.csv
fig_severity_dist.png
fig_timeperiod_type.png
fig_class_imbalance.png
fig_spatial_bubble.png
fig_dbscan_clusters.png
fig_kde_heatmap.png
fig_confusion_matrix.png
fig_feature_importance.png
fig_young_female_proba.png
```

## Tools and Libraries

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy
* Jupyter Notebook

## How to Run

1. Clone this repository.

```bash
git clone https://github.com/RitaWang101/la-traffic-collision-analysis.git
cd la-traffic-collision-analysis
```

2. Install the required Python libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

3. Place the raw data file in the project folder.

```text
Traffic_Collision_Data.csv
```

4. Open the notebook.

```bash
jupyter notebook "la_traffic_collision_severity_analysis.ipynb"
```

5. Run all cells from top to bottom.

## Project File

```text
la_traffic_collision_severity_analysis.ipynb
```

This notebook contains the full workflow, including data cleaning, exploratory analysis, spatial risk scoring, Random Forest modeling, model evaluation, and subgroup analysis.

## Notes

The raw data file is not included in this repository due to file size and data storage limitations. To reproduce the analysis, users should place the original `Traffic_Collision_Data.csv` file in the same directory as the notebook before running the code.
