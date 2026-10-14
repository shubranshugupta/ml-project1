# Machine Learning Course Project-I

**Academic Institution:** International Institute of Information Technology, Bangalore (IIIT-B)  
**Course:** Machine Learning (M.Tech)  
**Team Members:** Hitesh Dahiya, Shubhranshu Gupta, Anjali Kiran, Priti Aher  
**Evaluation Rubric:** Data Preprocessing (2M) | Model Diversity & Experimentation (3M) | Viva & Technical Report (5M) | Kaggle Leaderboard Performance (5M) — **Total: 15 Marks**  
**Project Guidelines:** [ML_Project_Statement.pdf](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/ML_Project_Statement.pdf)  
**Submission Date:** October 15, 2026  

---

## Executive Summary

This repository contains the end-to-end implementation, exploratory analysis, preprocessing pipelines, model architectures, validation protocols, and technical reports for **Machine Learning Course Project-I**. The project encompasses two distinct supervised machine learning tracks evaluated in competitive Kaggle-style benchmarks:

1. **Track 1: Flight Price Prediction (Supervised Regression)**
   - **Task:** Predict airline ticket prices (`price` in INR) from itinerary, scheduling, airline, and route metadata.
   - **Key Innovations:** Hierarchical median duration recovery; flight-prefix carrier mappings; out-of-fold target encoding across 11 categorical dimensions; market segmentation (Economy vs. Business); comparable itinerary fare statistical features; multi-stage Ridge residual correction; and geometric/arithmetic model ensembling.
   - **Top Result:** **Public Kaggle Leaderboard $R^2 = \mathbf{0.96181}$** (Geometric Ensemble of Ridge LR + Random Forest + Extra Trees).

2. **Track 2: VehicleVision Classification (Supervised 3-Class Multiclass Classification)**
   - **Task:** Classify vehicles into **Sedan (`0`)**, **SUV (`1`)**, or **Sports Car (`2`)** using 20 continuous engineering and aerodynamic specifications.
   - **Key Innovations:** Rigorous data hygiene removing 2,500 duplicate records; leakage-proof `StratifiedGroupKFold` validation; 17 physics-derived domain features (powertrain, geometry, aerodynamics, energy/handling); and comparative benchmarking of regularized Logistic Regression vs. 1,600-tree hybrid ensembles vs. soft-voting classifiers.
   - **Top Result:** **Validation Accuracy = $\mathbf{88.35\%}$** (Standalone regularized Logistic Regression) and **$87.73\%$** on strict deduplicated grouped validation.

---

## Repository Structure

```
project1/
├── ML_Project_Statement.pdf                # Official course project specification & evaluation rubric
├── README.md                               # Comprehensive M.Tech project documentation
├── LICENSE                                 # MIT License
│
├── flight/                                 # Track 1: Flight Price Prediction (Regression)
│   ├── Report_Flight_Price_Prediction.pdf  # Comprehensive technical report (6 pages)
│   ├── flight_lr.ipynb                     # Segmented Ridge regression, itinerary fares & residual stacking
│   ├── flight_dt.ipynb                     # 32-feature pipeline: Random Forest & Extra Trees with OOF encodings
│   ├── flight_ensemble.ipynb               # Ensembling: Geometric/Arithmetic blends, NNLS, & submission export
│   └── data/
│       ├── train.csv                       # Training dataset (15,589 rows × 12 columns)
│       ├── test.csv                        # Test dataset (3,898 rows × 11 columns)
│       ├── sample_submission.csv           # Benchmark submission format
│       ├── submission_ensemble_simple.csv  # Best submission file (Public R² = 0.96181)
│       ├── submission_dt_simple.csv        # Tree-only submission (RF + ET blend)
│       └── submission5.csv                 # Intermediate ensemble submission
│
└── vehicle/                                # Track 2: VehicleVision Classification (Multiclass)
    ├── Report_VehicleVision_Classification.pdf # Comprehensive technical report (5 pages)
    ├── vehicle_logistic.ipynb              # Baseline & standalone Logistic Regression pipeline
    ├── vehicle_dt.ipynb                    # 37-feature domain engineered 5-branch hybrid (1,600 trees + LR)
    ├── vehicle_ensemble.ipynb              # Deduplicated StratifiedGroupKFold pipeline & 95/5 soft voting
    └── data/
        ├── vehicle_train_10000.csv         # Raw training dataset (10,000 rows × 22 columns)
        ├── vehicle_test.csv                # Test dataset (2,500 rows × 21 columns)
        ├── vehicle_sample_submission.csv   # Benchmark submission format
        ├── vehicle_submission.csv          # Standalone retrained LR predictions
        ├── submission_hybrid_super_blend.csv # 37-feature engineered probability hybrid blend
        └── submission_simple_lr_bagging.csv # Cleaned grouped 95/5 soft voting predictions
```

---

## Evaluation Rubric Alignment

| Rubric Component | Weightage | Track 1: Flight Price Prediction Implementation | Track 2: VehicleVision Classification Implementation |
| :--- | :---: | :--- | :--- |
| **Data Preprocessing** | **2 Marks** | • Flight prefix carrier mapping to recover missing airlines.<br>• Hierarchical median imputation for duration.<br>• Cross-fitted target encoding for 11 categories.<br>• Derived booking windows & relative duration metrics. | • Identification & removal of 2,500 duplicate records.<br>• Conversion of infinite values to NaN with median imputation.<br>• Standard scaling in unified pipeline (`make_pipeline`).<br>• Prevention of train-val leakage via grouped split. |
| **Variety of Models Tried** | **3 Marks** | • Ridge Regression with pairwise interactions.<br>• Market-segmented models (Economy vs. Business).<br>• Multi-stage Ridge residual error correction.<br>• Random Forest (180 trees, depth 25).<br>• Extra Trees (180 trees, depth 25).<br>• NNLS Stacking, Geometric & Arithmetic Ensembles. | • L2-Regularized Logistic Regression (`lbfgs`).<br>• Bagged Decision Trees (Gini & Entropy splits).<br>• Random Forest (500 trees).<br>• Extra Trees (500 trees).<br>• 1,600-tree + LR probability hybrid.<br>• Soft-voting ensemble (`VotingClassifier`). |
| **Viva + Technical Report** | **5 Marks** | Full documentation in [Report_Flight_Price_Prediction.pdf](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/Report_Flight_Price_Prediction.pdf): mathematical formulations, error analyses, validation trade-offs, and design choices. | Full documentation in [Report_VehicleVision_Classification.pdf](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/Report_VehicleVision_Classification.pdf): physics-based feature derivations, leakage analysis, and linear vs. tree separability. |
| **Leaderboard Performance** | **5 Marks** | **Public Kaggle Leaderboard $R^2 = \mathbf{0.96181}$** achieved by the 50/50 Geometric Ensemble (`submission_ensemble_simple.csv`). | Robust validation accuracy of **$88.35\%$** (raw stratified) and **$87.73\%$** (deduplicated grouped), with class F1 up to **0.92**. |
| **Total** | **15 Marks** | Full pipeline verified, reproducible, and documented. | Full pipeline verified, reproducible, and documented. |

---

# Track 1: Flight Price Prediction

## 1. Problem Formulation & Dataset Schema
The objective is supervised regression: predict airline ticket pricing in Indian Rupees (`price`) given journey and booking parameters.

- **Training Records:** 15,589 rows × 12 columns
- **Test Records:** 3,898 rows × 11 columns (target `price` held out)
- **Primary Metrics:** Coefficient of Determination ($R^2$), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).

### Feature Dictionary
| Feature | Type | Description | Observed Domain Range / Cardinality |
| :--- | :---: | :--- | :--- |
| `id` | Categorical/ID | Unique itinerary row identifier | Retained for submission mapping; excluded from features |
| `airline` | Categorical | Operating air carrier name | 18 airlines (3,089 missing in train; 799 in test) |
| `flight` | Categorical | Specific flight code (e.g., `IX 1070`, `SG 8169`) | 1,070 unique flight codes |
| `source_city` | Categorical | City of departure | 6 metropolitan hubs |
| `departure_time` | Categorical | Departure departure time-window bucket | 6 buckets (Morning, Early Morning, Evening, Night, etc.) |
| `stops` | Categorical | Number of intermediate flight layovers | `zero`, `one`, `two_or_more` |
| `arrival_time` | Categorical | Flight arrival time-window bucket | 6 buckets |
| `destination_city` | Categorical | Arrival destination city | 6 metropolitan hubs |
| `flight_class` | Categorical | Cabin seating tier | `Economy`, `Business` |
| `duration` | Numerical | Total elapsed travel duration in hours | 0.92 to 38.08 hrs (3,095 missing in train; 776 in test) |
| `days_left` | Numerical | Days remaining between booking date and flight departure | 1 to 57 days |
| `price` | Numerical | **Target variable**: Final ticket price (INR) | ₹1,105 to ₹123,071 (heavily bimodal by cabin tier) |

---

## 2. Exploratory Data Analysis & Strategic Observations
1. **Pervasive Missing Values:** Approximately 20% of `airline` and `duration` values are missing. Deleting rows would sacrifice over 39% of the training dataset. We resolved this through deterministic carrier mapping from flight prefixes and hierarchical route medians.
2. **Cabin-Class Bimodality:** Business class tickets occupy a completely distinct pricing distribution (typically ₹40,000 - ₹100,000+) compared to Economy class (₹2,500 - ₹15,000), necessitating segmented modeling or logarithmic transformation.
3. **Nonlinear Booking Dynamics:** Ticket prices exhibit non-linear acceleration within the final 2–5 days before departure, requiring dedicated urgency indicators and reciprocal transforms ($1 / \text{days\_left}$).
4. **Duration as an Itinerary Signature:** Duration strongly correlates with specific routes and stop counts, allowing precise imputation using grouped medians.

---

## 3. Data Preprocessing & Feature Engineering

### 3.1 Tree Models Pipeline (32 Engineered Features in [flight_dt.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/flight_dt.ipynb))
- **Carrier Recovery:** Extracted airline prefix from `flight` (e.g., `IX` $\to$ Air India Express / Wayfarer Aviation).
- **Hierarchical Duration Imputation:**
  $$\text{Imputed Duration} = \text{Median}(\text{flight, route}) \to \text{Median}(\text{flight}) \to \text{Median}(\text{route, stops}) \to \text{Global Median}$$
- **Cross-Fitted Target Encoding:** Target-encoded 11 high-cardinality categorical features using 5-fold internal out-of-fold cross-fitting with smoothing to eliminate data leakage.
- **Booking Horizon Proxies:** Engineered $\log(\text{days\_left})$, $1/\text{days\_left}$, and binary thresholds for urgency: $\mathbb{I}(\text{days\_left} \le 2)$, $\mathbb{I}(\text{days\_left} \le 5)$, and $\mathbb{I}(\text{days\_left} \ge 25)$.
- **Route Efficiency & Relative Duration:**
  - $\text{Excess Duration} = \text{duration} - \min_{\text{route}}(\text{duration})$
  - $\text{Duration Ratio} = \frac{\text{duration}}{\min_{\text{route}}(\text{duration}) + 10^{-5}}$
  - $\text{Duration Per Stop} = \frac{\text{duration}}{\text{stops\_count} + 1}$
- **Frequency Encoding:** Frequency counts of flight code, carrier, route, and carrier-route pairs.
- **Logarithmic Target Scaling:** Models trained on $y = \log(1 + \text{price})$ to stabilize variance and minimize relative percentage error.

### 3.2 Linear Regression Pipeline (Segmented & Residual-Corrected in [flight_lr.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/flight_lr.ipynb))
- **Market Segmentation:** Data split into independent sub-models for Economy vs. Business (subdivided by carrier families 6E/AI/IX vs. premium carriers).
- **Itinerary Peer Statistics:** Extracted historical training fare aggregates (median, 25th percentile, minimum, mean) grouped by identical itinerary matching (`flight`, `route`, `stops`), backing off hierarchically to route-level aggregates.
- **Pairwise Feature Interactions & Scaling:** One-hot encoding of categorical dimensions + `StandardScaler` + second-order interaction terms fitted with penalized L2 Ridge regression.
- **Multi-Stage Residual Ridge Corrections:** Fitted auxiliary Ridge models sequentially to the residuals $\epsilon = y - \hat{y}$ using flight codes and progressively granular route-cabin-horizon keys.
- **Nearby-Day Kernel Smoothing:** Comparable fares smoothed using an exponential decay kernel over booking horizon:
  $$w_i = \exp\left(-\frac{|\text{days\_left}_i - \text{days\_left}_{\text{target}}|}{5}\right)$$
- **Non-Negative Least Squares (NNLS) Stacking:** Combined segment models and nearby-day estimators under non-negativity constraints.

---

## 4. Model Architectures & Validation Results

### 4.1 Comparative Model Performance (Flight Track)
All models were evaluated using standardized validation protocols: 5-fold Out-of-Fold (OOF) cross-validation for tree models, and an independent 3,118-row evaluation holdout for ensembling in [flight_ensemble.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/flight_ensemble.ipynb).

| Model / Pipeline | Validation Protocol | RMSE (INR) ↓ | MAE (INR) ↓ | Validation $R^2$ ↑ | Public Leaderboard $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Geometric 50/50 Ensemble (LR × [RF + ET])** | **Ensemble Eval (3,118 rows)** | **9,537.66** | **3,203.79** | **0.9354** | **0.96181 (Best)** |
| Arithmetic 50/50 Blend (LR + [RF + ET]) | Ensemble Eval (3,118 rows) | 9,532.92 | 3,207.82 | 0.9355 | — |
| Weighted Average (31.3% LR + 68.7% Trees) | Ensemble Eval (3,118 rows) | 9,608.19 | 3,159.49 | 0.9345 | — |
| NNLS Stacking (Segmented) | Ensemble Eval (3,118 rows) | 9,639.09 | 3,116.08 | 0.9341 | — |
| Segment-Weighted Blend | Ensemble Eval (3,118 rows) | 9,640.60 | 3,169.40 | 0.9340 | — |
| Ridge Stacking ($\alpha = 10$) | Ensemble Eval (3,118 rows) | 9,644.42 | 3,182.16 | 0.9340 | — |
| Main Segmented Ridge LR Pipeline | Ensemble Eval (3,118 rows) | 9,682.56 | 3,447.28 | 0.9335 | — |
| Tree Pipeline: RF + ET Blend | Ensemble Eval (3,118 rows) | 9,888.58 | 3,175.73 | 0.9306 | — |
| *Extra Trees Regressor (180 trees)* | *5-Fold OOF* | *10,329.75* | *3,141.54* | *0.9303* | — |
| *Random Forest Regressor (180 trees)* | *5-Fold OOF* | *10,425.36* | *3,207.66* | *0.9290* | — |
| *Decision Tree Blend (RF + ET)* | *5-Fold OOF* | *10,312.52* | *3,154.79* | *0.9305* | — |

> **Key Insight:** While the simple arithmetic blend edged out the geometric blend by ₹4.73 RMSE on the local holdout, the **Geometric Average** ($\hat{y} = \sqrt{\hat{y}_{\text{LR}} \cdot \hat{y}_{\text{Trees}}}$) generalized significantly better across extreme price ranges, resulting in the winning **$R^2 = \mathbf{0.96181}$** on the Kaggle public test set.

---

# Track 2: VehicleVision Classification

## 1. Problem Formulation & Dataset Schema
The objective is supervised multiclass classification: classify a vehicle into one of three vehicle categories based on comprehensive continuous mechanical, structural, and physical attributes.

- **Classes:** `0: Sedan` | `1: SUV` | `2: Sports Car`
- **Training Records:** 10,000 raw rows (7,500 unique after deduplication) × 22 columns
- **Test Records:** 2,500 rows × 21 columns (`vehicle_class` held out)
- **Primary Metrics:** Classification Accuracy, Precision, Recall, Macro/Weighted F1-score.

### Raw Features (20 Continuous Physical Predictors)
- **Engine & Performance:** `engine_displacement_l`, `horsepower_hp`, `engine_cylinders`, `torque_nm`, `acceleration_0_100_kmh_s`, `battery_capacity_kwh`.
- **Dimensions & Chassis:** `vehicle_weight_kg`, `wheelbase_mm`, `vehicle_length_mm`, `vehicle_width_mm`, `vehicle_height_mm`, `ground_clearance_mm`, `turning_radius_m`, `cargo_capacity_l`, `tire_width_mm`.
- **Dynamics & Efficiency:** `fuel_tank_capacity_l`, `city_fuel_efficiency_kmpl`, `highway_fuel_efficiency_kmpl`, `braking_distance_m`, `drag_coefficient`.

---

## 2. Critical Data Hygiene & Leakage Prevention
1. **Detection of 2,500 Duplicate Samples:** Exploratory analysis revealed that exactly 2,500 rows in `vehicle_train_10000.csv` had identical feature vectors and identical labels to other rows.
2. **Data Leakage Hazard:** A standard unstratified or random `train_test_split` would place duplicate copies of the same vehicle in both train and validation sets, producing artificially inflated, ungeneralizable accuracy scores.
3. **Methodological Fix:** In [vehicle_ensemble.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_ensemble.ipynb), duplicate feature-and-label rows were removed to establish a clean 7,500-sample training corpus. A `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=2026)` validation scheme was enforced where identical measurement vectors formed isolated groups, guaranteeing zero data leakage across folds.

---

## 3. Domain Feature Engineering (17 Derived Features)
Implemented in `create_super_features()` within [vehicle_dt.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_dt.ipynb), expanding feature dimensionality from 20 to 37 predictors:

| Category | Derived Feature | Mathematical Formulation | Domain Justification |
| :--- | :--- | :--- | :--- |
| **Powertrain Ratios** | `power_to_weight` | $\text{horsepower} / \text{weight}$ | Primary discriminator for Sports Cars |
| | `torque_to_weight` | $\text{torque} / \text{weight}$ | High low-end pull distinguishing SUVs/Sports Cars |
| | `battery_to_weight` | $\text{battery\_kwh} / \text{weight}$ | Hybrid/EV powertrain packaging |
| | `battery_to_accel` | $\text{battery\_kwh} / (\text{accel} + 10^{-5})$ | Instantaneous electrical acceleration capability |
| | `power_per_cyl` | $\text{horsepower} / (\text{cylinders} + 10^{-5})$ | Specific engine output efficiency |
| | `torque_per_disp` | $\text{torque} / (\text{displacement} + 10^{-5})$ | Volumetric forced-induction proxy |
| **Geometry & Chassis**| `volume_est` | $(\text{length} \times \text{width} \times \text{height}) \times 10^{-9}$ | Total vehicle bounding box envelope ($m^3$) |
| | `density_est` | $\text{weight} / (\text{volume\_est} + 10^{-5})$ | Material density and structural compactness |
| | `wheelbase_to_length`| $\text{wheelbase} / \text{length}$ | Cabin space proportion vs. overhangs |
| | `length_to_width` | $\text{length} / \text{width}$ | Aspect ratio of chassis |
| **Aerodynamics** | `frontal_area` | $(\text{width} \times \text{height}) \times 10^{-6}$ | Frontal silhouette area ($m^2$) |
| | `drag_area` | $\text{frontal\_area} \times C_d$ | Total effective aerodynamic resistance |
| **Handling & Energy** | `kinetic_energy` | $\frac{1}{2} \cdot \text{weight} \cdot (100 / 3.6)^2$ | Kinetic energy at highway speed (100 km/h) |
| | `braking_force` | $\text{kinetic\_energy} / (\text{braking\_dist} + 10^{-5})$ | Braking dissipation force demand |
| | `avg_fuel_efficiency`| $\frac{1}{2}(\text{city\_eff} + \text{highway\_eff})$ | Combined cycle fuel efficiency |
| | `fuel_tank_to_weight`| $\text{fuel\_tank} / \text{weight}$ | Endurance / utility packaging index |
| | `low_slung_ratio` | $\text{ground\_clearance} / \text{height}$ | Center-of-gravity proxy separating SUVs from Sedans |

---

## 4. Model Architectures & Validation Results

### 4.1 Evaluated Architectures
1. **Standalone Logistic Regression ([vehicle_logistic.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_logistic.ipynb)):** Median imputation + `StandardScaler` + multinomial Logistic Regression ($C=1.0$, `solver="lbfgs"`, `max_iter=3000`). Evaluated via 80/20 stratified split.
2. **Engineered Tree Hybrid ([vehicle_dt.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_dt.ipynb)):** 5-model ensemble on 37 features (300 Gini Bagged Trees, 300 Entropy Bagged Trees, 500 Random Forest, 500 Extra Trees, plus scaled Logistic Regression — total 1,600 trees). Blended probability output:
   $$P_{\text{tree}} = \frac{1}{4}(P_{\text{gini}} + P_{\text{entropy}} + P_{\text{RF}} + P_{\text{ET}}), \quad P_{\text{hybrid}} = 0.40 \cdot P_{\text{tree}} + 0.60 \cdot P_{\text{LR}}$$
3. **Cleaned Grouped Soft Voting ([vehicle_ensemble.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_ensemble.ipynb)):** Pipeline trained on 7,500 deduplicated records evaluated on 1,500 grouped validation rows. Soft voting combines 95% Logistic Regression + 5% Bagged Trees (80 estimators).

### 4.2 Comparative Performance Table (Vehicle Track)
| Model Implementation | Validation Protocol | Validation Accuracy | Sedan F1 | SUV F1 | Sports Car F1 | Generated Submission File |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Standalone Logistic Regression** | **80/20 Raw Stratified Split (2,000 val rows)** | **88.35%** | — | — | — | [`vehicle_submission.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/data/vehicle_submission.csv) |
| Logistic Regression (Cleaned Data) | Grouped Split on Cleaned Data (1,500 val rows) | **87.73%** | 0.83 | 0.90 | 0.92 | — |
| Soft Voting Ensemble (95% LR + 5% Bagging) | Grouped Split on Cleaned Data (1,500 val rows) | **87.67%** | 0.83 | 0.89 | 0.92 | [`submission_simple_lr_bagging.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/data/submission_simple_lr_bagging.csv) |
| Bagged Decision Trees (80 Trees) | Grouped Split on Cleaned Data (1,500 val rows) | 87.13% | 0.82 | 0.88 | 0.91 | — |
| 37-Feature 1,600-Tree Hybrid + LR | Full-Fit Test Inference (10,000 rows fit) | *Priors: 33.3% each* | — | — | — | [`submission_hybrid_super_blend.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/data/submission_hybrid_super_blend.csv) |

### 4.3 Detailed Classification Report (Grouped Cleaned Split)
$$\text{Macro Average F1} = \mathbf{0.88}, \quad \text{Weighted Average F1} = \mathbf{0.88}$$

| Class Index & Name | Precision | Recall | F1-Score | Support (Validation Rows) |
| :--- | :---: | :---: | :---: | :---: |
| **0: Sedan** | 0.83 | 0.83 | 0.83 | 521 |
| **1: SUV** | 0.88 | 0.91 | 0.89 | 495 |
| **2: Sports Car** | 0.93 | 0.90 | 0.92 | 484 |
| **Total / Overall** | **0.88** | **0.88** | **0.88** | **1,500** |

### 4.4 Diagnostic Insights & Viva Defense
- **High Linear Separability:** Continuous physical variables (ground clearance, power-to-weight, vehicle height) define smooth decision planes in $\mathbb{R}^{20}$. Consequently, L2-regularized Logistic Regression generalizes slightly better than non-linear tree ensembles, which tend to create step-like overfitted boundaries in sparse regions.
- **Sports Car Classification:** Achieves highest precision (**0.93**), driven by extreme low-slung ratios, elevated horsepower-to-weight, and minimal ground clearance.
- **SUV Classification:** Achieves highest recall (**0.91**), separated reliably by cargo volume, frontal drag area, and elevated chassis height.
- **Sedans as Intermediate Manifold:** Sedans occupy the intermediate boundary region between compact low-slung sports models and high-riding SUVs, accounting for slightly lower F1 (0.83).

---

## Summary of Kaggle Submission Artifacts

| Track | Submission File Name | Model Description | Record Count | Checks & Verifications |
| :--- | :--- | :--- | :---: | :--- |
| **Flight** | [`submission_ensemble_simple.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/data/submission_ensemble_simple.csv) | **Geometric 50/50 Blend:** $\sqrt{\hat{y}_{\text{LR}} \cdot \hat{y}_{\text{Trees}}}$ | 3,898 | Validated: Zero NaNs, strictly positive prices, sorted test IDs, **Public $R^2 = \mathbf{0.96181}$** |
| **Flight** | [`submission_dt_simple.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/data/submission_dt_simple.csv) | Equal blend of 5-fold RF (180 trees) + ET (180 trees) | 3,898 | Validated: Positive log-inverted outputs, bounds clipped |
| **Vehicle** | [`submission_hybrid_super_blend.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/data/submission_hybrid_super_blend.csv) | 37-feature hybrid: 4 tree ensembles (1,600 trees) + LR | 2,500 | Validated: Exactly balanced test predictions (~33.3% per class) |
| **Vehicle** | [`submission_simple_lr_bagging.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/data/submission_simple_lr_bagging.csv) | Deduplicated 95% LR + 5% Bagged Trees soft vote | 2,500 | Fully asserted: ID order preserved, class range $\in \{0, 1, 2\}$ |
| **Vehicle** | [`vehicle_submission.csv`](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/data/vehicle_submission.csv) | Standalone L2-regularized Logistic Regression pipeline | 2,500 | Validated: 88.35% validation pipeline refitted on all records |

---

## How to Reproduce & Execute

### 1. Environment Requirements
All code runs standard Python 3.9+ with scientific computing packages:
```bash
pip install numpy pandas scipy scikit-learn jupyter
```

### 2. Execution Order
Each notebook is self-contained and operates directly from within its respective directory or repository root:

- **Flight Track:**
  1. Open and execute [flight/flight_lr.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/flight_lr.ipynb) to fit the segmented Ridge regression, calculate peer statistics, and generate baseline linear predictions.
  2. Open and execute [flight/flight_dt.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/flight_dt.ipynb) to compute the 32 engineered features, execute 5-fold OOF target encodings, fit Random Forest & Extra Trees, and output `submission_dt_simple.csv`.
  3. Open and execute [flight/flight_ensemble.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/flight_ensemble.ipynb) to run comparative ensemble evaluations and generate the winning Kaggle submission `submission_ensemble_simple.csv`.

- **Vehicle Track:**
  1. Open and execute [vehicle/vehicle_logistic.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_logistic.ipynb) to run the 80/20 stratified baseline LR model and produce `vehicle_submission.csv`.
  2. Open and execute [vehicle/vehicle_dt.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_dt.ipynb) to construct the 17 domain physics features and fit the 5-branch 1,600-tree hybrid model.
  3. Open and execute [vehicle/vehicle_ensemble.ipynb](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/vehicle_ensemble.ipynb) to run duplicate detection, `StratifiedGroupKFold` validation, and generate `submission_simple_lr_bagging.csv`.

---

## Academic Honesty & Technical Reports
Detailed LaTeX-typeset technical project reports containing formal mathematical notations, residual histograms, calibration plots, and in-depth discussions are available in:
- [flight/Report_Flight_Price_Prediction.pdf](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/flight/Report_Flight_Price_Prediction.pdf)
- [vehicle/Report_VehicleVision_Classification.pdf](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/vehicle/Report_VehicleVision_Classification.pdf)

---

## Team Members

This project was developed and submitted by:
- **Hitesh Dahiya**
- **Shubhranshu Gupta**
- **Anjali Kiran**
- **Priti Aher**

*M.Tech in Computer Science and Engineering / Data Science, International Institute of Information Technology, Bangalore (IIIT-B).*

---

## License

This project is licensed under the terms of the MIT License. See the [LICENSE](file:///c:/Users/shubh/Downloads/iiitb/academics/ml/project1/LICENSE) file for complete license details.


