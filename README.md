# Alzheimer's Disease Agitation Prediction Model

A machine learning pipeline for predicting agitation states in dementia patients using physiological and environmental sensor data. This model aggregates hourly sensor readings to daily patterns and uses various ML algorithms to identify days when patients may experience agitation episodes.

## Overview

This project implements a comprehensive data science pipeline that:
- Processes multimodal sensor data (physiological vitals, sleep patterns, activity sensors)
- Handles missing data through iterative imputation
- Aggregates hourly measurements to daily behavioural patterns
- Applies feature engineering and selection techniques
- Trains multiple classification models to predict daily agitation events
- Provides model evaluation and cross-validation results

## Key Features

- **Data Preprocessing**: Robust handling of missing values, outlier detection, and feature scaling
- **Feature Engineering**: Categorical encoding, boolean conversion, and temporal aggregation
- **Class Balancing**: SMOTE oversampling to handle imbalanced agitation labels
- **Feature Selection**: Random Forest-based importance ranking and selection
- **Model Comparison**: Evaluation of 6 different ML algorithms (KNN, Random Forest, Logistic Regression, Gradient Boosting, SVM, Naive Bayes)
- **Cross-Validation**: K-fold validation for robust performance estimation

## Dataset Structure

The model expects CSV files with the following structure:
- **Training Data** (`train.csv`): Labeled hourly sensor readings with agitation indicators
- **Test Data** (`test.csv`): Unlabeled hourly sensor readings for prediction

### Key Features Used:
- **Activity Sensors**: bathroom, bedroom, door, kitchen, other_location usage
- **Physiological Data**: heart rate, blood pressure (systolic/diastolic), breathing rate
- **Sleep Patterns**: time to bed, time to rise, time in/out of bed, snoring
- **Flags**: daily heart rate and blood pressure monitoring indicators
- **Sleep Category**: global time-in-bed classification

## Installation

1. Clone this repository
2. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare your data**: Place `train.csv` and `test.csv` in your data directory
2. **Update data path**: Modify the `data_path` variable in the `main()` function
3. **Run the pipeline**:
```bash
python agitation_prediction.py
```

The script will:
- Load and preprocess the data
- Train multiple models and display performance metrics
- Generate predictions for the test set
- Save results to `test_preds.csv`

## Model Performance

The pipeline evaluates multiple algorithms and selects the best performer. In testing, the K-Nearest Neighbors (KNN) classifier showed optimal performance with cross-validation accuracy scores and balanced precision/recall metrics.

## Output

- **Console Output**: Detailed training progress, model evaluation metrics, and cross-validation results
- **Visualizations**: Feature importance plots and distribution histograms
- **Predictions File**: `test_preds.csv` containing binary agitation predictions for test data

## Technical Details

- **Aggregation Strategy**: Hourly data aggregated to daily using max/sum functions
- **Missing Data**: Iterative imputation for physiological measurements
- **Scaling**: MinMax normalization for numerical features
- **Validation**: Stratified patient-level train/validation split
- **Random Seed**: Set to 100 for reproducible results

## Requirements

See `requirements.txt` for complete dependency list. Key libraries include:
- pandas, numpy for data manipulation
- scikit-learn for ML algorithms and preprocessing
- imbalanced-learn for SMOTE oversampling
- matplotlib, seaborn for visualization

## Contributing

This model is designed for healthcare research applications. When extending or modifying the code, please ensure compliance with healthcare data privacy regulations and validate any changes against clinical domain expertise.

## License

Please ensure appropriate licensing and ethical approval for any healthcare data applications.
