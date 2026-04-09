# Strokes Gained Prediction Project

## Overview

This project uses machine learning to predict **Strokes Gained Total** for PGA Tour players based on performance statistics. It applies regression models to analyze how different aspects of a player’s game contribute to overall performance.

---

## Dataset

* File: `pgaTourData.csv`
* Contains player statistics such as:

  * Fairway Percentage
  * Average Driving Distance
  * Greens in Regulation (GIR)
  * Average Putts
  * Scrambling

---

## Methods

Two models are implemented:

* **Linear Regression** (baseline)
* **Random Forest Regressor** (improved performance)

The workflow includes:

1. Data loading and cleaning
2. Feature selection
3. Train-test split
4. Model training and evaluation
5. Visualization of results

---

## How to Run

1. Place `pgaTourData.csv` in the same folder as the script
2. Run the script:

   ```bash
   python v2_final_code.py
   ```

---

## Output

* Model performance metrics (R², RMSE, cross-validation scores)
* Generated visualisations:

  * Predicted vs Actual values
  * Feature importance
  * Residual plot
* Saved as: `figures.png`

---

## Notes

* The code uses relative paths to ensure portability across devices
* Missing values are handled using median imputation

---

## Author

Project developed as part of coursework (SPC4004).

