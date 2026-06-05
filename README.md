# German Credit Risk Classifier (XGBoost)

End-to-end binary classification pipeline predicting credit risk using XGBoost. 

## Dataset Architecture
* **Source:** Statlog (German Credit Data), UCI Repository ID 144.
* **Shape:** 1000 instances, 20 features (13 categorical, 7 numeric).
* **Target:** Binary (1 = Good Risk, 2 = Bad Risk).
* **Distribution:** 70% Good, 30% Bad.

## Pipeline & Preprocessing
* **Validation Strategy:** 80/20 stratified train-test split executed *prior* to any transformations to strictly prevent data leakage.
* **Feature Engineering:** `StandardScaler` applied to numeric variables to normalize scale variance; `OneHotEncoder` applied to categorical variables to prevent false ordinality.
* **Model Selection:** `XGBClassifier`. Hyperparameters (`max_depth=3`, `n_estimators=100`, `learning_rate=0.3`) optimized via isolated iterative experiments to balance bias and variance.

## Imbalance Handling
The 70/30 class imbalance and the asymmetrical cost of classification errors (False Negatives representing direct financial loss) were addressed using a three-pronged approach:

1. **Synthetic Data Generation (SMOTE):** Applied exclusively to the training subset to oversample the minority class, ensuring the model learns distinct decision boundaries without leaking test data.
2. **Cost-Sensitive Learning:** Configured the `scale_pos_weight` parameter to apply a heavier mathematical penalty for misclassifying the minority "Bad Risk" class.
3. **Threshold Calibration:** Shifted the binary decision threshold from the default 0.5 down to 0.35. This intentionally trades absolute Precision for maximized Recall, effectively capturing a higher volume of true defaults.

## Final Model Evaluation
* **AUC Score:** 0.7762
* **Recall:** 0.6667
* **F1-Score:** 0.6061

## Usage
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn ucimlrepo

python ml.py