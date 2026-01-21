# Alzheimer's Disease Detection: ML Bias-Variance Study
Author: **Jessica Sihite**  

Machine learning classification system comparing three supervised learning algorithms (Logistic Regression, Decision Tree, k-NN) for early Alzheimer's disease detection using 74,000+ patient records.

Completed as part of **COMP90049: Introduction to Machine Learning** at the University of Melbourne. This project explores how different ML algorithms handle the bias-variance trade-off in high-stakes clinical applications, where misclassification costs vary significantly.

Grade: H1 (29/30) 

## 🛠️ Dependencies
- **Python 3.8+**
- **Scikit-learn** - ML algorithms, preprocessing, evaluation
- **Pandas** - Data manipulation
- **NumPy** - Numerical operations

## 📊 Dataset
- **Source:** Alzheimer's Prediction Dataset (Global) - Kaggle
- **Size:** 74,283 patient records
- **Features:** 50 (after preprocessing from 24 original features)
- **Class Distribution:** Imbalanced to reflect real-world medical data

[Dataset download instructions in `/data` folder]

## 🔬 Methodology

### Feature Engineering
- Custom feature: Cognitive test deviation from healthy baseline
- Polynomial feature expansion (degree 2)
- One-hot encoding for categorical variables
- Standardization for numerical features

### Model Evaluation
- **Outer CV:** 6-fold stratified cross-validation
- **Inner CV:** 3-fold nested cross-validation for hyperparameter tuning
- **Metrics:** Accuracy, F1-Score, Precision, Recall

### Models Compared

1. **Logistic Regression** (Low variance, high bias)
   - Tuned hyperparameter: Regularization strength (C)
   
2. **Decision Tree** (High variance, low bias)
   - Tuned: Maximum depth, minimum samples to split
   
3. **k-Nearest Neighbors** (Moderate bias-variance)
   - Tuned: Number of neighbors (k)

## 📈 Key Results

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Logistic Regression** | **0.713** | **0.681** | 0.630 | **0.741** |
| Decision Tree | 0.721 | 0.695 | 0.634 | 0.771 |
| k-NN | 0.713 | 0.635 | **0.671** | 0.604 |

**Clinical Insights:**
- Logistic Regression: Best overall performance, highest recall (fewer missed diagnoses)
- k-NN: Highest precision (fewer false alarms)
- Decision Tree: Most variable, prone to overfitting

Full analysis available in `report.pdf`

## 🚀 How to Run

1. **Clone the repository**
```bash
git clone https://github.com/jlhsihite/alzheimers-ml-detection.git
cd alzheimers-ml-detection
```
2. **Install dependencies**
```bash
pip install -r requirements.txt
```
3. **Download dataset** (see `/data/README.md`)

4. **Update file path** in `alzheimers_classifier.py`:
```python
df = pd.read_csv("data/alzheimers_prediction_dataset.csv")
```

5. **Run the classifier**
```bash
python alzheimers_classifier.py
```

## 📝 Output

The script prints three results tables:
1. Per-fold evaluation metrics for each hyperparameter combination
2. Average scores across all folds
3. Standard deviation of scores

## **Directory Structure**
```
alzheimers-disease-detection/
├── README.md                    
├── alzheimers_classifier.py     
├── report.pdf                   
├── requirements.txt             
└── data/
    └── README.md                
