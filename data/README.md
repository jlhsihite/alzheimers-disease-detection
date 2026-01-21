# Dataset Information

This project uses the **Alzheimer's Prediction Dataset (Global)** from Kaggle.

## Dataset Details
- **Source:** [Kaggle - Alzheimer's Prediction Dataset](https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global)
- **Author:** Ankush Panday
- **Size:** 74,283 patient records
- **Features:** 50 (after preprocessing from 24 original features)
- **Format:** CSV file
- **Filename:** `alzheimers_prediction_dataset.csv`

## How to Get the Dataset

### Option 1: Download from Kaggle

1. Visit: https://www.kaggle.com/datasets/ankushpanday1/alzheimers-prediction-dataset-global
2. Click "Download" (requires free Kaggle account)
3. Extract the CSV file
4. Place `alzheimers_prediction_dataset.csv` in this `/data` folder

### Option 2: Use Kaggle API
```bash
# Install Kaggle API
pip install kaggle

# Download dataset (requires Kaggle API credentials)
kaggle datasets download -d ankushpanday1/alzheimers-prediction-dataset-global
unzip alzheimers-prediction-dataset-global.zip -d data/
```

## Usage in Code

After downloading, update the file path in `alzheimers_classifier.py`:
```python
# Update this line with the correct path
df = pd.read_csv("data/alzheimers_prediction_dataset.csv")
```

## License
This dataset is hosted on Kaggle under their terms of service. Please refer to the original Kaggle page for licensing information.

**The dataset is not included in this repository** due to size constraints and licensing considerations.
Please download it separately using the instructions above.
