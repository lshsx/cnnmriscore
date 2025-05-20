# cnnmriscore
This project aims to develop a hierarchical deep learning framework based on structural brain MRI, primarily for disease diagnosis classification such as Alzheimer's disease (AD). The framework incorporates three types of loss functions:

🧮 Regression Loss – for continuous cognitive score prediction

🧠 Classification Loss – for distinguishing subject classes (e.g., AD vs NC)

🔗 Fusion Loss – for integrating features across multiple tasks/models

📁 Project Structure

```

├── main_loss123.py       # Main training script (3 models with hierarchical losses)
├── fusion.py             # Fusion module (combines scores and features)
├── test_123.py           # Evaluation script (separate and fused model testing)
├── dataloader_one.py     # Data loading 
├── config.py             # Configurations (paths, hyperparameters, etc.)
└── README.md             # Project documentation


````

---

📦 Requirements

- Python 3.8+
- PyTorch >= 1.10
- numpy
- nibabel
- pandas
- scikit-learn

Install dependencies with:

```bash
pip install -r requirements.txt
````

---

📊 Data Preparation

* MRI NIfTI files (`.nii` or `.nii.gz`) should be placed in a folder like:

```
./data/24train/AD/
```
