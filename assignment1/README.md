# 📘 CS231n Assignment 1

This folder contains my implementation and experimentation for **Assignment 1 of Stanford CS231n**.

The assignment focuses on implementing **Linear SVM**, **Softmax classifier**, and a **Two-Layer Net**,  
and improving their performance through hyperparameter tuning on the CIFAR-10 dataset.

---

## 📁 Directory Structure

<pre><code>
assignment1/
├── svm.ipynb                 # SVM classifier experiment notebook
├── softmax.ipynb             # Softmax classifier experiment notebook
├── two_layer_net.ipynb       # Two-layer neural network experiment
├── README.md                 
└── ../py/                    # Core implementation files
    ├── linear_svm.py         # SVM loss and gradient
    ├── softmax.py            # Softmax loss and gradient
    ├── fc_net.py             # Two-layer network implementation
    ├── layer.py              # Affine, ReLU, and related layers
    ├── layer_utils.py        # Utility layers (Affine + ReLU)
    ├── optim.py              # Optimizers (SGD, Adam)
    ├── solver.py             # Training loop abstraction
    └── linear_classifier.py  # Shared classifier logic
</code></pre>

**Note:** The `py/` directory contains the core implementation files (`.py`), which are imported and used within the corresponding Jupyter notebooks (`.ipynb`).


---

## 📄 Assignment Overview

### 🟦 `svm.ipynb` - Linear SVM Classifier

- SVM loss and gradient computation implemented using both naive loops and vectorized operations (`linear_svm.py`)

---

### 🟨 `softmax.ipynb` - Softmax Classifier

- Softmax loss and gradient computation implemented using both naive loops and vectorized operations (`softmax.py`)

---

### 🟥 `two_layer_net.ipynb` - Two-Layer Net

- Implementation of a two-layer neural network and exploration of hyperparameter tuning
