# 📘 CS231n Assignment 2  

This folder contains my implementation and experimentation for **Assignment 2 of Stanford CS231n**.

The assignment covers the implementation of **multi-layer fully connected networks**, **Batch Normalization**, **Dropout**, and **Convolutional Neural Networks (CNNs)**,  
and includes hands-on training experiments using **PyTorch**.

---

## 📁 Directory Structure

<pre><code>
assignment2/
├── README.md                       
├── BatchNormalization.ipynb        # Batch Normalization experiments
├── Dropout.ipynb                   # Dropout experiments
├── ConvolutionalNetworks.ipynb     # Convolutional Neural Network experiments
├── PyTorch.ipynb                   # PyTorch experiments
├── ../py/                          # Core implementation files
│   ├── fc_net.py                   # Fully connected network implementation
│   ├── layers.py                   # Affine, BatchNorm, Dropout layers
│   ├── cnn.py                      # Three-layer CNN implementation
│   ├── optim.py                    # Optimizers (SGD, Adam)
│   ├── layer_utils.py              # Utility layers (Affine + ReLU, etc.)
│   ├── fast_layers.py              # Fast convolutional layer implementations
│   └── solver.py                   # Training loop abstraction
├── ../figures/                     # Supporting figures
</code></pre>

**Note:** The `py/` directory contains the core implementation files (`.py`), which are imported and used within the corresponding Jupyter notebooks (`.ipynb`).  
**Note:** The `figures/` directory includes manually computed and organized materials used during experimentation.

---

## 📄 Assignment Overview

### 🟦 `BatchNormalization.ipynb` – Batch Normalization

- **Implementation of Batch Normalization and Layer Normalization** (`layers.py`)

---

### 🟨 `Dropout.ipynb` – Dropout

- **Forward and backward implementations of the Dropout layer** (`layers.py`)
- **Comparison of model behavior with and without Dropout on a small dataset**

---

### 🟥 `ConvolutionalNetworks.ipynb` – Convolutional Neural Networks

- **Forward and backward implementations of convolutional layers** (`layers.py`)
- **Implementation of a three-layer CNN** (`cnn.py`)
- **Additional implementations of Spatial Batch Normalization and Spatial Group Normalization** (`layers.py`)

---

### 🟩 `PyTorch.ipynb` – PyTorch (CIFAR-10 Classification)

- **Introduction to PyTorch fundamentals**
- **Image classification on the CIFAR-10 dataset using PyTorch**
