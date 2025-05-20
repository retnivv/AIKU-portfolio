# 📘 CS231n Assignment 2  

이 폴더는 Stanford CS231n 과제 2의 실습 기록과 구현 내용을 정리한 공간입니다.  
과제 2는 Multi-Layer Fully Connected Network, Batch normalization, Dropout, CNN을 직접 구현하고,  
PyTorch를 활용한 모델 학습 실습 과정을 포함합니다.

---

## 📁 폴더 구조

<pre><code>
assignment2/
├── README.md                        # 현재 문서
├── BatchNormalization.ipynb        # Batch Normalization 실습 노트북
├── Dropout.ipynb                   # Dropout 실습 노트북
├── ConvolutionalNetworks.ipynb     # CNN 실습 노트북
├── PyTorch.ipynb                   # PyTorch 실습 노트북 ⭐️
├── ../py/                          # classifier 및 layer 구현 코드
│   ├── fc_net.py                   # Fully Connected Network 구현
│   ├── layers.py                   # Affine, Batchnorm, Dropout 등 Layer 구현
│   ├── cnn.py                      # Three Layer CNN 구현
│   ├── optim.py                    # SGD, Adam 등 Optimizer 구현
│   ├── layer_utils.py              # Affine+ReLU 등 조합 레이어 편의 모듈
│   ├── fast_layers.py              # 고속 Convolution Layer 모듈
│   └── solver.py                   # 모델 학습 클래스
├── ../figures/                     # 관련 이미지
</code></pre>

※ `py/` 폴더에는 핵심 구현 코드('.py' 파일)들이 들어있으며, `.ipynb` 파일에서 이를 import하여 사용합니다.
※ `figures/` 폴더 내의 이미지는 실습 과정 중 필요한 계산을 직접 수행하고 정리한 자료입니다.

---

## 📄 과제 개요

### 🟦 `BatchNormalization.ipynb` - Batch Normalization

- **Batch Normalization 및 Layer Normalization 구현** (`layers.py`)
- **실습 중 수기로 정리한 역전파 계산 그래프:**

<details> <summary><strong>📌 BN backward pass</strong></summary> <p align="center"> 
<img src="https://github.com/retnivv/AIKU-portfolio/raw/main/cs231n/assignment2/image/batchnorm_backward.jpg" width="750"/> </p> </details> <details> <summary><strong>📌 BN alternative backward pass</strong></summary> <p align="center"> 
<img src="https://github.com/retnivv/AIKU-portfolio/raw/main/cs231n/assignment2/image/batchnorm_backward_alt.jpg" width="750"/> </p> </details>

---

### 🟨 `Dropout.ipynb` - Dropout

- **Dropout 계층의 forward & backward 구현** (`layers.py`)
- **Dropout을 적용했을 때와 적용하지 않았을 때 small dataset으로 학습된 모델의 성능 비교**

---

### 🟥 `ConvolutionalNetworks.ipynb` - CNN

- **Convolutional layer forward & backward 구현** (`layers.py`)
- **Three Layer CNN 구현** (`cnn.py`)
- **(+) Spatial Batch Normalization, Spatial Group Normalization 구현** (`layers.py`)

---

### 🟩⭐️ `PyTorch.ipynb` - Pytorch (Final Project : CIFAR-10 classification)

- **PyTorch 사용법 학습**
- **PyTorch를 활용하여 CIFAR-10 이미지셋 분류 작업 수행**
- **최종 모델 성능**
  - Best Validation Accuracy ≈ `0.85`
  - Test Accuracy ≈ `0.83`
- **모델 구조**  
   - 5개의 Residual Block (Conv-BN-ReLU 포함) + FC 2층 + Softmax  
   - Dropout 및 L2 정규화 적용


## 🧠 학습/실험 중 깨달은 점

- BatchNormalization.ipynb
  - BatchNorm의 역전파 계산과정
  - BatchNorm과 LayerNorm의 비교
- Dropout.ipynb
  - Dropout의 정규화 기능
- ConvolutionalNetworks.ipynb
  - Convolutional layer의 (naive) forward, backward 계산 과정
  - Spatial Batch Normalization, Spatial Group Normalization과 BatchNorm, LayerNorm의 비교
- PyTorch.ipynb
  - PyTorch 사용법
  - nn.Module vs nn.Sequential
  - Residual Network PyTorch로 구현

---

## ✍️ 기타 정보

- 실습 환경: Google Colab  
- CIFAR-10 데이터셋 사용 

---

## 📝 참고

> 본 노트북은 학습 목적의 실습 결과로, 모든 구현은 직접 작성 및 테스트하였습니다.  
> `.ipynb` 파일은 주석 및 마크다운 셀의 정제를 최소화하고, 실습 당시의 흐름과 고민이 자연스럽게 드러나도록 원본 형태를 최대한 유지하였습니다.
> `figures/` 폴더 내 이미지 또한 원본 계산 흐름을 보존하고자 그대로 첨부하였습니다.
