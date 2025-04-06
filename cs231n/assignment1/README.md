# 📘 CS231n Assignment 1

이 폴더는 Stanford CS231n 과제 1의 실습 기록과 구현 내용을 정리한 공간입니다.  
과제 1은 Linear SVM, Softmax classifier, 그리고 Two-Layer Net을 직접 구현하고,  
하이퍼파라미터 튜닝을 통해 CIFAR-10 데이터셋 분류 성능을 개선하는 과정을 포함합니다.

---

## 📁 폴더 구조

<pre><code>assignment1/
├── svm.ipynb                 # SVM classifier 실습 노트북
├── softmax.ipynb             # Softmax classifier 실습 노트북
├── two_layer_net.ipynb             # Two-layer Net 실습 노트북
├── README.md                 # 현재 문서
└── ../py/           # classifier 및 layer 구현 코드
    ├── linear_svm.py         # SVM loss 및 gradient 구현
    ├── softmax.py            # Softmax loss 및 gradient 구현
    ├── fc_net.py            # Two-Layer Net 구현
    ├── layer.py            # Affine, Relu 등 각종 Layer 구현
    ├── layer_utils.py		# Affine+Relu Layer 구현
    ├── optim.py            # SGD, Adam 등 Optimizer 구현
    ├── solver.py            # 모델 학습 클래스
    └── linear_classifier.py  # 공통 기반 클래스 (SVM/Softmax 공통 로직)
</code></pre>

※ `py/` 폴더에는 핵심 구현 코드('.py' 파일)들이 들어있으며, `.ipynb` 파일에서 이를 import하여 사용합니다.

---

## 📄 과제 개요

### 🟦 `svm.ipynb` - Linear SVM Classifier

- **SVM loss 및 gradient를 반복문 방식과 벡터화 방식으로 각각 구현** (`linear_svm.py`)
- **최종 모델 성능**
  - Best Validation Accuracy ≈ `0.382`
  - Test Accuracy ≈ `0.366`

---

### 🟨 `softmax.ipynb` - Softmax Classifier

- **Softmax loss 및 gradient를 반복문 방식과 벡터화 방식으로 각각 구현** (`softmax.py`)
- **최종 모델 성능**
  - Best Validation Accuracy ≈ `0.401`
  - Test Accuracy ≈ `0.386`

---

### 🟥 `two_layer_net.ipynb` - Two-Layer Net

- **2층 신경망 구현 및 하이퍼파라미터 튜닝 과정 실습**
- **최종 모델 성능**
  - Best Validation Accuracy ≈ `0.539`
  - Test Accuracy ≈ `0.518`

---

## 🧠 학습/실험 중 깨달은 점

- svm.ipynb & softmax.ipynb
  - `deepcopy`를 사용하지 않아도 최적 모델 참조(`best_model`)가 안전하게 유지되는지 여부 확인

- two_layer_net.ipynb
  - 하이퍼파라미터 튜닝의 흐름 실습
  - Adam과 SGD의 비교
  - 다중 반복문을 피하기 위한 itertools.product 사용
  - uniform(5e-5, 5e-3)과 5 * 10 ** uniform(-5,-3)의 차이

---

## ✍️ 기타 정보

- 실습 환경: Google Colab  
- CIFAR-10 데이터셋 사용 

---

## 📝 참고

> 본 노트북은 학습 목적의 실습 결과로, 모든 구현은 직접 작성 및 테스트하였습니다.  
> `.ipynb` 파일은 주석 및 마크다운 셀의 정제를 최소화하고,  실습 당시의 흐름과 고민이 자연스럽게 드러나도록 원본 형태를 최대한 유지하였습니다.