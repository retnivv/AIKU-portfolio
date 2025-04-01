# 📘 CS231n Assignment 1 - Linear Classifiers

이 폴더는 Stanford CS231n 과제 1의 실습 기록과 구현 내용을 정리한 공간입니다.  
이번 과제에서는 Linear SVM과 Softmax classifier를 직접 구현하고,  
하이퍼파라미터 튜닝을 통해 CIFAR-10 데이터셋 분류 성능을 개선하는 과정을 포함합니다.

---

## 📁 폴더 구조

<pre><code>assignment1/
├── svm.ipynb                 # SVM classifier 실습 노트북
├── softmax.ipynb             # Softmax classifier 실습 노트북
├── README.md                 # 현재 문서
└── ../classifiers/           # 핵심 구현 모듈
    ├── linear_svm.py         # SVM loss 및 gradient 구현
    ├── softmax.py            # Softmax loss 및 gradient 구현
    └── linear_classifier.py  # 공통 기반 클래스 (SVM/Softmax 공통 로직)
</code></pre>

※ `classifiers/` 폴더에는 핵심 구현 코드('.py' 파일)들이 들어있으며, `.ipynb` 파일에서 이를 import하여 사용합니다.

---

## 📄 과제 개요

### 🟦 `svm.ipynb` - Linear SVM Classifier

- **SVM loss 및 gradient를 반복문 방식과 벡터화 방식으로 각각 구현** (`linear_svm.py`)
- **최종 모델 성능**
  - Best Validation Accuracy ≈ `0.382`

---

### 🟨 `softmax.ipynb` - Softmax Classifier

- **Softmax loss 및 gradient를 반복문 방식과 벡터화 방식으로 각각 구현** (`softmax.py`)
- **최종 모델 성능**
  - Best Validation Accuracy ≈ `0.401`

---

## 🧠 학습/실험 노트

- `deepcopy`를 사용하지 않아도 최적 모델 참조(`best_model`)가 안전하게 유지되는지 여부 확인(svm.ipynb)

---

## 📌 기타 정보

- 실습 환경: Google Colab  
- CIFAR-10 데이터셋 사용 

---

## 📝 참고

> 본 노트북은 학습 목적의 실습 결과로, 모든 구현은 직접 작성 및 테스트하였습니다.
> .ipynb 파일은 주석 및 마크다운 셀의 정제를 최소화하고, 실습 당시의 흐름과 고민이 자연스럽게 드러나도록 원본 형태를 최대한 유지하였습니다.