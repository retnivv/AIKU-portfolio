# 📘 CS231n Assignment 2

이 폴더는 Stanford CS231n 과제 2의 실습 기록과 구현 내용을 정리한 공간입니다.  
과제 2는 Multi-Layer Fully Connected Network, Batch normalization, Dropout, CNN을 직접 구현하고,  
Pytorch를 활용한 모델 학습 실습 과정을 포함합니다.
---

## 📁 폴더 구조

<pre><code>assignment2/
├── README.md                 # 현재 문서
├── BatchNormalization.ipynb  			# Batch Normalization 실습 노트북
├── ../py/                   # classifier 및 layer 구현 코드
│   ├── fc_net.py            # Fully Connected Network 구현
│   ├── layers.py            # Affine, Batchnorm 등 각종 Layer 구현
│   ├── optim.py             # SGD, Adam 등 Optimizer 구현
│   └── solver.py            # 모델 학습 클래스
├── ../figures/              # 관련 이미지
</code></pre>

※ `py/` 폴더에는 핵심 구현 코드('.py' 파일)들이 들어있으며, `.ipynb` 파일에서 이를 import하여 사용합니다.

---

## 📄 과제 개요

### 🟦 `BatchNormalization.ipynb` - Batch Normalization

- **Batch Normalization 및 Layer Normalization 구현** (`layers.py`)
- 이미지 경로 추가

---

### 🟨 진행예정

- 

---

### 🟥 진행예정

- 

---

## 🧠 학습/실험 중 깨달은 점

- BatchNormalization.ipynb
  - BatchNorm의 역전파 계산과정
  - BatchNorm과 LayerNorm의 비교

---

## ✍️ 기타 정보

- 실습 환경: Google Colab  
- CIFAR-10 데이터셋 사용 

---

## 📝 참고

> 본 노트북은 학습 목적의 실습 결과로, 모든 구현은 직접 작성 및 테스트하였습니다.  
> `.ipynb` 파일은 주석 및 마크다운 셀의 정제를 최소화하고,  실습 당시의 흐름과 고민이 자연스럽게 드러나도록 원본 형태를 최대한 유지하였습니다.
> `figures/` 폴더 내의 이미지는 실습 과정 중 필요한 계산을 직접 수행하고 정리한 자료입니다.