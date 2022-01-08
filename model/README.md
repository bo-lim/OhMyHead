## model
---

### 1. scalp_dataset_efficientnet_MinkyuKim.ipynb
김민규가 만든 모발 상태 분류 모델입니다. DenseNet에 classifier만 교체 후 따로 훈련시켜 사용하는 방식을 선택했으며 학습과 테스트는 아직 끝내지 못했습니다. 
<br>
(22.01.07) Base model을 EfficientNet에서 DenseNet으로 변경했습니다.
(22.01.08) EfficientNet 기반 네트워크에서 얻은 값을 입력받아 두피의 상태와 중증도를 판단하는 네트워크를 생성했습니다. 그리고 새로운 네트워크를 위한 손실함수, 데이터셋도 생성했습니다. 또한 학습한 모델의 parameter를 저장하고 불러오는 함수, 테스트하는 함수도 구현했습니다.
