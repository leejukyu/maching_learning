# 머신러닝 시작하기
## 1. 큰 그림 보기
### 1) 문제 정의
#### (1) 비즈니스의 목적 정의
#### (2) 현재 솔루션 구성 확인
#### (3) 지도학습, 비지도학습, 강화학습?
- 지도학습 : 정답이 있는 데이터를 활용해 학습 ex) 분류, 회귀
- 비지도학습 : 정답 라벨이 없는 데이터를 비슷한 특징끼리 군집화해 예측 ex) 클러스터링, k means
- 강화학습 : 자신이 한 행동에 대해 보상을 받으며 학습 ex) DQN, A3C
#### (4) 분류, 회귀?
- 분류(Classifier) : 클래스 예측
  - 이진 분류기 : logistic regression, support vector machine 등
  - 다중 분류기 : SGD, random forest, naive Bayes 등
- 회귀(Regression) : 값 예측
  - 선형회귀(linear regression)
  - 다항 회귀(polynomial regression)
#### (5) 배치학습, 온라인학습?
- 온라인 학습 : 새로운 샘플이 생겼을 때 점진적으로 학습하는 것
### 2) 성능 측정 지표 선택
#### (1) 회귀문제의 전형적인 성능지표 => 평균제곱오차(RMSE)
#### (2) 이상치로 보이는 구역이 많은 회귀문제 => 평균절대오차(MAE)
#### (3) 분류
- 교차 검증을 사용한 정확도 측정
  - cross_val_score(모델명, x_train, y_train, cv, scoring)
  - cv - 폴드 수
  - scoring - "accuracy"
  - 성능이 좋지만 교차 검증 점수가 나쁘면 과대적합, 양쪽 모두 좋지 않으면 과소 적합
- 오차 행렬
  - confusion_matrix(y_train, y_train_pred)
  - y_train_pred - cross_val_predict(모델명, x_trian, y_train, cv), 각 테스트 폴드에서 얻은 에측값
- 정밀도(precision)와 재현율(recall =민감도(sencitivity))
  - 정밀도 : 양성 예측의 정확도, precision_score(y_train, y_train_pred)
  - 재현율 : 정확하게 감지한 양성 샘플의 비율, recall_score(y_train, y_train_pred)
  - F1 : 정밀도와 재현율의 조화 평균, f1_score(y_train, y_train_pred)
  - 정밀도/재현율 트레이드 오프 : 정밀도 99%를 달성하자! -> 재현율 얼마에서??
    - 훈련 세트에 있는 모든 샘플의 점수 구하기 : y_scores = cross_val_predict(모델명, x_train, y_train, cv, method="decision_function")
    - decision_function : 결정함수, 각 샘플의 점수를 계산
    - precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)
    - threshold : 임곗값, 임곗값이 오르면 정밀도가 높아지고 내리면 재현율이 높아짐
- ROC곡선과 곡선 아래 면적(AUC)
  - 거짓 양성 비율에 대한 진짜 양성 비율
  - 민감도에 대한 1-특이도(진짜 음성 비율) 그래프
  - roc_aur_curve(y_train, y_scores)
  - 완벽한 분류기 AUC = 1, 완전한 랜덤 분류기 AUC = 0.5

### 3) 가정검사
## 2. 데이터를 구한다
## 3. 탐색 및 시각화
## 4. 머신러닝 알고리즘을 위한 데이터 준비
## 5. 모델을 선택하고 훈련
### 1) 서포트 벡터 머신(SVM, support vector machine)
- 클래스를 구분하는 결정 경계와 샘플 사이의 마진을 가능한 가장 크게 하는 것이 목적(완벽하게 나누는 것과 가장 넓은 도로를 만드는 것 사이의 절충)
- 매우 강력하고 선형이나 비선형 분류, 회귀, 이상치 탐색에도 사용 가능한 다목적 머신러닝 모델
- 복잡한 분류 문제, 작거나 중간 크기의 데이터셋에 적합
- 서포트 벡터 : 도로 경계에 위치한 샘플에 의해 전적으로 결정되는 샘플
- 가장 먼저 선형커널(LinearSVC)를 시도하고 훈련세트가 너무 크지 않으면 가우시안 RBF커널도 시도, 시간이 된다면 교차 검증과 그리드 탐색
### 2) 결정 트리(decision tree)
- 분류와 회귀, 다중출력도 가능한 다목적 머신러닝
## 6. 모델 조정
### 1) 서포트 벡터 머신
- C
- gamma
### 1) 회귀 조정
#### (1) 규제 : 과대적합 감소, 훈련이 끝나면 규제가 없는 성능 지표로 평가, 릿지가 기본이지만 특성이 몇 개 뿐이면 라쏘나 엘라스틱넷
- 릿지 회귀(ridge)
  - 규제항(노름의 제곱을 2로 나눈 것)을 비용 함수에 추가하여 모델의 가중치가 가능한 작게 유지
  - 하이퍼 파라미터 알파는 모델을 얼마나 많이 규제할지 조절, 0이면 선형회귀와 동일
- 라쏘 회귀(lasso)
  - 규제항(가중치 벡터의 노름)을 비용 함수에 추가하여 덜 중요한 특성의 가중치를 제거
- 엘라스틱넷(elastic net)
  - 릿지와 라쏘 절충 모델
- 조기 종료
  - 검증 에러가 최솟값에 도달하면 훈련 중지
## 7. 솔루션 제시
## 8. 시스템 론칭, 모니터링, 유지 보수
