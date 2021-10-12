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
### [2) 결정 트리(decision tree)](DecisionTree.ipynb)
- 분류와 회귀, 다중출력도 가능한 다목적 머신러닝
- 규칙 노드(Decision Node) : 규칙 조건, 리프 노드(Leaf Node) : 결정된 클래스 값
- 새로운 규칙 조건마다 서브트리 생성, 깊이(depth)가 깊어질수록 결정 트리의 예측 성능 저하 가능성 높음 -> 사전에 제한 필요
- 적은 노드로 높은 예측을 목표 -> 어떻게 트리를 분할해서 균일한 데이터 셋을 구성할 수 있을까??
- Graphviz : 시각화할 수 있는 패키지
### 3) 앙상블(ensemble learning)
- 예측기로부터 예측을 수집한 모델
- 예측기가 가능한 한 서로 독립적일 때 최고 성능
- 각기 다른 알고리즘으로 학습(VotingClassifier)
  - 하드 보팅(hard voting) : 다수의 분류기가 결정한 최종 보팅 결괏값으로 선정
  - 소프트 보팅(soft voting) : 확률을 모두 더하고 이를 평균해서 가장 높은 레이블 값 선정 -> 더 많이 
- 같은 알고리즘을 사용하고 샘플링을 서로 다르게 학습
  - 배깅(bagging) : 훈련 세트에서 중복을 허용하여 샘플링
    - obb 평가 : 선택되지 않은 훈련 샘플, obb샘플로 평가 가능
  - 페이스팅(pasting) : 중복을 허용하지 않고 샘플링
  - 배깅이 페이스팅보다 편향이 조금 더 높지만 예측기들의 상관관계를 줄여 앙상블의 분산을 감소 => 배깅을 더 많이 사용(교차검증으로 더 나은 쪽 선택)
  - 랜덤 포레스트(random forest) : 배깅(페이스팅)을 적용한 결정트리 앙상블(결정트리 기반 알고리즘), 하이퍼파라미터가 너무 많다
- 부스팅(boosting) : 약한 학습기를 여러 개 연결하여 강한 학습기를 만드는 방법
  - 에이다부스트(ada) : 오류 데이터에 가중치를 부여하면서 부스팅을 수행
  - GBM(gradient) : 가중치 업데이트를 경사하강법으로, 회귀도 가능, 랜덤 포레스트보다 성능이 뛰어난 경우가 많지만 오래걸림
    - XGBoost(eXtra Gradient) : **트리기반 앙상블 중 가장 각광**, 자동 과적합 규제, 내장된 교차 검증, 결손값 자체 처리, Tree prunning(나무 가지치기)-긍정 이득이 없는 분할을 가지치기
## 6. 모델 조정
### 1) 서포트 벡터 머신
- C
- gamma
### 2) 결정 트리, randomforest, GBM
- min_samples_split : 노트 분할 최소 샘플 데이터 수, 디폴트-2, 작을수록 과적합 가능성 증가, 과적합 제어
- min_samples_leaf : 말단 노드가 되기 위한 최소 샘플 데이터 수, 과적합 제어 용도, 비대칭적 데이터의 경우 작게 설정
- max_features : 최대 피처 개수, 디폴트-None(전체 수), int타입-개수, float타입-퍼센트, sqrt(=auto)-전체 수 제곱근, log-log2(전체 수)
- max_depth : 최대 깊이, 디폴트-None, 깊어지면 과적합 위험
- max_leaf_nodes : 말단 노드 최대 
- cv : kfold 수
- (+ random forest) n_estimators : 결정 트리의 개수, 디폴트-10, 많을수록 좋은 성능
- (+ GBM) loss : 경사하강법에서 사용할 비용함수
- (+ GBM) learning_rate : 학습을 진행할 때마다 적용하는 학습률, 0~1 사이 값(디폴트-0.1), 너무 작으면 성능이 높지만 오래걸림, 너무 크면 성능이 떨어짐, n_estimators와 상호 보완
- (+ GBM) subsample : 데이터 샘플링 비율, 디폴트-1(전체)
### 3) XGBoost
- 일반 파라미터 : 디폴트값을 바꾸는 경우가 거의 없음
  - booster : gbtree(tree based model), gblinear(linear model)
  - silent : 디폴트-0, 출력메세지를 나타내고 싶지 않ㅇ르 경우 1
  - nthread : 스레드 개수, 디폴트-전체
- 부스터 파라미터 : 트리 최저고하, 부스팅, regularization 등과 관련 파라미터
  - eta : 학습률, 0~1사이, 디폴트-0.1, 0.01~0.2 사이의 값 선호
  - num_boost_rounds : = n_estimators
  - min_child_weight: 디폴트-1, 추가적으로 가지를 나눌지 결정하기 위해 필요한 데이터들의 weight 총합, 클수록 분할 자제
  - gamma : 디폴트-0, 최소 손실 감소 값, 해당 값 보다 크면 리프 분리, 값이 클수록 과적합
  - colsample_bytree : = max_features, 디폴트=1
  - lambda : L2 Regularization 적용 값, 피처가 많을수록 적용 검토, 값이 클수록 과적합 감소
  - scale_pos_weight : 디폴트-1, 비대칭 클래스 데이터 셋의 균형을 유지하기 위해 사용
- 학습 태스크 파라미터 : 학습 수행 시의 객체 함수, 평가를 위한 지표 등
  - 
### 4) 회귀 조정
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
