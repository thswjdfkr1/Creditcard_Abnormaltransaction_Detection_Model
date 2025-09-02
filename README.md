 # 주제   
신용카드 이상거래 탐지모델 개발     

# 데이터탐색     
<img width="801" height="574" alt="image" src="https://github.com/user-attachments/assets/ba71f3c2-fc04-471b-997a-c1d8a9f3e17a" />     

* 정상거래인 0과 이상거래인 1의 차이가 매우 심한 불균형데이터     

# 초기 모델링    
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/a58dd878-6c6b-42de-bac4-650136aa54c8" />     
 
| model	| precision_test	| recall_test |	f1_score_test |    
| ----- | ----- | ----- | ----- |
| 0	| MLP	| 0.87	| 0.81	| 0.84 |      
| 1	| LightGBM	| 0.94	| 0.82	| 0.87 |      
| 2	| ANN	| 0.80	| 0.82	| 0.81 |        
| 3	| Random Forest	| 0.90	| 0.80	| 0.85 |      
| 4	| XGBoost	| 0.95	| 0.81	| 0.87 |     
| 5	| Decision Tree	| 0.80	| 0.78	| 0.79 |     
| 6	| AdaBoost	| 0.84	| 0.78	| 0.80 |     
| 7	| KNeighbors	| 0.88	| 0.78	| 0.83 |      
| 8	| Logistic Regression	| 0.89	| 0.71	| 0.79 |      
| 9	| SVM	| 0.92	| 0.68	| 0.78 |      
| 10	| Gradient Boost	| 0.62	| 0.15	| 0 |        

* 가장 성능이 높은 XGBoost와 LGBMClassifier 모델을 통한 추가적인 모델링 진행     

# 랜덤언더샘플링     
```
under_df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = under_df.loc[df['Class'] == 1]
non_fraud_df = under_df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
random_under_df = normal_distributed_df.sample(frac=1, random_state=rs)

random_under_df.head()
```
<img width="1527" height="821" alt="image" src="https://github.com/user-attachments/assets/bb63862f-0f52-4db6-acf3-5f1a1b2b5401" />     
* 큰 샘플의 갯수를 작은 샘플의 갯수로 맞춰줌     

# smote 오버샘플링 + RandomSearch     
```
print('Length of X (train): {} | Length of y (train): {}'.format(len(original_Xtrain), len(original_ytrain)))     
print('Length of X (test): {} | Length of y (test): {}'.format(len(original_Xtest), len(original_ytest)))     

# List to append the score and then find the average     
lgbm_accuracy_lst = []
lgbm_precision_lst = []
lgbm_recall_lst = []
lgbm_f1_lst = []
lgbm_auc_lst = []

# Implementing SMOTE Technique
# Cross Validating the right way
# Parameters

lgbm_params = {
    'n_estimators': [50, 100, 150],
}

# # # RandomizedSearchCV 객체 생성
rand_lgbm = RandomizedSearchCV(LGBMClassifier(n_jobs=-1,verbosity=-1, boost_from_average=False,
                            random_state=rs), lgbm_params, n_iter=5)

# 교차 검증 루프
for train_idx, test_idx in sss.split(original_Xtrain, original_ytrain):
    # SMOTE와 RandomizedSearchCV를 포함한 파이프라인
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy='minority', random_state=42), rand_lgbm)

    # 훈련 데이터로 모델 학습
    pipeline.fit(original_Xtrain[train_idx], original_ytrain[train_idx])

    # 최적 모델 추출
    lgbm_best_est = rand_lgbm.best_estimator_

    # 테스트 데이터에 대한 예측
    y_test = original_ytrain[test_idx]
    X_test = original_Xtrain[test_idx]
    y_pred = lgbm_best_est.predict(X_test)

    # 확률 기반으로 AUC 계산
    y_prob = lgbm_best_est.predict_proba(X_test)[:, 1]

    lgbm_accuracy_lst.append(pipeline.score(X_test, y_test))
    lgbm_precision_lst.append(precision_score(y_test, y_pred))
    lgbm_recall_lst.append(recall_score(y_test, y_pred))
    lgbm_f1_lst.append(f1_score(y_test, y_pred))
    lgbm_auc_lst.append(roc_auc_score(y_test, y_prob))

print('---' * 45)
print('')
print("accuracy: {}".format(np.mean(lgbm_accuracy_lst)))
print("precision: {}".format(np.mean(lgbm_precision_lst)))
print("recall: {}".format(np.mean(lgbm_recall_lst)))
print("f1: {}".format(np.mean(lgbm_f1_lst)))
print('---' * 45)
```

# smote 오버샘플링 + GridSearch     
```
lgbm_params = {'num_leaves': [3, 5, 7],
               'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.03, 0.05, 1],
    'n_estimators': [50,100, 150],
}
grid_lgbm_sm = GridSearchCV(LGBMClassifier(), lgbm_params, cv=5, error_score='raise')
grid_lgbm_sm.fit(Xsm_train, ysm_train)
print(grid_lgbm_sm.best_params_)
lgbm_param_sm = grid_lgbm_sm.best_estimator_

y_pred_lgb_over = lgbm_best_est.predict(x_train)
```     

# Under_Over Sampling 비교     
<img width="1189" height="790" alt="image" src="https://github.com/user-attachments/assets/be353099-2e27-4318-8ee3-46d52d8025b0" />     

| model	| accuracy	| precision	recall	| f1_score	| roc_auc |     
| ----- | ----- | ----- | ----- | ----- |
| 0	| XGboost_under	| 0.13	| 0.00	| 0.99	| 0.56	| 0.56 |
| 1	| Xgboost_over	| 1.00	| 0.95	| 0.93	| 0.97	| 0.97 |   
| 2	| LightGBM_over	| 1.00	| 0.92	| 0.92	| 0.96	| 0.96 |      
| 3	| LightGBM_under	| 0.40	| 0.00	| 0.90	| 0.65	| 0.65 |           

# 결론     
* SMOTE 적용: 불균형 데이터셋에 SMOTE를 적용해 레이블 불균형(사기 거래보다 정상 거래가 더 많음)을 해결함.     
* 모델 성능 차이: 오버 샘플링된 데이터셋에서 신경망이 무작위 언더샘플링 데이터셋을 사용하는 모델 보다 사기 거래를 올바르게 예측하는 경우가 적었음.     
* 정상 거래 오탐지: 언더샘플링 데이터에서는 많은 정상 거래를 사기 거래로 잘못 분류하는 문제가 발생함. 이는 고객 불만과 금융 기관의 단점으로 어이질 수 있음.     
* 데이터 셔플링: 데이터 셔플링을 구현했기 때문에 예측과 정확도가 변동될 수 있음.     
