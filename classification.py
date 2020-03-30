#%%
from sklearn.svm import SVC
from gensim.models import doc2vec
from sklearn.metrics import roc_curve, auc, mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold 
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import joblib

#%% # 讀取建立train data 和 test data
all_data = pd.read_csv('all_data.csv')
model = doc2vec.Doc2Vec.load('doc2vec.model')

x = []
for i in range(0, len(model.docvecs)):
    x.append(model.docvecs[i].tolist())

x_train = x[:25000]
y_train = all_data['sentiment'][:25000]
x_test = x[25000:]

#%% # SVM
svc_model = GridSearchCV(SVC(), param_grid={
                        "kernel":('linear', 'poly', 'rbf', 'sigmoid')}, cv=KFold(n_splits=5),
                        scoring='accuracy').fit(x_train, y_train)
# svc_model.best_estimator_.get_params() 最佳參數
joblib.dump(svc_model, 'svc_model.pkl')

# %% # 讀取模型
svc_model = joblib.load('svc_model.pkl')

# 評估
mse = mean_squared_error(y_train, svc_model.predict(x_train))
mae = mean_absolute_error(svc_model.predict(x_train), y_train)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%svc_model.best_score_)

# ROC
y_score = svc_model.decision_function(x_train)
fpr, tpr, threshold = roc_curve(y_train, y_score)
roc_auc = auc(fpr, tpr)

# %%
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('roc.png')
plt.show()

#%% #決策樹
dtree_model = GridSearchCV(tree.DecisionTreeClassifier(), param_grid={
                        "criterion":("gini", "entropy"),
                        "splitter": ("best", "random")}, cv=KFold(n_splits=5),
                        scoring='accuracy').fit(x_train, y_train)
# dtree_model.best_estimator_.get_params() 最佳參數
joblib.dump(dtree_model , 'dtree_model.pkl')

# %% # 讀取模型
dtree_model = joblib.load('dtree_model.pkl')

# 評估
mse = mean_squared_error(y_train, dtree_model.predict(x_train))
mae = mean_absolute_error(dtree_model.predict(x_train), y_train)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%dtree_model.best_score_)

#%% # 隨機森林
rforest_model = GridSearchCV(RandomForestClassifier(), param_grid={
                        "n_estimators":range(100, 1000, 100),}, cv=KFold(n_splits=5),
                        scoring='accuracy').fit(x_train, y_train)
# rforest_model.best_estimator_.get_params() 最佳參數
joblib.dump(rforest_model , 'rforest_model.pkl')

# %% # 讀取模型
rforest_model= joblib.load('rforest_model.pkl')

# 評估
mse = mean_squared_error(y_train, rforest_model.predict(x_train))
mae = mean_absolute_error(rforest_model.predict(x_train), y_train)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%rforest_model.best_score_)

#%% # 邏輯迴歸
logistic_model = GridSearchCV(LogisticRegression(), 
                            param_grid={
                                'warm_start':('True', 'False'),
                                'solver':('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga')},
                            cv=KFold(n_splits=5),
                            scoring='accuracy').fit(x_train, y_train)
# logistic_model.best_estimator_.get_params() 最佳參數
joblib.dump(logistic_model , 'logistic_model.pkl')

# %% # 讀取模型
logistic_model= joblib.load('logistic_model.pkl')

# 評估
mse = mean_squared_error(y_train, logistic_model.predict(x_train))
mae = mean_absolute_error(logistic_model.predict(x_train), y_train)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%logistic_model.best_score_)

#%% # KNN
knn_model = GridSearchCV(KNeighborsClassifier(),
                        param_grid={'n_neighbors':range(3, 10),
                                    'weights':('uniform', 'distance'),
                                    'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute')},
                                cv=KFold(n_splits=5),
                                scoring='accuracy').fit(x_train, y_train)
# knn_model_model.best_estimator_.get_params() 最佳參數
joblib.dump(knn_model , 'knn_model.pkl')

# %% # 讀取模型
knn_model= joblib.load('knn_model.pkl')

# 評估
mse = mean_squared_error(y_train, knn_model.predict(x_train))
mae = mean_absolute_error(knn_model.predict(x_train), y_train)
print("MSE:" + str(mse))
print("MAE: " + str(mae))
print("accuracy: %f"%knn_model.best_score_)

# %% # 預測結果輸出
preds = svc_model.predict(x_test).tolist()
preds[:] = [int(x) for x in preds]

# %%
test_submission = pd.DataFrame({'id':all_data['id'][25000:].tolist(), 'sentiment':preds})
test_submission.to_csv('submission_file.csv', index=0)