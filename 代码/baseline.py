from collections import Counter
import tensorflow as tf
import tensorflow.keras.backend as K
import pandas as pd
import numpy as np
import xgboost as xgb
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier, RUSBoostClassifier
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier, \
    GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from catboost import CatBoostClassifier, Pool, metrics
# from catboost.utils import get_roc_curve, select_threshold
# from frozendict import frozendict
import lightgbm as lgb
from scipy import special, optimize
import os
from imbens.ensemble import SMOTEBoostClassifier, BalanceCascadeClassifier

from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from load_data1 import y, X

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# 为每个标签训练一个模型
f1_score_list = []
accuracy_score_list = []
precision_score_list = []
recall_score_list = []
auc_score_list = []
matrix_list = []
aupr_score_list = []
mcc_score_list = []
gm_score_list = []


for table in y.columns:
    # 获取存在值的数据
    rows = np.isfinite(y[table]).values  # 存在值的行
    X_fin = X[rows].values
    y_fin = y[table][rows].values

    # 将0、1对换（因为1是positive，0是negative)
    index_0 = list(np.where(y_fin == 0)[0])
    index_1 = list(np.where(y_fin == 1)[0])
    y_fin[index_0] = 1
    y_fin[index_1] = 0

    # 按照3:1:1的比例划分训练集、验证集和测试集
    X_train, X_tmp, y_train, y_tmp = train_test_split(X_fin, y_fin, train_size=0.6, test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, train_size=0.5, test_size=0.5)

    # 训练集
    train_label = y_train.astype(np.int_)  # 训练的label
    train_data = X_train  # 训练的data
    # train_label = np.array(train_label).reshape(-1, 1)
    # encoder = OneHotEncoder()
    # train_label = encoder.fit_transform(train_label).toarray()

    # 验证集
    val_label = y_val.astype(np.int_)  # 验证的label
    val_data = X_val  # 验证的data

    # 测试集
    test_label = y_test.astype(np.int_)  # 验证的label
    test_label = np.array(test_label).reshape(-1, 1)
    encoder = OneHotEncoder()
    test_label = encoder.fit_transform(test_label).toarray()
    test_data = X_test  # 验证的data

    # Z-Score标准化
    standard_scaler = StandardScaler()
    train_data = standard_scaler.fit_transform(train_data)
    val_data = standard_scaler.transform(val_data)
    test_data = standard_scaler.transform(test_data)

    # 特征填充+选择
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_data = imputer.fit_transform(train_data)
    val_data = imputer.transform(val_data)
    test_data = imputer.transform(test_data)
    print('特征填充完成！')

    # # 重采样
    # # 对少类数据进行过采样
    # sm = SMOTE(random_state=42)
    # X_resampled, y_resampled = sm.fit_resample(X_train[rows_train].values, train_label)
    # 对多类数据进行欠采样
    # cc = ClusterCentroids(random_state=42)
    # cc = RandomUnderSampler(sampling_strategy='all')
    # X_resampled, y_resampled = cc.fit_resample(X_train[rows_train].values, train_label)
    #
    #
    #
    # # 过采样+欠采样
    # # smote_enn = SMOTETomek(random_state=0)
    # print(Counter(list(train_label)))
    # # X_resampled, y_resampled = smote_enn.fit_resample(X_train[rows_train].values, train_label)
    # print(Counter(list(y_resampled)))

    # # LR
    # clf = LogisticRegression(penalty='l2').fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # SVM
    # clf = LinearSVC(penalty='l2',  # 正则化
    #                 C=67.7,
    #                 tol=1e-2)  # 停止标准的误差
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict(test_data)
    # y_pred = np.array(y_pred).reshape(-1, 1)
    # encoder = OneHotEncoder()
    # y_pred = encoder.fit_transform(y_pred).toarray()  # y转换为one-hot形式

    # # 不平衡的随机森林
    # clf = BalancedRandomForestClassifier(n_estimators=250, sampling_strategy='all')
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # 不平衡的Bagging
    # # 基分类器为树
    # clf = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=250,
    #                                 sampling_strategy='all')
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # new = []
    # for pre in y_pred:
    #     p = pre
    #     p[1] = pre[1] / c
    #     p[0] = 1 - p[1]
    #     new.append(p)
    # y_pred = np.array(new)

    # # 不平衡的Boosting方法
    # # AdaBoost
    # clf = RUSBoostClassifier(n_estimators=250, algorithm='SAMME.R', sampling_strategy='all')
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # SMOTEBoost
    # clf = SMOTEBoostClassifier(random_state=0)
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # BalanceCascade
    # clf = BalanceCascadeClassifier()
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)


    # # logistic regression
    # LR = LogisticRegression(C=1.0, penalty='l2', tol=0.01)
    # # LR.fit(X_train[rows_train].values, train_label)
    # LR.fit(X_resampled, y_resampled)
    # y_pred = LR.predict(X_val[rows_val].values)

    # # EasyEnsembleClassifier
    # from imblearn.ensemble import EasyEnsembleClassifier
    # clf = EasyEnsembleClassifier(random_state=0)
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # DT
    # clf = DecisionTreeClassifier(random_state=0)
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # KNN
    # clf = KNeighborsClassifier()
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # GNB
    # clf = GaussianNB()
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # adaboost
    # clf = AdaBoostClassifier()
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # ET
    # clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # # GBDT
    # clf = GradientBoostingClassifier(n_estimators=100, random_state = 0)
    # clf.fit(train_data, train_label)
    # y_pred = clf.predict_proba(test_data)

    # XGB
    clf = xgb.XGBClassifier()
    clf.fit(train_data, train_label)
    y_pred = clf.predict_proba(test_data)

    # # random forests
    # clf = RandomForestClassifier(n_estimators=250)
    # clf.fit(train_data, train_label)  # 训练
    # y_pred = clf.predict_proba(test_data)

    # # 1-hidden NN
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Dense(2000, activation=tf.nn.relu, input_shape=(train_data.shape[1],)),
    #     tf.keras.layers.Dense(100, activation=tf.nn.relu),
    #     tf.keras.layers.Dense(1, activation='sigmoid')
    # ])[focal_loss(alpha=.25, gamma=2)]
    # model = TabNet(train_data.shape[1], 1, 2, 10, 10)
    # model.compile(optimizer='adam',
    #               loss=[tf.keras.losses.BinaryCrossentropy(), mask_loss],
    #               loss_weights=[1, 0.01],
    #               metrics=['accuracy'])
    # model.fit(train_data, train_label, epochs=30)  # 训练模型
    # probability_model = tf.keras.Sequential([
    #     model,
    #     tf.keras.layers.Softmax()
    # ])
    # y_pred = model.predict(val_data)
    # new_y_pred = [[1-p[0], p[0]] for p in y_pred]
    # print(y_pred)
    # y_pred = np.array(new_y_pred)

    # # CatBoost
    # params = frozendict({
    #     'loss_function': 'Logloss',
    #     'eval_metric': 'AUC',
    #     'iterations': 500,
    #     'scale_pos_weight': w,
    #     'learning_rate': 0.15,
    #     'depth': 5,
    #     'l2_leaf_reg': 3
    # })
    # pool = Pool(train_data, train_label)
    # clf = CatBoostClassifier(**params)
    # clf.fit(train_data, train_label)
    # roc_curve_values = get_roc_curve(clf, pool)
    # boundary = select_threshold(clf, curve=roc_curve_values, FNR=0.01)  # 根据训练集选择阈值
    # if boundary > 0.5:
    #     boundary = 0.5
    # print('阈值：', boundary)
    # y_pred = clf.predict_proba(val_data)

    # 指标
    y_score = y_pred.copy()
    y_true = test_label

    auc_score_list.append(roc_auc_score(y_true, y_score))  # AUC
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(y_pred, axis=1)
    precision, recall, thresholds = precision_recall_curve(y_true, y_score[:, 0])
    aupr_score_list.append(auc(recall, precision))  # AUPR
    f1_score_list.append(f1_score(y_true, y_pred, average='macro'))  # F1
    accuracy_score_list.append(accuracy_score(y_true, y_pred))  # accuracy
    precision_score_list.append(precision_score(y_true, y_pred, average='macro'))   # precision
    recall_score_list.append(recall_score(y_true, y_pred, average='macro'))  # recall
    mcc_score_list.append(matthews_corrcoef(y_true, y_pred))  # mcc
    C = confusion_matrix(y_true, y_pred)  # 混淆矩阵
    matrix_list.append(C)

print('f1-score:{:.3f}±{:.3f}'.format(np.mean(f1_score_list), np.std(f1_score_list)))
print('accuracy:{:.3f}±{:.3f}'.format(np.mean(accuracy_score_list), np.std(accuracy_score_list)))
print('precision:{:.3f}±{:.3f}'.format(np.mean(precision_score_list), np.std(precision_score_list)))
print('recall:{:.3f}±{:.3f}'.format(np.mean(recall_score_list), np.std(recall_score_list)))
print('AUC:{:.3f}±{:.3f}'.format(np.mean(auc_score_list), np.std(auc_score_list)))
print('AUPR:{:.3f}±{:.3f}'.format(np.mean(aupr_score_list), np.std(aupr_score_list)))
print('MCC:{:.3f}±{:.3f}'.format(np.mean(mcc_score_list), np.std(mcc_score_list)))
print('混淆矩阵:')
for i in range(len(matrix_list)):
    print(matrix_list[i])