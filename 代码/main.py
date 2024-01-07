import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector, VarianceThreshold, RFECV, RFE
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, \
    precision_recall_curve, auc, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from collections import Counter
from load_data import X, y, FocalLoss, categorical_feature, cols
import lightgbm as lgb
from scipy import special, optimize
import os
import matplotlib.pylab as plt

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


for table in y.columns:
    # 获取nan的数据
    nan_rows = list(y[table].isna())  # 值为nan的行
    nan_data = X[nan_rows].values
    nan_label = np.ones(shape=(nan_data.shape[0]))
    print('缺失标签数量：', len(nan_label))

    # 获取存在值的数据
    rows = np.isfinite(y[table]).values  # 存在值的行
    X_fin = X[rows].values
    y_fin = y[table][rows].values
    print('已知标签数量：', len(y_fin))

    # 只含正例的数据集
    rows_pos = list(np.where(y_fin == 1)[0])
    X_pos = X_fin[rows_pos, :]
    pos_data = X_pos
    pos_label = np.zeros(shape=(pos_data.shape[0]))

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
    num_0 = list(train_label).count(1)  # 0的个数
    num_1 = list(train_label).count(0)  # 1的个数
    w = num_0 / (num_0 + num_1)  # 不平衡比例
    im = num_0 / num_1

    # 验证集
    val_label = y_val.astype(np.int_)  # 验证的label
    val_data = X_val  # 验证的data

    # 测试集
    test_label = y_test.astype(np.int_)  # 验证的label
    test_label = np.array(test_label).reshape(-1, 1)
    encoder = OneHotEncoder()
    test_label = encoder.fit_transform(test_label).toarray()
    test_data = X_test  # 验证的data

    # # Z-Score标准化
    # standard_scaler = StandardScaler()
    # train_data = standard_scaler.fit_transform(train_data)
    # val_data = standard_scaler.transform(val_data)
    # test_data = standard_scaler.transform(test_data)
    # pos_data = standard_scaler.transform(pos_data)

    # # 特征填充+选择
    # imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # train_data = imputer.fit_transform(train_data)
    # val_data = imputer.transform(val_data)
    # test_data = imputer.transform(test_data)
    # pos_data = imputer.transform(pos_data)
    # print('特征填充完成！')

    # estimator = LinearSVC(dual='auto')  # 由于需要提供特征重要性，所以不能用KNN，这里就使用了LinearSVM
    # selector = RFE(estimator, n_features_to_select=5000).fit(train_data, train_label)
    # selector = SelectFromModel(GradientBoostingClassifier()).fit(train_data, train_label)
    # selector = VarianceThreshold(threshold=0.2).fit(train_data, train_label)  # 方差阈值设为0.2
    # test_data = selector.transform(test_data)
    # selector = RelifF(100, 0.1, 30).fit(train_data, train_label)
    # test_data = selector.transform(test_data)
    # print('特征选择后：', test_data.shape)

    # 求c
    train_pos_data, test_pos_data, train_pos_label, test_pos_label = train_test_split(pos_data, pos_label, train_size=0.8, test_size=0.2)
    # index = np.random.choice(range(len(nan_label)), len(train_label)).tolist()  # 拒绝不平衡
    # nan_data = nan_data[index, :]
    # nan_label = nan_label[index]
    train_c_data = np.vstack((train_pos_data, nan_data))  # 仅含正例的数据加上无标签的数据
    train_c_label = np.append(train_pos_label, nan_label)
    train_C = lgb.Dataset(train_c_data, train_c_label, feature_name=cols, categorical_feature=categorical_feature)
    param = {'verbose': -1, 'learning_rate': 0.01, 'metric': ['auc', 'binary_logloss'], 'objective': 'binary', 'num_threads': 4}  # 设置参数
    model = lgb.train(
        param,
        train_set=train_C,
        num_boost_round=350,  # 迭代次数
        feature_name=cols,
        categorical_feature=categorical_feature
    )  # 训练模型
    y_pred = model.predict(test_pos_data, categorical_feature=categorical_feature)  # 在仅含正例的数据上进行预测
    c = 0
    i = 0
    for pro in y_pred:
        c += pro
        i += 1
    c = c / i
    c = 1 - c
    print('c:', c)

    # LightGBM
    print('im:', im)
    if im > 10:  # 不平衡比例大于10
        fl = FocalLoss(alpha=w, gamma=1.7)
    else:
        fl = FocalLoss(alpha=w, gamma=0.5)

    train = lgb.Dataset(train_data, train_label,
                        init_score=np.full_like(train_label, fl.init_score(train_label), dtype=float), feature_name=cols, categorical_feature=categorical_feature)
    val = lgb.Dataset(val_data, val_label, reference=train,
                      init_score=np.full_like(val_label, fl.init_score(val_label), dtype=float), feature_name=cols, categorical_feature=categorical_feature)

    param = {'learning_rate': 0.01, 'metric': ['auc'], 'num_threads': 4, 'early_stopping_round': 150}  # 设置参数
    model = lgb.train(
        param,
        train_set=train,
        num_boost_round=350,  # 迭代次数
        valid_sets=[train, val],
        fobj=fl.lgb_obj,
        feval=fl.lgb_eval,
        feature_name=cols,
        categorical_feature=categorical_feature
    )  # 训练模型

    # train = lgb.Dataset(train_data, train_label, feature_name=cols, categorical_feature=categorical_feature)
    # val = lgb.Dataset(val_data, val_label, reference=train, feature_name=cols, categorical_feature=categorical_feature)
    # param = {'verbose': -1, 'learning_rate': 0.01, 'metric': ['auc', 'binary_logloss'], 'objective': 'binary', 'num_threads': 4}  # 设置参数
    # model = lgb.train(
    #     param,
    #     train_set=train,
    #     num_boost_round=350,  # 迭代次数
    #     valid_sets=[train, val],
    #     feature_name=cols,
    #     categorical_feature=categorical_feature
    # )  # 训练模型
    bias = special.expit(fl.init_score(train_label))

    # 预测
    y_pred = bias + model.predict(test_data, categorical_feature=categorical_feature)
    # y_pred = model.predict(test_data, categorical_feature=categorical_feature)
    new_y_pred = [[(1-p)/c, 1-((1-p)/c)] for p in y_pred]
    # new_y_pred = [[(1 - p), p] for p in y_pred]
    y_pred = np.array(new_y_pred)
    min_max_scaler = preprocessing.MinMaxScaler()
    y_pred = min_max_scaler.fit_transform(y_pred)
    print(y_pred)  # 归一化

    # 评估指标
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

    # 特征重要性+树
    plt.figure(figsize=(12, 6))
    lgb.plot_importance(model, max_num_features=30)
    plt.title("Featurertances")

    lgb.plot_tree(model, tree_index=3, figsize=(20, 8), show_info=['split_gain'])  # 画第三棵树
    plt.show()

    print('导出决策树的pdf图像到本地')  # 这里需要安装graphviz应用程序和python安装包
    graph = lgb.create_tree_digraph(model, tree_index=3, name='Tree3')
    graph.render(view=True)

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