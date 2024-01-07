import pandas as pd
from scipy import special, optimize
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

dir = './data/'
df = pd.read_csv(dir + 'AVALON.csv')  # 分子指纹
X_finger = df.iloc[:, 13::]  # 分子指纹特征
df = pd.read_csv(dir + 'mordred.csv')  # 分子描述符
X_continue = df.iloc[:, 13::]  # 分子描述符特征
y = df.iloc[:, 0:12]  # 标签

data_files = ['ECFP2.csv', 'ECFP4.csv', 'ECFP6.csv', 'MACCS.csv', 'pubchem.csv', 'rdkit2d.csv']
for data_file in data_files:
    df = pd.read_csv(dir + data_file)
    X_tmp = df.iloc[:, 13::]  # 分子特征
    if data_file != 'rdkit2d.csv':  # 分子指纹特征
        X_finger = pd.concat([X_finger, X_tmp], axis=1)  # 拼接
    else:
        X_continue = pd.concat([X_continue, X_tmp], axis=1)  # 拼接

print(X_continue.shape)
new_X_continue = X_continue.dropna(axis='columns', thresh=int(0.1*X_continue.shape[0]))  # 丢掉nan占比多于0.9的特征
temp_cols = new_X_continue.columns
# 特征填充+选择
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
new_X_continue = imputer.fit_transform(new_X_continue)
print('特征填充完成！')

standard_scaler = StandardScaler()
X_continute = standard_scaler.fit_transform(new_X_continue) # Z-Score标准化
X = pd.concat([pd.DataFrame(new_X_continue, columns=temp_cols), X_finger], axis=1)  # 将连续型数据和离散型数据拼接起来

categorical_feature = X_finger.columns.tolist()
cols = X.columns.tolist()

class FocalLoss:
    def __init__(self, gamma, alpha=None):
        # 使用FocalLoss只需要设定以上两个参数,如果alpha=None,默认取值为1
        self.alpha = alpha
        self.gamma = gamma

    def at(self, y):
        # alpha 参数, 根据FL的定义函数,正样本权重为self.alpha,负样本权重为1 - self.alpha
        if self.alpha is None:
            return np.ones_like(y)
        return np.where(y, self.alpha, 1 - self.alpha)

    def pt(self, y, p):
        # pt和p的关系
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return np.where(y, p, 1 - p)

    def __call__(self, y_true, y_pred):
        # 即FL的计算公式
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        return -at * (1 - pt) ** self.gamma * np.log(pt)

    def grad(self, y_true, y_pred):
        # 一阶导数
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma
        return at * y * (1 - pt) ** g * (g * pt * np.log(pt) + pt - 1)

    def hess(self, y_true, y_pred):
        # 二阶导数
        y = 2 * y_true - 1  # {0, 1} -> {-1, 1}
        at = self.at(y_true)
        pt = self.pt(y_true, y_pred)
        g = self.gamma

        u = at * y * (1 - pt) ** g
        du = -at * y * g * (1 - pt) ** (g - 1)
        v = g * pt * np.log(pt) + pt - 1
        dv = g * np.log(pt) + g + 1

        return (du * v + u * dv) * y * (pt * (1 - pt))

    def init_score(self, y_true):
        # 样本初始值寻找过程
        res = optimize.minimize_scalar(
            lambda p: self(y_true, p).sum(),
            bounds=(0, 1),
            method='bounded'
        )
        p = res.x
        log_odds = np.log(p / (1 - p))
        return log_odds

    def lgb_obj(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)  # sigmoid
        return self.grad(y, p), self.hess(y, p)

    def lgb_eval(self, preds, train_data):
        y = train_data.get_label()
        p = special.expit(preds)
        is_higher_better = False
        return 'focal_loss', self(y, p).mean(), is_higher_better

