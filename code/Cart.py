import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 重新加载数据集
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")


def fill_missing_values(data):
    data['HomePlanet'].fillna('Uknow', inplace=True)
    data['Destination'].fillna('TRAPPIST-1e', inplace=True)
    data['CryoSleep'].fillna(False, inplace=True)
    data['VIP'].fillna(False, inplace=True)

    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['RoomService'].fillna(data['RoomService'].mean(), inplace=True)
    data['FoodCourt'].fillna(data['FoodCourt'].mean(), inplace=True)
    data['ShoppingMall'].fillna(data['ShoppingMall'].mean(), inplace=True)
    data['Spa'].fillna(data['Spa'].mean(), inplace=True)
    data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=True)


# 处理数据缺省值
fill_missing_values(train_data)
fill_missing_values(test_data)

# 数据预处理 - 训练集
y_train = train_data['Transported']
x_train = train_data.drop(['PassengerId', 'Name', 'Transported'], axis=1)
x_test = test_data.drop(['PassengerId', 'Name'], axis=1)

# 区分数值列和类别列
numeric_columns_train = x_train.select_dtypes(include=['number'])  # 数值列
categorical_columns_train = x_train.select_dtypes(exclude=['number'])  # 类别列
categorical_columns_train_c = categorical_columns_train.copy()

numeric_columns_test = x_test.select_dtypes(include=['number'])  # 数值列
categorical_columns_test = x_test.select_dtypes(exclude=['number'])  # 类别列
categorical_columns_test_c = categorical_columns_test.copy()

# 类别列合并
combined_data = pd.concat([categorical_columns_train, categorical_columns_test], axis=0)
combined_data_c = combined_data.copy()

# 处理类别列：用数字表示
# 创建LabelEncoder对象
label_encoders = {}
for column in combined_data:
    label_encoders[column] = LabelEncoder()
    combined_data_c[column] = label_encoders[column].fit_transform(combined_data[column])

# 处理类别列：使用之前的 LabelEncoder 对象
# 对每一列使用LabelEncoder进行转换
for column in categorical_columns_train.columns:
    categorical_columns_train_c[column] = label_encoders[column].transform(categorical_columns_train[column])

# 合并处理后的数值列和类别列得到新的 X_train
X_train = pd.concat([numeric_columns_train, categorical_columns_train_c], axis=1)

# 对每一列使用LabelEncoder进行转换
for column in categorical_columns_test.columns:
    categorical_columns_test_c[column] = label_encoders[column].transform(categorical_columns_test[column])

# 合并处理后的数值列和类别列得到新的 X_test
X_test = pd.concat([numeric_columns_test, categorical_columns_test_c], axis=1)


# print(X_test.dtypes)

class DecisionNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        # 对于内部节点
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain

        # 对于叶节点
        self.value = value


# 计算Gini指数
def calculate_gini_index(y_subsets):
    gini = 0
    total_samples = sum(len(y) for y in y_subsets)
    for y in y_subsets:
        p = len(y) / total_samples
        score = p * (1 - p)
        gini += score
    return gini


def split_dataset(X, y, feature_index, threshold):
    """根据特征索引和阈值分割数据集"""
    left_idx = np.where(X[:, feature_index] <= threshold)
    right_idx = np.where(X[:, feature_index] > threshold)

    return X[left_idx], X[right_idx], y[left_idx], y[right_idx]


# 获取最佳分割点
def get_best_split(X, y, n_features):
    best_feature, best_threshold, best_info_gain, best_gini = None, None, -1, 999
    for feature_index in range(n_features):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            _, _, y_left, y_right = split_dataset(X, y, feature_index, threshold)
            if len(y_left) > 0 and len(y_right) > 0:
                gini = calculate_gini_index([y_left, y_right])
                info_gain = best_gini - gini
                if info_gain > best_info_gain:
                    best_gini, best_info_gain = gini, info_gain
                    best_feature, best_threshold = feature_index, threshold
    return best_feature, best_threshold, best_info_gain


# 构建决策树
def build_tree(X, y, n_features, max_depth, min_samples_split, depth=0):
    n_samples, n_features = X.shape
    if n_samples >= min_samples_split and depth <= max_depth:
        best_feature, best_threshold, best_info_gain = get_best_split(X, y, n_features)
        if best_info_gain > 0:
            left_X, right_X, left_y, right_y = split_dataset(X, y, best_feature, best_threshold)
            left_child = build_tree(left_X, left_y, n_features, max_depth, min_samples_split, depth + 1)
            right_child = build_tree(right_X, right_y, n_features, max_depth, min_samples_split, depth + 1)
            return DecisionNode(feature_index=best_feature, threshold=best_threshold, left=left_child,
                                right=right_child, info_gain=best_info_gain)
    return DecisionNode(value=np.bincount(y).argmax())


def predict(tree, x):
    """单个样本的预测"""
    if tree.value is not None:  # 叶节点
        return tree.value
    feature_val = x[tree.feature_index]
    if feature_val <= tree.threshold:
        return predict(tree.left, x)
    else:
        return predict(tree.right, x)


# 构建决策树
def Cart():
    n_features = X_train.shape[1]
    max_depth = 8
    min_samples_split = 2
    return build_tree(X_train.values, y_train.values, n_features, max_depth, min_samples_split)


# 对测试数据进行预测
model = Cart()
predictions = [predict(model, x) for x in X_test.values]

# 将预测结果转换为布尔值
predictions_bool = [bool(pred) for pred in predictions]

# 保存预测结果到CSV文件
result_df = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions_bool
})

# 将预测结果保存到CSV文件中
result_df.to_csv('result_cart.csv', index=False)

print("预测完成!")
