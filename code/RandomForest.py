import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# 重新加载数据集
train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")


def fill_missing_values(data):
    data['HomePlanet'].fillna('未知', inplace=True)
    data['Destination'].fillna('TRAPPIST-1e', inplace=True)
    data['CryoSleep'].fillna(False, inplace=True)
    data['VIP'].fillna(False, inplace=True)

    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['RoomService'].fillna(data['RoomService'].mean(), inplace=True)
    data['FoodCourt'].fillna(data['FoodCourt'].mean(), inplace=True)
    data['ShoppingMall'].fillna(data['ShoppingMall'].mean(), inplace=True)
    data['Spa'].fillna(data['Spa'].mean(), inplace=True)
    data['VRDeck'].fillna(data['VRDeck'].mean(), inplace=True)


# 处理数据缺失值
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

class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, max_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.models = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.models = []
        for _ in range(self.n_estimators):
            # 随机选择样本集
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            X_subset = X[sample_indices]
            y_subset = y[sample_indices]

            # 随机选择特征子集
            if self.max_features is not None and self.max_features <= n_features:
                feature_indices = np.random.choice(n_features, self.max_features, replace=False)
                X_subset = X_subset[:, feature_indices]

            # 构建CART决策树并添加到模型列表中
            tree = DecisionTreeClassifier(max_depth=5)
            tree.fit(X_subset, y_subset)  # 添加了fit方法
            self.models.append(tree)

    def predict(self, X):
        # 对所有决策树的预测取平均值或投票
        predictions = np.array([tree.predict(X) for tree in self.models])
        return np.mean(predictions, axis=0)  # 取平均值作为最终预测结果


# 构建随机森林模型
model = RandomForest(n_estimators=20, max_depth=8, max_features=None)

# 训练模型
model.fit(X_train.values, y_train.values)

# 在测试集上进行预测
predictions = model.predict(X_test.values)

# 将预测结果转换为布尔值
predictions_bool = [bool(pred) for pred in predictions]

# 保存预测结果到CSV文件
result_df = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions_bool
})

# 将预测结果保存到CSV文件中
result_df.to_csv('result_rf.csv', index=False)

print("预测完成!")
