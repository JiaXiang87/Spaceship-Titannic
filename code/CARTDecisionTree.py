import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import numpy as np

# 载入数据集
train_data = pd.read_csv('../data/train.csv')
test_data = pd.read_csv('../data/test.csv')

def convert_cabin_value(test_data, train_data):
    for idx, value in enumerate(test_data['Cabin']):
        if isinstance(value, float):  # Check if the value is a float
            # Handle NaN values or other non-string types here
            # For example, you can set a default value or do something else
            # For now, I'll just continue to the next value
            continue

        if value not in train_data['Cabin'].values:
            deck, num, side = value.split('/')
            num = int(num)

            train_cabins = set(train_data['Cabin'])
            transformed_value = value  # 初始化 transformed_value
            step = 1  # 初始化步长为 1
            flag = 1

            while transformed_value not in train_cabins:
                # Try transforming side
                if side == 'S':
                    side = 'P'
                else:
                    side = 'S'

                num += step*flag  # 更新 num
                step += 1
                flag *= -1

                print(f"尝试: {transformed_value}")  # 添加此行以调试
                transformed_value = f"{deck}/{num}/{side}"
                if transformed_value in train_cabins:
                    test_data.at[idx, 'Cabin'] = transformed_value
                    break

convert_cabin_value(test_data, train_data)

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


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class CARTDecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None


    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _calculate_gini(self, y):
        # 计算基尼不纯度
        classes = np.unique(y)
        gini = 0
        total_samples = len(y)

        for cls in classes:
            p = np.sum(y == cls) / total_samples
            gini += p * (1 - p)

        return gini

    # 在 _best_split 方法中修改 X 切片操作的方式
    def _best_split(self, X, y):
        best_gini = float('inf')
        best_criteria = None
        n_features = X.shape[1]

        for feature_index in range(n_features):
            thresholds = np.unique(X.iloc[:, feature_index])  # 使用 iloc 获取特定列

            for threshold in thresholds:
                mask = X.iloc[:, feature_index] <= threshold  # 使用 iloc 获取特定列
                y_left = y[mask]
                y_right = y[~mask]

                if len(y_left) > 0 and len(y_right) > 0:
                    gini = self._calculate_gini(y_left) + self._calculate_gini(y_right)
                    if gini < best_gini:
                        best_gini = gini
                        best_criteria = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'left_mask': mask
                        }

        return best_criteria

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # 停止条件
        if (self.max_depth is not None and depth >= self.max_depth) or n_classes == 1 or n_samples <= 1:
            value = np.bincount(y).argmax()
            return Node(value=value)

        best_criteria = self._best_split(X, y)
        if best_criteria is not None:
            left_mask = best_criteria['left_mask']
            feature_index = best_criteria['feature_index']
            threshold = best_criteria['threshold']

            left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
            right = self._grow_tree(X[~left_mask], y[~left_mask], depth + 1)
            return Node(feature_index=feature_index, threshold=threshold, left=left, right=right)

    def predict(self, X):
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, x, node):
        if node.value is not None:
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)


model = CARTDecisionTree(max_depth=5, min_samples_split=2)
model.fit(X_train, y_train)

# 对测试集进行预测
y_pred = model.predict(X_test)

# 创建一个DataFrame保存预测结果
result_df = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Transported': y_pred})

# 将预测结果保存到CSV文件中
result_df.to_csv('result_cart.csv', index=False)
