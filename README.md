# Spaceship Titanic

这是一个基于 Kaggle「Spaceship Titanic」数据集的课程/练习项目。项目尝试使用支持向量机、决策树、随机森林和 AdaBoost 等分类方法，预测乘客是否被传送（`Transported`）。仓库同时保留了原始数据、若干预测结果 CSV 和课程报告。

> 目前 `code/` 中的脚本是独立的实验脚本，不是统一封装的命令行程序。不同脚本的数据预处理和模型实现有所差异，运行结果也可能与 `result/` 中保存的历史结果不同。

## 目录结构

```text
.
├── code/       # 独立的模型训练和预测脚本
├── data/       # train.csv、test.csv 和 sample_submission.csv
├── doc/        # 课程/实验报告（DOCX）
├── result/     # 已保存的模型预测结果（CSV）
└── README.md
```

## 文件说明

| 文件 | 说明 |
| --- | --- |
| `code/SVM.py`、`code/SVM2.py`、`code/1.py` | SVM 相关实验版本；`1.py` 也会输出名为 `result_svm.csv` 的文件。 |
| `code/Cart.py` | 手写 CART 决策树实现。 |
| `code/CARTDecisionTree.py` | 另一份 CART 决策树实现。 |
| `code/RandomForest.py` | 手写随机森林实验，基学习器使用 scikit-learn 决策树。 |
| `code/MyEL.py` | 使用 scikit-learn `RandomForestClassifier` 的随机森林实验。 |
| `code/AdaBoost.py` | 自定义 AdaBoost 实验，基学习器为决策树。 |
| `data/train.csv` | 带 `Transported` 标签的训练数据。 |
| `data/test.csv` | 待预测数据，不包含 `Transported` 标签。 |
| `data/sample_submission.csv` | 提交格式示例。 |
| `result/*.csv` | 已保存的预测文件，通常包含 `PassengerId` 和 `Transported` 两列。 |
| `doc/*.docx` | 项目相关课程和实验文档。 |

## 环境与运行

脚本使用 Python，以及 `pandas`、`numpy` 和 `scikit-learn`。请在项目根目录创建并启用虚拟环境后安装依赖：

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS / Linux 可使用：source .venv/bin/activate
python -m pip install pandas numpy scikit-learn
```

当前脚本通过相对路径 `../data/train.csv` 和 `../data/test.csv` 读取数据，因此应从 `code` 目录运行。例如：

```bash
cd code
python MyEL.py
```

将 `MyEL.py` 替换为其他脚本文件名即可尝试对应实验。脚本会在**当前工作目录**写出预测 CSV；按上述方式运行时，输出会落在 `code/` 中，而不是 `result/`。若要保存到 `result/`，运行后请自行移动文件，或修改脚本中的输出路径。

## 当前脚本的注意事项

- 多个脚本会读取数据、训练模型并立即生成预测，导入脚本也会触发这些操作。
- 多个脚本使用同一个输出文件名。例如 `Cart.py` 和 `CARTDecisionTree.py` 都写入 `result_cart.csv`；`SVM.py`、`SVM2.py` 和 `1.py` 都写入 `result_svm.csv`。运行时可能覆盖之前生成的同名文件。
- 预处理逻辑并不完全一致。类别特征主要通过 `LabelEncoder` 转换，缺失值处理也存在差异；部分版本对训练集和测试集类别的编码方式不同。因此，这些脚本适合视作不同实验版本，不保证所有脚本都能在当前数据上无修改运行。
- 手写随机森林中的随机抽样没有固定随机种子，重复运行可能得到不同结果。
- `result/` 中是已经保存的输出文件，不代表由当前脚本在当前环境下重新运行得到的结果。

## 数据与报告

`data/` 中包含 Spaceship Titanic 训练集、测试集和提交示例。公开仓库前，请在此处补充数据集原始来源链接及其许可/使用条件，并确认报告中没有不适合公开的个人或课程信息。

## 后续整理建议

为提升项目的可复现性，可以进一步统一数据路径和输出目录、为脚本设置独立且清晰的输出文件名、整理依赖到 `requirements.txt`，并增加验证集评估与运行参数说明。
