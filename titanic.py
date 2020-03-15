import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction import DictVectorizer

titanic = pd.read_csv("train.csv")

print(titanic.describe())

#Sex
titanic.loc[titanic["Sex"] == "male","Sex"] = 0
titanic.loc[titanic["Sex"] == "female","Sex"] = 1

print(titanic["Sex"].unique())

#Embarked
# titanic["Embarked"] = titanic["Embarked"].fillna('S')
# titanic.loc[titanic["Embarked"] == "S","Embarked"] = 0
# titanic.loc[titanic["Embarked"] == "C","Embarked"] = 1
# titanic.loc[titanic["Embarked"] == "Q","Embarked"] = 2

#筛选特征值和目标值
x = titanic[{"Pclass","Age","Sex"}]
y = titanic[{"Survived"}]

#数据处理-缺失值处理
x["Age"].fillna(x["Age"].mean(),inplace = True)

#转换成字典
x = x.to_dict(orient = "records")
#print(x)

#数据集划分
x_train, x_test, y_train, y_test = train_test_split(x, y , random_state =22)

#字典特征抽取
transfer = DictVectorizer()
x_train = transfer.fit_transform(x_train)
x_test = transfer.transform(x_test)

#决策树预估器
estimator = DecisionTreeClassifier(criterion = "entropy")
estimator.fit(x_train,y_train)

y_train.to_csv('output.csv')

#模型评估
score = estimator.score(x_test,y_test)
print("准确率为:\n",score)

#可视化决策树
# export_graphviz(estimator,out_file = "Titanic_tree.dot",feature_names = transfer.get_feature_names)

# 计算得准确率为: 0.78
