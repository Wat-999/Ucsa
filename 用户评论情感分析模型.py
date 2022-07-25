# 1.数据读取
import pandas as pd
df = pd.read_excel('产品评价.xlsx')
df.head()  #1代表好评,0代表差评


#2.中文分词
import jieba
words = []
for i, row in df.iterrows():
    words.append(' '.join(jieba.cut(row['评论'])))


# 3.构造特征变量和目标变量
#3.1构造特征变量
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
X = vect.fit_transform(words)  #文本向量化
X = X.toarray()      
print(X)

#查看词袋
words_bag = vect.vocabulary_  #
print(words_bag)
print(len(words_bag))

#查看词频矩阵
import pandas as pd
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.DataFrame(X).head()

# 3.2目标变量提取
y = df['评价']
y.head()



#4.神经网络模型的搭建与使用
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)


#搭建模型
mlp =MLPClassifier()  )
mlp.fit(X_train, y_train)

#模型使用
#对测试集数据进行预测
y_pred = mlp.predict(X_test)
print(y_pred)  


#汇总预测值和实际值
a = pd.DataFrame()  # 创建一个空DataFrame
a['预测值'] = list(y_pred)
a['实际值'] = list(y_test)
a.head()

#查看所有测试集数据对预测准确度
score = accuracy_score(y_pred, y_test)
print(score)
#print(mlp.score(X_test, y_test))  #效果同上(此函数为神经网络模型自带)


#输入待分类对评价
comment = input('请输入您对本商品的评价：')
comment = [' '.join(jieba.cut(comment))]
print(comment)
X_try = vect.transform(comment)
y_pred = mlp.predict(X_try.toarray())
print(y_pred)

#模型对比（朴素贝叶斯模型）
from sklearn.naive_bayes import GaussianNB
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)
print(y_pred)

from sklearn.metrics import accuracy_score
score = accuracy_score(y_pred, y_test)
print(score)


