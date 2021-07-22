import pickle,os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

with open('All_WS_list.pkl', 'rb') as fp:
    All_list = pickle.load(fp)
fp.close()

chi_paper_name_WS_list = []
chi_keyword_WS_list= []
abstract_WS_list = []
content_WS_list= []
reference_WS_list= []

for i in range(0,5):
    if i == 0:
        for j in All_list[i]:
            chi_paper_name_WS_list.append(j)
    if i == 1:
        for j in All_list[i]:
            chi_keyword_WS_list.append(j)
    if i == 2:
        for j in All_list[i]:
            abstract_WS_list.append(j)
    if i == 3:
        for j in All_list[i]:
            content_WS_list.append(j)
    if i == 4:
        for j in All_list[i]:
            reference_WS_list.append(j)


def get_blank_list(list):
    tem_list = []
    for i in list:
        tem = ""
        for j in i :
            tem = tem + j +" "
        tem_list.append(tem)
    print("確認長度：",len(tem_list))
    return tem_list

def get_tf_idf(corpus,name):
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        print (u"-------這裡輸出第",i,u"類文本的tf-idf權重------")
        # with open(f'{name}.txt','a') as file:
        #     file.write(f"-------這裡輸出第{i}類文本的tf-idf權重------\n")
        #     file.write("")
        for j in range(len(word)):
            if weight[i][j] != 0.0:
                print(word[j],weight[i][j])
                # with open(f'{name}.txt','a') as file:
                #     file.write(f'{word[j]},{weight[i][j]}\n')