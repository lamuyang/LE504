import pickle,os
import sys
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

with open('All_changed.pkl', 'rb') as fp:
    All_list = pickle.load(fp)
fp.close()
with open('All_WS_list.pkl', 'rb') as fp:
    All_list_OLD = pickle.load(fp)
fp.close()

chi_paper_name_WS_list = []
chi_keyword_WS_list= []
abstract_WS_list = []
content_WS_list= []
reference_WS_list= []

for i in range(0,len(All_list)):
    chi_paper_name_WS_list.append(All_list[i][0])
    chi_keyword_WS_list.append(All_list[i][1])
    abstract_WS_list.append(All_list[i][2])
    content_WS_list.append(All_list[i][3])
    reference_WS_list.append(All_list[i][4])

chi_paper_name_WS_list2 = []
chi_keyword_WS_list2= []
abstract_WS_list2 = []
content_WS_list2= []
reference_WS_list2= []

for i in All_list_OLD:
    for j in range(len(All_list_OLD[0])):
        chi_paper_name_WS_list2.append(All_list_OLD[0][j])
        chi_keyword_WS_list2.append(All_list_OLD[1][j])
        abstract_WS_list2.append(All_list_OLD[2][j])
        content_WS_list2.append(All_list_OLD[3][j])
        reference_WS_list2.append(All_list_OLD[4][j])
    break
        

def get_blank_list(list):
    tem_list = []
    for i in list:
        tem = ""
        for j in i :
            tem = tem + j +" "
        tem_list.append(tem)
    # print("確認長度：",len(tem_list))
    return tem_list

def get_tf_idf(corpus,name):
    tem_list = []
    vectorizer=CountVectorizer()#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer=TfidfTransformer()#该类会统计每个词语的tf-idf权值
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))#第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word=vectorizer.get_feature_names()#获取词袋模型中的所有词语
    weight=tfidf.toarray()#将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):#打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        # print (u"-------這裡輸出第",i,u"類文本的tf-idf權重------")
        tem_tem = []
        for j in range(len(word)):
            if weight[i][j] != 0.0:
                # print(word[j],weight[i][j])
                tem_tem.append([[word[j]],[weight[i][j]]])
        tem_list.append(tem_tem)
    return tem_list

def add (list2,list):
    for i in list2:
        list.append(i)
    print(f"確認長度：{len(list)}")

'''
add(chi_paper_name_WS_list2,chi_paper_name_WS_list)
chi_paper_name_WS_BLANK = get_blank_list(chi_paper_name_WS_list)
tem_list = get_tf_idf(chi_paper_name_WS_BLANK,"log_chi_paper_name")
print(f"最終長度：{len(tem_list)}")
with open("tem.pkl", 'wb') as fp:
    pickle.dump(tem_list, fp)
fp.close()

print(len(chi_keyword_WS_list))
add(chi_keyword_WS_list2,chi_keyword_WS_list)
chi_keyword_WS_BLANK = get_blank_list(chi_keyword_WS_list)
tem_list = get_tf_idf(chi_keyword_WS_BLANK,"log_chi_keyword_name")
print(f"最終長度：{len(tem_list)}")
with open("tem.pkl", 'wb') as fp:
    pickle.dump(tem_list, fp)
fp.close()

print(len(abstract_WS_list))
add(abstract_WS_list2,abstract_WS_list)
abstract_WS_BLANK = get_blank_list(abstract_WS_list)
tem_list = get_tf_idf(abstract_WS_BLANK,"abstract_keyword_name")
print(f"最終長度：{len(tem_list)}")
with open("tem.pkl", 'wb') as fp:
    pickle.dump(tem_list, fp)
fp.close()

print(len(content_WS_list))
add(content_WS_list2,content_WS_list)
content_WS_BLANK = get_blank_list(content_WS_list)
tem_list = get_tf_idf(content_WS_BLANK,"content_keyword_name")
print(f"最終長度：{len(tem_list)}")
with open("tem.pkl", 'wb') as fp:
    pickle.dump(tem_list, fp)
fp.close()
'''
print(len(reference_WS_list))
add(reference_WS_list2,reference_WS_list)
reference_WS_BLANK = get_blank_list(reference_WS_list)
tem_list = get_tf_idf(reference_WS_BLANK,"reference_keyword_name")
print(f"最終長度：{len(tem_list)}")
with open("tem.pkl", 'wb') as fp:
    pickle.dump(tem_list, fp)
fp.close()