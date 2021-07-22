import pickle,os
import sys

with open('All_changed.pkl', 'rb') as fp:
    All_list = pickle.load(fp)
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

ALL_keyword_dic = []

for i in chi_paper_name_WS_list:
    for j in i :
        ALL_keyword_dic.append(j)
for i in chi_keyword_WS_list:
    for j in i :
        ALL_keyword_dic.append(j)
for i in abstract_WS_list:
    for j in i :
        ALL_keyword_dic.append(j)
for i in content_WS_list:
    for j in i :
        ALL_keyword_dic.append(j)
for i in reference_WS_list:
    for j in i :
        ALL_keyword_dic.append(j)

# ALL_keyword_dic = dict() #出現次數統計
tem_dict = {}
for i in ALL_keyword_dic:
    if i not in tem_dict:
        tem_dict[i] = 1
    elif i in tem_dict:
        tem_dict[i] = tem_dict[i] + 1
# print(tem_dict["Aggregate"])
# print(tem_dict)
# print(len(tem_dict))
# print(sorted(tem_dict.values()))
# for i in range(5000,6000):
#     print(f"No.{i}  {sorted(tem_dict.values())[i]}")
count = sorted(tem_dict.items(), key=lambda d: d[1])
# print(count)


with open('count.pkl', 'wb') as fp:
    pickle.dump(count, fp)
fp.close()