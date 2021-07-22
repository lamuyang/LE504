import pickle
with open('tem.pkl', 'rb') as fp:
    tem_list = pickle.load(fp)
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

# print(len(chi_keyword_WS_list))
# print(len(tem_list))
print(tem_list[14421])
temtem = []
for i in range(0,237):
    temtem.append(tem_list[i])
# for i in name:
#     print(name)

with open("reference_WS_TFidt.pkl", 'wb') as fp:
    pickle.dump(temtem, fp)
fp.close()