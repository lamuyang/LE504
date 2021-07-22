import pickle

def open_pkl(pkl):
    with open(f"{pkl}", 'rb') as fp:
        sorted = pickle.load(fp)
    fp.close()
    return sorted
def get_over_score(list,score):
    list_all_word = []
    stop_word = ["url","html",'http',"www","edu","00年","wwwukoln",'pp',"第一","第二","第三","第四","第五","第六","不詳"]
    for j in list:
        tem = []
        for i in range(0,len(j)):
            if j[i][1] >= score and j[i][0] not in stop_word:
                tem.append(j[i])
        # for i in tem:
        #     print(i)
        for i in tem:
            list_all_word.append(i[0])
        # print("==========")
    print(f"長度是：{len(list_all_word)}")
    return list_all_word

paper_name = open_pkl("001_paper_name_sorted.pkl")
keyword = open_pkl("002_chi_keyword_sorted.pkl")
abstract = open_pkl("003_abstract_sorted.pkl")
content = open_pkl("004_content_sorted.pkl")
reference = open_pkl("005_reference_sorted.pkl")

paper_name_list = get_over_score(paper_name,0.15)
keyword_list = get_over_score(keyword,0.15)
abstract_list = get_over_score(abstract,0.15)
content_list = get_over_score(content,0.15)
reference_list = get_over_score(reference,0.15)

with open("./all_pkl/01_paper_name_list.pkl", 'wb') as fp:
    pickle.dump(paper_name_list, fp)
fp.close()
with open("./all_pkl/02_keyword_list.pkl", 'wb') as fp:
    pickle.dump(keyword_list, fp)
fp.close()
with open("./all_pkl/03_abstract_list.pkl", 'wb') as fp:
    pickle.dump(abstract_list, fp)
fp.close()
with open("./all_pkl/04_content_list.pkl", 'wb') as fp:
    pickle.dump(content_list, fp)
fp.close()
with open("./all_pkl/05_reference_list.pkl", 'wb') as fp:
    pickle.dump(reference_list, fp)
fp.close()
