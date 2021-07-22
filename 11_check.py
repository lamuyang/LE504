from os import name


def open_pkl(pkl):
    import pickle
    with open(f"{pkl}", 'rb') as fp:
        sorted = pickle.load(fp)
    fp.close()
    return sorted
demo = open_pkl("./demo_data/002_demo.pkl")
old_name = open_pkl("./all_pkl/01_paper_name_list.pkl")
old_keyword = open_pkl("./all_pkl/02_keyword_list.pkl")
old_abstract = open_pkl("./all_pkl/03_abstract_list.pkl")
old_content = open_pkl("./all_pkl/04_content_list.pkl")
old_reference = open_pkl("./all_pkl/05_reference_list.pkl")
demo_name = demo[0]
demo_keyword = demo[1]
demo_abstract = demo[2]
demo_content = demo[3]
demo_reference = demo[4]
def get_score(list,old):
    score = 0
    for i in list:
        if i in old:
            print(i)
            score += 1
    print(f"原始長度：{len(list)}")
    print(score)
    # print(score/len(list))
    return score/len(list)
def fin_score(a,b,c,d,e):
    return 3*a+4*b+3*c+d+e
name_sc = get_score(demo_name,old_name)
keyword_sc = get_score(demo_keyword,old_keyword)
abstract_sc = get_score(demo_abstract,old_abstract)
content_sc = get_score(demo_content,old_content)
reference_sc = get_score(demo_reference,old_reference)
print("===========")
print(name_sc*3)
print(keyword_sc*4)
print(abstract_sc*3)
print(content_sc)
print(reference_sc)
print("===========")
print(fin_score(name_sc,keyword_sc,abstract_sc,content_sc,reference_sc))
print(demo)