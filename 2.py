from pymongo import MongoClient
import re
import pickle
import get_data
allfields = get_data.get_mongodb_row("LINS")
print(len(allfields[0]))

allfields_list = []
for i in allfields:
    allfields_list.append(i[0])
    allfields_list.append(i[1])
    allfields_list.append(i[2])
    allfields_list.append(i[3])
    allfields_list.append(i[4])


allfields_Final = []
all_new_field_list = []


for row in allfields_list:
    a = re.findall('「([a-zA-Z\s]*?)」', row)
    b = re.findall('「([\u4e00-\u9fa5]*?)」', row)

    c = re.findall('『([a-zA-Z\s]*?)』', row)
    d = re.findall('『([\u4e00-\u9fa5]*?)』', row)

    e = re.findall('《([a-zA-Z\s]*?)》', row)
    f = re.findall('《([\u4e00-\u9fa5]*?)》', row)

    g = re.findall('"([a-zA-Z\s]*?)"', row)
    h = re.findall('"([\u4e00-\u9fa5]*?)"', row)

    i = re.findall('【([a-zA-Z\s]*?)】', row)
    j = re.findall('【([\u4e00-\u9fa5]*?)】', row)

    k = re.findall('（([a-zA-Z\s]*?)）', row)
    l = re.findall('（([\u4e00-\u9fa5]*?)）', row)

    m = re.findall('〔([\u4e00-\u9fa5]*?)〕', row)

    n = re.findall('〝([\u4e00-\u9fa5]*?)〞', row)

    o = re.findall('\[([a-zA-Z\s]*?)\]', row)
    p = re.findall('\[([\u4e00-\u9fa5]*?)\]', row)

    keywords_list = a + b + c + d + e + f + g + h + i + j + k + l + m + n + o + p
    # print(keywords_list)
    allfields_Final.append((row, keywords_list))
    # print(allfields_Final)

with open("allfields_plus_keywords.txt", 'w', encoding='utf-8') as f:
	index = 0
	for i in allfields_Final:
		index += 1
		row = i[0]
		keyword = i[1]
		f.write('{} : {}\n{}\n\n'.format(index, row, keyword))

# 去除重複 keyword
keyword_dict = dict() #出現次數統計
for i in allfields_Final:
	k_list = i[1]
	if k_list == []:
		pass
	else:
		for k in k_list:
			if len(k) == 1:
				pass
			elif k in keyword_dict:
				keyword_dict[k] += 1
			else:
				keyword_dict[k] = 1
# print(keyword_dict)

with open("allfields_keywords_unique.txt", 'w', encoding='utf-8') as f:
	index = 0
	for k in keyword_dict:
		index += 1
		f.write('{}\t【{}】 ： {} 次數\n'.format(index, k, keyword_dict[k]))


with open("corpus_wikidict_PAGETITLE.pkl", 'rb') as fp:
    wiki_dict = pickle.load(fp)
fp.close()

with open("corpus_only_allfieldskeywordsDict.pkl", 'wb') as fp:
    pickle.dump(wiki_dict, fp)
fp.close()


wiki_dict.update(keyword_dict)
with open("WikiDict_plus_allfieldskeywordsDict.pkl", 'wb') as fp:
    pickle.dump(wiki_dict, fp)
fp.close()

