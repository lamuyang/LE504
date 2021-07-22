import pickle
with open("reference_WS_TFidt.pkl", 'rb') as fp:
    chi_name = pickle.load(fp)
fp.close()
sorteded = []
for i in chi_name:
    tem_tem = []
    for j in i:
        tem = 0
        for z in j:
            tem += 1
            if tem % 2 == 0:
                tem_tem.append(z[0])
        #     # print(z)
        
        # print("==========")
    # print(f"trtrtrtr:{tem_tem}")
    # print(sorted(tem_tem, reverse = True))
    sorteded.append(sorted(tem_tem, reverse = True))
    # print("------------")
# print(len(sorteded))
# for i in sorteded:
#     print(i)
listlsit = []
for i in chi_name:
    dict = {}
    for j in i:
        # print(j[0][0])
        # print(j[1][0])
        dict[j[0][0]] = j[1][0]
    # print("========")
    listlsit.append(dict)
sortededed = []
for i in listlsit:
    print(i)
    print(sorted(i.items(), key=lambda d: d[1], reverse=True))
    sortededed.append(sorted(i.items(), key=lambda d: d[1], reverse=True))
print(len(listlsit))
print(len(sortededed))
with open("005_reference_sorted.pkl", 'wb') as fp:
    pickle.dump(sortededed, fp)
fp.close()
'''
for i in sorteded:
    print(i)
for j in chi_name:
    for s in j:
        print(s[0][0])
        print(s[1][0])
    break
'''