import pickle
with open('All_changed.pkl', 'rb') as fp:
    test = pickle.load(fp)
    print(len(test))
    '''
    for i in range(0,5):
        print(test[3][i])   
        '''

name = []
keyword = []
abstract = []
content = []
reference = []
for i in range(0,len(test)):
    name.append(test[i][0])
    keyword.append(test[i][1])
    abstract.append(test[i][2])
    content.append(test[i][3])
    reference.append(test[i][4])
for i in name:
    print(i)