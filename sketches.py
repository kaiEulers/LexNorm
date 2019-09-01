#%% Test ngram2()
s = 'Gorbachev'
t = 'Gorbechyov'
print(ngram2(s, t))


#%% Test soundex()
sList = ['Loan', 'Loew', 'Lough', 'Lewicks']
sDex = []
for s in sList:
#    sDex.append(soundex(s))
    sDex.append(phonix(s))
print(sDex)


#%% Number of mis-spelt words with n vowels
v = []
vCnt = []
n = 11
for i in range(n):
    v.append([x for x in data if x[2] == i])
    vCnt.append(len(v[i]))

for i in range(len(vCnt)):
    print("Words with " + str(i) + " vowels:\t" + str(vCnt[i]))


#%% Test
nRetRel = 0
nRet = 0
for t in data:
    nRet += t[-3]
    nRetRel += t[-1]

print(nRetRel/nRet)