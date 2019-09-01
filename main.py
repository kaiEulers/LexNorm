#%% Knowlegde Technologies COMP90049
# Lexical Normalisation
import os
import pickle
import time
import numpy as np
import matplotlib.pyplot as plt
os.system("clear")

# =============================================================================
# Soundex Function
# =============================================================================
def soundex(w):
    """
    soundex(s) returns a tuple (w, sc, nV )
    w is the original word
    sc is the soundex of the word
    nV is the number of vowels in the word
    """
    nV = 0 
    sc_raw = [w[0]]
    # Create soundex
    for c in w[1:]:
        if c in "aeiouhwy":
            sc_raw.append(0)
            nV += 1
        elif c in "bfpv":
            sc_raw.append(1)
        elif c in "cgjkqsxz":
            sc_raw.append(2)
        elif c in "dt":
            sc_raw.append(3)
        elif c in "l":
            sc_raw.append(4)
        elif c in "mn":
            sc_raw.append(5)
        elif c in "r":
            sc_raw.append(6)
    # Remove duplicate terms
    sc = []
    for i in sc_raw:
        if i not in sc:
            sc.append(i)
    # Remove zeros        
    for i in range(sc.count(0)):
        sc.remove(0)    
    # Truncate all terms after the 4th term
    if len(sc) > 4:
        sc = sc[:4]
        
    return (w, tuple(sc), nV)
    
# =============================================================================
# Phonix+ Function
# =============================================================================
def phonix(w):
    """
    phonix(s) returns a tuple (w, sc, nV )
    w is the original word
    sc is the soundex of the word
    nV is the number of vowels in the word
    """
    nV = 0
    phnx_raw = [w[0]];
    # Create soundex
    for c in w[1:]:
        if c in "aeiouhwy":
            phnx_raw.append(0);
            nV += 1
        elif c in "bp":
            phnx_raw.append(1);
        elif c in "cjkq":
            phnx_raw.append(2);
        elif c in "dt":
            phnx_raw.append(3);
        elif c in "l":
            phnx_raw.append(4);
        elif c in "mn":
            phnx_raw.append(5);
        elif c in "r":
            phnx_raw.append(6);
        elif c in "fv":
            phnx_raw.append(7);
        elif c in "sxz":
            phnx_raw.append(8);
    # Remove duplicate terms
    phnx = [];
    for i in phnx_raw:
        if i not in phnx:
            phnx.append(i);
    # Remove zeros        
    for i in range(phnx.count(0)):
        phnx.remove(0)
        
    return (w, tuple(phnx), nV)

# =============================================================================
# ngram Function
#  =============================================================================
def ngram2(tok, dic):
    """ngram(string, string) computes N-Gram Distance with n = 2"""
    sub_tok = [];
    sub_dic = [];    
    # Create sub-strings list with n = 2
    for i in range(len(tok) - 1):
        sub_tok.append((tok[i], tok[i+1]))
    for i in range(len(dic) - 1):
        sub_dic.append((dic[i], dic[i + 1]))     
    # Create list with common terms in both sub-strings list
    sub_cmn = tuple(set(sub_tok) & set(sub_dic))
    # Compute N-Gram    
    nGram = len(sub_tok) + len(sub_dic) - 2*len(sub_cmn)
    
    return nGram

# =============================================================================
# # Edit Distance Function
# =============================================================================
def match(a, b):
    """
    match(a, b) returns 0 if a matches b
                returns 1 if a doesn't match b
    """
    if a == b:
        return 0
    else:
        return 1
        
def editD(tok, dic):
    """
    editD(tok, dic) returns the minimum edit distance between tuples tok and dic
    """
    A = np.zeros([len(dic), len(tok)])
    for col in range(len(tok)):
        A[0, col] = col
    for row in range(len(dic)):
        A[row, 0] = row
    
    for row in range(len(dic)):
        for col in range(len(tok)):
            A[row, col] = min(
                    A[row-1, col] + 1,
                    A[row, col-1] + 1,
                    A[row-1, col-1] + match(dic[row], tok[col]),
                    )
    return int(A[len(dic)-1][len(tok)-1])

# =============================================================================
# Open Files
# =============================================================================
# Open files and store text into lists
path_msplTxt = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/raw/misspell.txt"
path_crtTxt = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/raw/correct.txt"
path_dictTxt = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/raw/dict.txt"
with open(path_msplTxt) as msplFile:
    mspl = [line.strip() for line in msplFile];
with open(path_crtTxt) as crtFile:
    crt = [line.strip() for line in crtFile];
with open(path_dictTxt) as dictFile:
    dic = [line.strip() for line in dictFile];

# Convert all words in dictionary into soundex
dictSdex = [soundex(w) for w in dic];
dictPhnx = [phonix(w) for w in dic];


# Save all data
path_mspl = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/mspl.dat"
pickle.dump(mspl, open(path_mspl, "wb"))

path_crt = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/crt.dat"
pickle.dump(crt, open(path_crt, "wb"))

path_dic = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/dic.dat"
pickle.dump(dic, open(path_dic, "wb"))

path_dictSdex = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/dictSdex.dat"
pickle.dump(dictSdex, open(path_dictSdex, "wb"))

path_dictPhnx = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/data/dictPhnx.dat"
pickle.dump(dictPhnx, open(path_dictPhnx, "wb"))


#%%
# =============================================================================
# Implement Soundex for string matching
# Returns tuple data containing (w, sDex, vN, nRet, nRel)
#    w is the original word
#    sDex is the soundex of the word
#    nV is the # of vowels in the word
#    nRet is the total # of returned results
#    nRel is the total # of relevant results 
# =============================================================================
os.system("clear")
mspl = pickle.load(open("data/mspl.dat", "rb"))
crt = pickle.load(open("data/crt.dat", "rb"))
dic = pickle.load(open("data/dic.dat", "rb"))
dictSdex = pickle.load(open("data/dictSdex.dat", "rb"))


dataSdex = []

timeStart = time.time()
# Traverse thru possibly mis-spelt words
N = len(mspl)
#N = 10
for i in range(N):
#    Convert mis-spelt word to soundex
    msplData = soundex(mspl[i]) 
    
#    Search thru dictionary's soundex for matching mis-spelt soundex
    ret = [x for x in dictSdex if msplData[1] == x[1]]
    nRet = len(ret)
#    Track returned words
#    print("Mis-spelt: " + mspl[i] + " Correct: " + crt[i])
#    print([w[0] for w in ret])
    
#    Are any of the returned results that are relevant?
#    Find returned term that matches with the correct(relevant) word
    retRel = [r for r in ret if r[0] == crt[i]]
    nRetRel = len(retRel)
    
#    Find if there are relevant data in the dictionary for the mis-spelt word
    rel = [x for x in dictSdex if x[0] == crt[i]]
    nRel = len(rel)
    
    dataSdex.append(msplData + (nRet, nRel, nRetRel))
    
#    Track progess of program
    print(str(i+1) + "/" + str(N))
    
timeEnd = time.time()
pTime_sDex = (timeEnd - timeStart)

# Save results for Soundex
path_sDex = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_sDex.dat"
pickle.dump(dataSdex, open(path_sDex, "wb"))

path_pTime_sDex = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/pTime_phx.dat"
pickle.dump(pTime_sDex, open(path_pTime_sDex, "wb"))

os.system('say "Complete"')


#%%
# =============================================================================
# Implement Phonix+ for string matching
# Returns tuple data containing (w, sDex, vN, nRet, nRel)
#    w is the original word
#    sDex is the soundex of the word
#    nV is the # of vowels in the word
#    nRet is the total # of returned results
#    nRel is the total # of relevant results 
# =============================================================================
os.system("clear")
mspl = pickle.load(open("data/mspl.dat", "rb"))
crt = pickle.load(open("data/crt.dat", "rb"))
dic = pickle.load(open("data/dic.dat", "rb"))
dictPhnx = pickle.load(open("data/dictPhnx.dat", "rb"))


dataPhnx = []

timeStart = time.time()
# Traverse thru possibly mis-spelt words
N = len(mspl)
#N = 10
for i in range(N):
#    Convert mis-spelt word to soundex
    msplData = phonix(mspl[i]) 
    
#    Search thru dictionary's soundex for matching mis-spelt soundex
    ret = [x for x in dictPhnx if msplData[1] == x[1]]
#   If no results are returned, find element with the minimum edit distance    
    if len(ret) == 0:
        data_editD = []
        for k in range(len(msplData)):
            eD = editD(msplData[1], dictPhnx[k][1])
            data_editD.append(msplData + (eD,))
            
        minEditD = min([y[-1] for y in data_editD])    
        print("Min edit distance: " + str(minEditD))
        ret = [x for x in data_editD if x[-1] == minEditD]
    
    nRet = len(ret)
#    Track returned words
#    print("Mis-spelt: " + mspl[i] + " Correct: " + crt[i])
#    print([w[0] for w in ret])
    
#    Are any of the returned results that are relevant?
#    Find returned term that matches with the correct(relevant) word
    retRel = [r for r in ret if r[0] == crt[i]]
    nRetRel = len(retRel)
    
#    Find if there are relevant data in the dictionary for the mis-spelt word
    rel = [x for x in dictSdex if x[0] == crt[i]]
    nRel = len(rel) 

    dataPhnx.append(msplData + (nRet, nRel, nRetRel))
    
#    Track progess of program
    print(str(i+1) + "/" + str(N))
    
timeEnd = time.time()
pTime_phnx = (timeEnd - timeStart)

# Save results for Soundex
path_phnx = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_phnx.dat"
pickle.dump(dataPhnx, open(path_phnx, "wb"))

path_pTime_phnx = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/pTime_phx.dat"
pickle.dump(pTime_phnx, open(path_pTime_phnx, "wb"))


os.system('say "Complete"')


#%% Compute Cumulative Precision

path_sDex = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_sDex.dat"
dataSdex = pickle.load(open(path_sDex, "rb"))

path_phnx = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_phnx.dat"
dataPhnx = pickle.load(open(path_phnx, "rb"))

# Compute precision for Soundex
dataSorted = []
n = 5
for i in range(n):
    dataSorted.append(t for t in dataSdex if t[2] == i)

prec_sdex = []
for i in range (len(dataSorted)):    
    retRel = 0
    ret = 0
    for t in dataSorted[i]:
        retRel += t[-1]
        ret += t[-3]
    prec_sdex.append(retRel/ret)
    
# Compute precision for Phonix+
dataSorted = []
n = 5
for i in range(n):
    dataSorted.append(t for t in dataPhnx if t[2] == i)

prec_phnx = []
for i in range (len(dataSorted)):    
    retRel = 0
    ret = 0
    for t in dataSorted[i]:
        retRel += t[-1]
        ret += t[-3]
    prec_phnx.append(retRel/ret)

#x = list(range(len(prec_sdex)))
    x = np.arange(len(prec_sdex))

# Plot results
barWidth = 0.25
plt.bar(x, prec_sdex, width = barWidth, zorder = 2) 
plt.bar(x + barWidth, prec_phnx, width = barWidth, zorder = 2)
# Labels
plt.xticks(x + barWidth/2, x)
plt.legend(["Soundex", "Phonix+"])
plt.xlabel('# of vowels after first letter')
plt.ylabel('Precision')
# Grid
plt.grid()
ax = plt.gca()
ax.grid(linestyle = ":")


#%% Compute Cumulative Recall
path_sDex = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_sDex.dat"
dataSdex = pickle.load(open(path_sDex, "rb"))

path_phnx = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_phnx.dat"
dataPhnx = pickle.load(open(path_phnx, "rb"))

# Compute precision for Soundex
dataSorted = []
n = 5
for i in range(n):
    dataSorted.append(t for t in dataSdex if t[2] == i)

recl_sdex = []
for i in range (len(dataSorted)):    
    retRel = 0
    ret = 0
    for t in dataSorted[i]:
        retRel += t[-1]
        ret += t[-2]
    recl_sdex.append(retRel/ret)
    
# Compute precision for Phonix+
dataSorted = []
n = 5
for i in range(n):
    dataSorted.append(t for t in dataPhnx if t[2] == i)

recl_phnx = []
for i in range (len(dataSorted)):    
    retRel = 0
    rel = 0
    for t in dataSorted[i]:
        retRel += t[-1]
        rel += t[-2]
    recl_phnx.append(retRel/rel)

#x = list(range(len(prec_sdex)))
x = np.arange(len(prec_sdex))

# Plot results
barWidth = 0.2
plt.bar(x, recl_sdex, width = barWidth, zorder = 2) 
plt.bar(x + barWidth, recl_phnx, width = barWidth, zorder = 2)
# Labels
plt.xticks(x + barWidth/2, x)
plt.legend(["Soundex", "Phonix"])
plt.xlabel('# of vowels after first letter')
plt.ylabel('Recall')
# Grid
plt.grid()
ax = plt.gca()
ax.grid(linestyle = ":")


#%% Compute Average Precision

path_sDex = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_sDex.dat"
dataSdex = pickle.load(open(path_sDex, "rb"))

path_phnx = "/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/result_phnx.dat"
dataPhnx = pickle.load(open(path_phnx, "rb"))

# Remove data with no returns
dataSdex = [t for t in dataSdex if t[-1] != 0]
dataPhnx = [t for t in dataPhnx if t[-1] != 0]

# Compute average precision for Soundex
dataSorted = []
n = 5
for i in range(n):
    dataSorted.append(t for t in dataSdex if t[2] == i)

prec_sdex = []
for i in range(len(dataSorted)):
    prec_sdex.append([t[-1]/t[-3] for t in dataSorted[i]])
    
precAvg_sDex = [sum(prec_sdex[i])/len(prec_sdex[i]) for i in range(len(prec_sdex))]
    
# Compute average precision for Phonix+
dataSorted = []
n = 5
for i in range(n):
    dataSorted.append(t for t in dataPhnx if t[2] == i)

prec_phnx = []
for i in range(len(dataSorted)):
    prec_phnx.append([t[-1]/t[-3] for t in dataSorted[i]])
    
precAvg_phnx = [sum(prec_phnx[i])/len(prec_phnx[i]) for i in range(len(prec_phnx))]


#x = list(range(len(prec_sdex)))
x = np.arange(len(prec_sdex))

# Plot results
fig = plt.figure(dpi=300)
barWidth = 0.2
plt.bar(x, precAvg_sDex, width=barWidth, zorder = 2)
plt.bar(x + barWidth, precAvg_phnx, width=barWidth, zorder = 2)
# Labels
plt.xticks(x + barWidth/2, x)
plt.legend(["Soundex", "Phonix"])
plt.xlabel('# of vowels after first letter')
plt.ylabel('Precision')
# Grid
plt.grid()
ax = plt.gca()
ax.grid(linestyle = ":")

fig.savefig("/Users/kaisoon/Google Drive/Code/Python/COMP90049_KT/LexNorm/result/results-plot.png", dpi=300)
plt.show()

