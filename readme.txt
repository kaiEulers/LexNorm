README for proj1_submit.py

The following README file briefly details the step-by-step procedures implemented by proj1_submit.py works.

Step 1:
matplotlib.pyplot is imported to graph the output at the end of this program.

Step 2:
The function soundex(w) is created. It returns a tuple (w, sc, nV) when used on a word.
    w is the original word
    sc is the sound code of the word
    nV is the number of vowels in the word
The function phonix(w) is created. It returns a tuple (w, sc, nV) when used on a word.
    w is the original word
    sc is the sound code of the word
    nV is the number of vowels in the word
The function match(a, b) is created. It returns 0 if a matches b and 1 if a doesn't match b.
The function editD(tok, dic) is created. It returns the minimum edit distance between tok and dic.

Step 3:
Text files of the mis-spelt words, the dictionary, and the correctly spelt words are opened in the program and saved into variables mspl, crt, dic respectively.

Step 4:
soundex() in used on all words in the dictionary, returning a list of tuples. Outputs are stored in dictSdex.
phonix() in used on all words in the dictionary, returning a list of tuples. Outputs are stored in dictPhnx.

Step 5:
soundex() is used on a mis-spelt word in mspl. Output is stored in msplData.

Step 6:
Search through the sound codes in the dictionary dic and return a list of tuple/s where the sound code/s matches that of the sound code of the mis-spelt word msplData. Output is a list of returned matches stored in ret. Compute the number of elements in ret.

Step 7:
Search through the list of returned matches for the word that appears in the list of correctly spelt words. Output is a  returned match that are relevant stored in retRel. Compute the number of elements in retRel.

Step 8:
Search through the dictionary for a value that is relevant. Output is a relevant match stored in rel. Compute the number of elements in rel. This step is used to calculate recall and does not contribute to the goal of this program.

Step 9:
A new tuple (w, sc, nV, nRet, nRel, nRetRel) is created
    nRet is the number of returned matches
    nRel is the number of relevant matches
    nRetRel is the number of returned matches that are relevant

Step 5 to 9 is repeated for every word in the list of mis-spelt words mspl. The output at the end of step 9 is appended to form a list of tuple stored in dataSdex.

Step 10:
Step 5 to 9 is also repeated for every word in the list of mis-spelt words mspl but using phonix() this time. The output at the end of step 9 is appended to form a list of tuple stored in dataPhnx.

Step 11:
Elements in dataSdex are sorted into 5 lists. The first list contains words with no vowels, the second contains words with 1 vowel, the third contains words with 1 vowel, and so on and so forth until the fifth, which contains words with 4 vowels. The 5 lists are stored in dataSorted.

Step 12:
For each list in dataSorted, the number of returned matches are summed up. The number of returned matches that are relevant are also summed up. The precision for each list is then computed. The output is a list of precision value for each vowel count group of dataSorted, stored in prec_sdex.

Step 13:
Step 11 to 13 is repeated for elements in dataPhnx. The output is a list of precision value stored in prec_phnx

Step 14:
Computed precisions in prec_sdex and prex_phnx are visualised on a bar graph using the matplotlib library
