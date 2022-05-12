import os
import pandas as pd

'''
This file takes all of the sentences for each bible type
And creates a CSV with these sentences.
Equivalent sentences all sit in the same row.
The column names are the type of each bible.
'''

mydict = {}
for dir in os.listdir("./bibles"):
  mydict[dir] = 0

print(mydict)

prevdir = "ASV"
newdict = {}
i = 0
textfilelist = set()
for mydir in mydict:
  for subdir in os.listdir("./bibles/" + mydir + "/"):
    if i == 0:
      for element in os.listdir("./bibles/" + mydir + "/" + subdir + "/"):
        textfilelist.add(element)
    for textfile in os.listdir("./bibles/" + mydir + "/" + subdir + "/"):
      newdict[mydir + '_' + textfile] = len(open("./bibles/" + mydir + "/" + subdir + "/" + textfile).readlines())
  i += 1

myset = set()
for file in textfilelist:
  j = 0
  amount = 0
  for key in newdict:
    if key.split('_')[1] == file:
      if j == 0:
        amount = newdict[key]
        j += 1
      else:
        if newdict[key] != amount:
          myset.add(key.split('_')[1])

finalfiles = [x for x in textfilelist if x not in myset]

df = pd.DataFrame()
# {'ASV': 0, 'LEB': 0, 'WEB': 0, 'BBE': 0, 'DARBY': 0, 'DRA': 0, 'YLT': 0, 'KJV': 0}
asvlist = []
leblist = []
weblist = []
bbelist = []
darbylist = []
dralist = []
yltlist = []
kjvlist = []
for finalfile in finalfiles:
  for line in open("./Bibles2/ASV" + "/" + finalfile).readlines():
    asvlist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/LEB" + "/" + finalfile).readlines():
    leblist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/WEB" + "/" + finalfile).readlines():
    weblist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/BBE" + "/" + finalfile).readlines():
    bbelist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/DARBY" + "/" + finalfile).readlines():
    darbylist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/DRA" + "/" + finalfile).readlines():
    dralist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/YLT" + "/" + finalfile).readlines():
    yltlist.append(line.lstrip('0123456789.- '))
  for line in open("./Bibles2/KJV" + "/" + finalfile).readlines():
    kjvlist.append(line.lstrip('0123456789.- '))

df['ASV'] = asvlist
df['LEB'] = leblist
df['WEB'] = weblist
df['BBE'] = bbelist
df['DARBY'] = darbylist
df['DRA'] = dralist
df['YLT'] = yltlist
df['KJV'] = kjvlist

df.to_csv("./equal_sentences.csv")
    
