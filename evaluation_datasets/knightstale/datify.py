import pandas as pd
'''
This python script formats the parallel lines from the Knight's Tale, from the Canterbury Tales
The text has old english sentences, each followed by a translated modern English sentence.
These are formed into parallel lines in the csv, with each row holding two versions of the same line.
The columns are titled 'original' and 'modern'.
'''
df = pd.DataFrame()
myfile = open("thetext.txt")
lines = myfile.readlines()
even = True
oldlines = []
newlines = []
for line in lines:
  if even:
    oldlines.append(line)
    even = False
  else:
    newlines.append(line)
    even = True

df['original'] = oldlines
df['modern'] = newlines

df.to_csv("knights_tale.csv")
