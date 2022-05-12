import pandas as pd

'''
This python script formats the lines from the aeneid, by Virgil, creating a CSV
'''

myfile = open('virgil.txt', 'r')
lines = myfile.readlines()
newlines = []
for line in lines:
  newlines.append(line.strip())
df= pd.DataFrame()
df["text"] = newlines
df.to_csv('virgil_lines.csv')
