import pandas as pd

'''
This python script formats the lines from moby dick, creating a CSV
'''

myfile = open('moby.txt', 'r')
lines = myfile.readlines()
newlines = []
for line in lines:
  newlines.append(line.strip())
df= pd.DataFrame()
df["text"] = newlines
df.to_csv('moby_lines.csv')
