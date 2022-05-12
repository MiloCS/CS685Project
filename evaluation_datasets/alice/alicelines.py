import pandas as pd

'''
This python script formats the lines from alice in wonderland, creating a CSV
'''

myfile = open('alice.txt', 'r')
lines = myfile.readlines()
newlines = []
for line in lines:
  newlines.append(line.strip())
df= pd.DataFrame()
df["text"] = newlines
df.to_csv('alice_lines.csv')
