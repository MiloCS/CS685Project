import os
import pandas as pd
import shutil as sh

'''
This python file looks through the Bibles directory, and moves all files out of nested subdirectories
So that every bible type subdirectory contains all of the text files for that bible.
'''

mydict = {}
for dir in os.listdir("./Bibles"):
  mydict[dir] = 0

#Move textfiles out of subdirectories 
for mydir in mydict:
  for subdir in os.listdir("./Bibles2/" + mydir + "/"):
    for textfile in os.listdir("./Bibles2/" + mydir + "/" + subdir + "/"):
      sh.copy("./Bibles2/" + mydir + "/" + subdir + "/" + textfile, "./Bibles2/" + mydir + "/" + textfile)
    sh.rmtree("./Bibles2/" + mydir + "/" + subdir)

