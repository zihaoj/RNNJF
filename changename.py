import glob
import os

files = glob.glob("V47*")
print files

for f in files:
    newf = f.replace("4000k", "5000k")
    print f
    print newf
    os.system("mv "+f+" "+newf)
