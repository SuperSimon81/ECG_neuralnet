import numpy as np

#Read the header file for the first header and get the diagnosis. Then add the csv to a list and myocardial infarction or list
import os
import sys,tty
import readchar


data = list()
for root, dirs, files in os.walk("PTBDB"):
    for file in files:
        if file.endswith(".hea"):
             #print(os.path.join(root, file))
             fp = open(os.path.join(root, file))

             diagnosisline = [line for line in fp if line.startswith('# Reason for admission:')]
             diagnosis = diagnosisline[0].rstrip("\n")
             diagnosis = diagnosis[24:]
             fullname = os.path.join(root,file.rstrip(".hea")+".csv")
             if diagnosis == "Myocardial infarction":
                print(fullname, "has an infarction")
                data.append((fullname,1))
             elif diagnosis == "Healthy control":
                print(fullname, "is a healthy control")
                data.append((fullname, 0))
             #print(,file.rstrip(".hea")+".csv")

             break

#while True:
#    key = readchar.readkey()
#    print("you pressed",key)

# tty.setcbreak(sys.stdin)
# key = ord(sys.stdin.read(1))  # key captures the key-code
# # based on the input we do something - in this case print something
# if key==97:
#     print ("you pressed a")
# else:
#     print ("you pressed something else ..."  )
# sys.exit(0)