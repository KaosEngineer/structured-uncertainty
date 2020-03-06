#!python

import xml.etree.ElementTree as ET
from optparse import OptionParser

parser = OptionParser()
parser.add_option('-i','--input', dest='in_file', default=None)
parser.add_option('-o','--output', dest='out_file', default=None)

(options,args) = parser.parse_args()

if options.in_file==None or options.out_file==None:
    print("Input or output file missing\nUsage: python sgml-map.py -i in-file -o out-file")
    exit(1)

ifile = open(options.in_file,'r')
ofile = open(options.out_file,'w')

pathStart=False
for l in ifile:
    if pathStart:
        a = l.strip().split(":")
        A = 0
        print('Parsing: ', a)
        if a[0]=='':
            pathStart=False
            continue
        for i in range(len(a)):
            b = a[i].split(",")
            if b[0]=="I":
                #if float(b[4])>0.000000:
                    ofile.write(" 1") # I - insertion
                    A=A+1
            elif b[0]=="C":
                #if float(b[4])>0.000000:
                    ofile.write(" 0 ") # C - correct
                    A=A+1
            elif b[0]=="D":
                # do nothing for deletions
                continue
            elif b[0]=="S":
                #if float(b[4])>0.000000:
                    ofile.write(" 1 ") # S - substitution
                    A=A+1
            else:
                print("Unexpected line: " + str(b))
                exit(1)
        if A>0:
            ofile.write("\n")
        pathStart=False
    elif l.startswith('<PATH'):
        pathStart=True
ifile.close()
ofile.close()
exit(0)