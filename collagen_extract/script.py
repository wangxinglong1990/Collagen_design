##Python script for collagen fragments extraction
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-seq', type=str)
parser.add_argument('-len', type=int)
args = parser.parse_args()

input_seq = args.seq
length = args.len
all_seq = []
if length == 30:
    for k in range(len(input_seq)-30+1):
        seq = input_seq[k:k + 30]
        if seq[0] == 'G' and seq[3] == 'G' and seq[6] == 'G' and seq[9] == 'G' and seq[12] == 'G' and seq[15] == 'G' and seq[18] == 'G' and seq[21] == 'G' and seq[24] == 'G' and seq[27] == 'G':
            all_seq.append(seq)
    f=open('extract.txt','w')
    f.close()
    f=open('extract.txt','a+')
    for i in all_seq:
        f.write('%s\n'%i)
    f.close()
if length == 27:
    for k in range(len(input_seq)-27+1):
        seq = input_seq[k:k + 27]
        if seq[0] == 'G' and seq[3] == 'G' and seq[6] == 'G' and seq[9] == 'G' and seq[12] == 'G' and seq[15] == 'G' and seq[18] == 'G' and seq[21] == 'G' and seq[24] == 'G' :
            all_seq.append(seq)
    f=open('extract.txt','w')
    f.close()
    f=open('extract.txt','a+')
    for i in all_seq:
        f.write('%s\n'%i)
    f.close()
if length == 33:
    for k in range(len(input_seq)-33+1):
        seq = input_seq[k:k + 33]
        if seq[0] == 'G' and seq[3] == 'G' and seq[6] == 'G' and seq[9] == 'G' and seq[12] == 'G' and seq[15] == 'G' and seq[18] == 'G' and seq[21] == 'G' and seq[24] == 'G' and seq[27] == 'G' and seq[30] == 'G':
            all_seq.append(seq)
    f=open('extract.txt','w')
    f.close()
    f=open('extract.txt','a+')
    for i in all_seq:
        f.write('%s\n'%i)
    f.close()
  
##Python script for collagen fragments properties calculation
import peptides
f= open(r'peptide_library.txt')
a=f.readlines()
f.close()
list = []
for i in a:
    list.append(i.strip('\n'))
f= open('peptide_properties','w')
f.close()
f= open('peptide_properties','a+')
for i in list:
    peptide = peptides.Peptide('%s'%i)
    f.write('%s %s %s %s %s\n'%(peptide.boman(),
                          peptide.charge(pKscale="Murray"),
                          peptide.hydrophobicity(scale="Barley"),
                            peptide.isoelectric_point(pKscale="Murray"),
                             peptide.molecular_weight()))
f.close()
