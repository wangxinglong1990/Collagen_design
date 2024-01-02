import numpy as np

def encode(seq):
    encoded_seq = np.zeros(len(seq)*21,int)
    for j in range(len(seq)):
        if seq[j] == 'H':
            encoded_seq[j*21] = 1
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'D':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 1
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'R':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 1
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'F':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 1
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'A':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 1
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0
        elif seq[j] == 'C':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 1
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'G':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 1
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'Q':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 1
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'E':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 1
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'K':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 1
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'L':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 1
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'M':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 1
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'N':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 1
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'S':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 1
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'Y':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 1
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0
        elif seq[j] == 'T':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 1
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'I':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 1
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'W':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 1
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'P':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 1
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0

        elif seq[j] == 'V':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 1
            encoded_seq[j*21+20] = 0
        elif seq[j] == 'O':
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 1
        else:
            encoded_seq[j*21] = 0
            encoded_seq[j*21+1] = 0
            encoded_seq[j*21+2] = 0
            encoded_seq[j*21+3] = 0
            encoded_seq[j*21+4] = 0
            encoded_seq[j*21+5] = 0
            encoded_seq[j*21+6] = 0
            encoded_seq[j*21+7] = 0
            encoded_seq[j*21+8] = 0
            encoded_seq[j*21+9] = 0
            encoded_seq[j*21+10] = 0
            encoded_seq[j*21+11] = 0
            encoded_seq[j*21+12] = 0
            encoded_seq[j*21+13] = 0
            encoded_seq[j*21+14] = 0
            encoded_seq[j*21+15] = 0
            encoded_seq[j*21+16] = 0
            encoded_seq[j*21+17] = 0
            encoded_seq[j*21+18] = 0
            encoded_seq[j*21+19] = 0
            encoded_seq[j*21+20] = 0
    encoded_seq = encoded_seq.reshape(len(seq),21)
    return encoded_seq

def get_input(seqs,seq_lenth):
    data = np.zeros((1,seq_lenth,21))
    count = 0
    for i in seqs:
        count += 1
        if len(i) <= seq_lenth:
            i = i + (seq_lenth-len(i))*"X"
        if len(i) > seq_lenth:
            i = i[0:64]
        if count == 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = data + single_seq
            data = np.expand_dims(data, axis=0)
        if count != 1:
            single_seq = encode(i)
            single_seq = np.expand_dims(single_seq, axis=0)
            single_seq = np.expand_dims(single_seq, axis=0)
            data = np.concatenate ((data,single_seq),axis=0)
    return data
