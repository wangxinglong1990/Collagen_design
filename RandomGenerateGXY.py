import random
residues = ['A','G','D','N','F','P','Q','C','E','H','I','K','L','M','R','S','T','V','W','Y','O']
seq = 'GPOGPOGPOGAWGPOGPOGPOGPO'
#print(random.choice(residues))
f = open ('extract.txt', 'w')
f.close()
c=0
l = ['a','b']
num_list = []
for i in range(1,10):
    num_list.append(i*3+1)
    num_list.append(i * 3 + 2)
num_list.append(1)
num_list.append(2)
print(num_list)
for i in range(1):
    c+=1
    f = open ('extract.txt','a+')
    par = 'G'+random.choice(residues)+random.choice(residues)+'G'+random.choice(residues)+random.choice(residues)
    #print( random.choice(l))
    if random.choice(l) == 'a':
        #print(1)
        w_seq = seq + par
    if random.choice(l) == 'b':
        #print(2)
        w_seq = par + seq
    x = int(random.choice(num_list))
    f_seq = w_seq[0:x]+random.choice(residues)+ w_seq[(x+1):30]
    print(f_seq)
    x = int(random.choice(num_list))
    f_seq = f_seq[0:x] + random.choice(residues) + f_seq[(x + 1):30]
    print(f_seq)
    x = int(random.choice(num_list))
    f_seq = f_seq[0:x] + random.choice(residues) + f_seq[(x + 1):30]
    print(f_seq)
    x = int(random.choice(num_list))
    f_seq = f_seq[0:x] + random.choice(residues) + f_seq[(x + 1):30]
    print(f_seq)
    x = int(random.choice(num_list))
    f_seq = f_seq[0:x] + random.choice(residues) + f_seq[(x + 1):30]
    print(f_seq)
    x = int(random.choice(num_list))
    f_seq = f_seq[0:x] + random.choice(residues) + f_seq[(x + 1):30]
    print(f_seq)
    f.write('%s\n'%f_seq)

    f.close()
