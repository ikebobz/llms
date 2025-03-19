import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=float('inf'))
sorted_uniquechars = []
words = []
with open('pg2701.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    words = content.split()
    length = len(words)
    maxword = max(len(w) for w in words)
    minword = min(len(w) for w in words)
    """print('The shortest word is:', minword)
    print('The longest word is:', maxword)
    print(length)"""
    b = {}
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]):
            #print(ch1,ch2)
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1
    bsorted = sorted(b.items(), key=lambda x: -x[1])
    uniquechars = set(''.join (words))
    sorted_uniquechars = sorted(list(uniquechars))
    sorted_uniquechars.append('.')
    #print(bsorted)
    #clprint(sorted_uniquechars)
    print('Length of sorted list is :',len(sorted_uniquechars))
    #print(words)
length_of_uniquechars = len(sorted_uniquechars)
N = torch.zeros([length_of_uniquechars,length_of_uniquechars],dtype=torch.int32)
#print(a.dtype)
stoi = {ch:i+1 for i,ch in enumerate(sorted_uniquechars)}
stoi['.'] = 0
itos = {i:ch for ch,i in stoi.items()}
#print(itos)
for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1,ch2 in zip(chs,chs[1:]): 
            N[stoi[ch1],stoi[ch2]] += 1
#print('The matrix is:',N)
"""plt.figure(figsize=(16,16))
plt.imshow(N,cmap='Blues')
for i in range(length_of_uniquechars):
    for j in range(length_of_uniquechars):
        chstr = itos[i] + itos[j]
        plt.text(j,i,chstr,ha='center',va='bottom',color='gray')
        plt.text(j,i,N[i,j].item(),ha='center',va='top',color='gray')
plt.axis('off')
plt.show()"""
g = torch.Generator().manual_seed(2145678900)
M = N.float()
M /= M.sum(dim=1,keepdim=True)
#print(M)

"""s = N[0].float()
s = s/s.sum()
for k in range(20):
     iy = torch.multinomial(s,num_samples=1,replacement=True,generator=g).item()
     print(itos[iy])"""
for i in range(30):
    out = []    
    ix = 0
    while True:
        p = M[ix]
        ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))



#print(ix)
#print(itos[ix.item()])
# Open the PDF file


