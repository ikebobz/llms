import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(threshold=float('inf'))
sorted_uniquechars = []
words = []
with open('pg2701.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    words = content.split() #read the contents of the file into a list
    length = len(words)
    maxword = max(len(w) for w in words) #find the length of the longest word
    minword = min(len(w) for w in words) #find the length of the shortest word
    """print('The shortest word is:', minword)
    print('The longest word is:', maxword)
    print(length)"""
    # Count the frequency of each bigram (pair of consecutive characters) in the words
    b = {}
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            bigram = (ch1, ch2)
            b[bigram] = b.get(bigram, 0) + 1

    # Sort the bigrams by frequency in descending order
    bsorted = sorted(b.items(), key=lambda x: -x[1])

    # Create a sorted list of unique characters found in the words
    uniquechars = set(''.join(words))
    sorted_uniquechars = sorted(list(uniquechars))
    sorted_uniquechars.append('.')
    
    print('Length of sorted list is :',len(sorted_uniquechars))
   
# Determine the length of the sorted unique characters list
length_of_uniquechars = len(sorted_uniquechars)

# Initialize a matrix N of zeros with dimensions based on the length of unique characters
N = torch.zeros([length_of_uniquechars, length_of_uniquechars], dtype=torch.int32)

# Create a dictionary to map each character to a unique index
stoi = {ch: i + 1 for i, ch in enumerate(sorted_uniquechars)}
stoi['.'] = 0  # Map the special character '.' to index 0

# Create a reverse dictionary to map indices back to characters
itos = {i: ch for ch, i in stoi.items()}

# Populate the matrix N with bigram counts
for w in words:
    chs = ['.'] + list(w) + ['.']  # Add special characters at the beginning and end of each word
    for ch1, ch2 in zip(chs, chs[1:]): 
        N[stoi[ch1], stoi[ch2]] += 1  # Increment the count for the corresponding bigram in the matrix
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
# Set a manual seed for the random number generator for reproducibility
# Convert the matrix N to float type and normalize it by dividing each row by its sum
g = torch.Generator().manual_seed(2145678900)
M = N.float()
M /= M.sum(dim=1, keepdim=True)
#print(M)

"""s = N[0].float()
s = s/s.sum()
for k in range(20):
     iy = torch.multinomial(s,num_samples=1,replacement=True,generator=g).item()
     print(itos[iy])"""
# Generate 30 sequences of characters based on the bigram probabilities
# Each sequence starts with the special character '.' and continues until the special character '.' is encountered again
# The sequences are generated using multinomial sampling from the probability distribution of the bigrams
for i in range(30):
    out = []    
    ix = 0
    while True:
        p = M[ix]
        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))



#print(ix)
#print(itos[ix.item()])
# Open the PDF file


