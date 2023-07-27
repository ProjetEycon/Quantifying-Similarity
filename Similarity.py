import numpy as np
from tqdm import tqdm
data=np.load("vectors.py")
same=open("similarity.txt","w")
for i in tqdm(range(data.shape[0])):
        _data=np.mean(np.absolute(data[:,:]-data[i,:]),axis=1)
        same.write(str(list(np.argsort(_data)[:30]))+"\n")