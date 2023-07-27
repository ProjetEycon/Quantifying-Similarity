from matplotlib import pyplot as plt
import cv2 as cv
import json
from tqdm import tqdm
import os
os.mkdir("click")
path=json.load(open("pathimages.json"))
clicks=[x.replace("[",'').replace("]","").replace(" ","").replace("\n","").split(",") for x in open("click.txt","r").readlines()]
for c in tqdm(clicks):
    cpt=1
    try:
        for img in c:
            p=path[img]
            plt.subplot(1,len(c),cpt)
            plt.imshow(cv.resize(cv.imread(p,0),(256,256)),cmap="gray")
            cpt+=1
        plt.savefig("click\\"+c[0]+".png")
    except:
        print("lecture error")
    