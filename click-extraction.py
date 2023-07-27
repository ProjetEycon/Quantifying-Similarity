from tqdm import tqdm
vis=open("similarity.txt","r").readlines()
from tqdm import tqdm
vis=[[int(y) for y in x.replace("[","").replace("]","").replace("\n","").replace(" ","").split(",")] for x in vis]
resultat=[]
print(vis[63727])
for ligne in tqdm(vis):
    clicksmax=[]
    for i in range(2,15):
        clicks=True
        _set=ligne[:i]
        for element in ligne[1:i]:
            if set(_set)!=set(vis[element][:i]):
                clicks=False
        if clicks:
            clicksmax=ligne[:i]
    resultat.append(clicksmax)
resultat=list(filter(lambda x:len(x)>0,resultat))
test=open("click.txt","w")
for ligne in resultat:
    test.write(str(ligne)+"\n")
       