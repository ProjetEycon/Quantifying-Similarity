from Model.models import autoencoder,transformeur
from Data_set.Dataset import EYCON,caption_parser
from Data_set.utils import *
from tqdm import tqdm
import numpy as np
visual_modal=autoencoder().to(device=device)
dataset=EYCON()
textual_model=transformeur()
vectors=np.zeros((len(dataset),1536+1024))
for i in tqdm(range(len(dataset))):
    image,caption=dataset[i]
    vectors[i,:-1024]=visual_modal(image.unsqueeze(0).to(device)).cpu().detach().numpy()
    vectors[i,-1024:]=textual_model(caption_parser(caption)).cpu().detach().numpy()
np.save("vectors.py",vectors)