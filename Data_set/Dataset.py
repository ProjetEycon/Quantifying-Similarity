from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dset
from torchvision.utils import save_image
import torchvision.utils as vutils
from torchsummary import summary
import os
from PIL import Image
import json
import string
from unidecode import unidecode
from .utils import *
class EYCON(Dataset):
    def __init__(self, lst=lst,train=False, shuffle=True):
        self.data =sorted(lst[:int(len(lst)*1)] if not train  else lst[int(len(lst)*0.8):])
        self.nbdata = len(self.data)
    def __getitem__(self, index):
        try:
            x = Image.open(self.data[index]).convert('L').convert('RGB')
            y = transform(x).to(device)
            return y,self.data[index]
        except:
            return torch.ones((3,256,256)).to(device),"erreur"
    def __len__(self):
        return self.nbdata
def caption_parser(path=""):
    if path in caption.keys():
        legende=unidecode(caption[path]).lower()
        legende =legende.translate(str.maketrans('', '', string.punctuation))
        return [legende]
    else:
        return [""]