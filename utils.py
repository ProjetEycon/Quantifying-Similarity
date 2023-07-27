import torch
from torchvision import transforms
import json
image_size = 256
nb_channls=3
workers=0
num_epochs = 10
batch_size = 64
learning_rate = 1e-3
weight_decay = 1e-5
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        ])
caption=json.load(open("caption.json"))
imagepath=json.load(open("pathimages.json"))
lst=list(imagepath.values())