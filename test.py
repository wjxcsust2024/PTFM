import torch
import numpy as np
from torch.utils.data import DataLoader
from Net import Net
import os
from dataloader2 import Datases_loader as dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batchsz = 1 #3
model= Net().to(device)
model= model.eval()
savedir = r'/tmp/new9/weights/crack537_3_4.pth'
imgdir = r'/tmp/new9/Deepcrack/Deepcrack/test_img'
labdir = r'/tmp/new9/Deepcrack/Deepcrack/test_lab'
#imgdir = r'/tmp/new9/crack776/crack776/test/image'
#labdir = r'/tmp/new9/crack776/crack776/test/label'

imgsz = 512
resultsdir = r'/tmp/new9/results'

dataset = dataloader(imgdir, labdir, imgsz, imgsz)
testsets = DataLoader(dataset, batch_size=batchsz, shuffle=False)

def test():
    model.load_state_dict(torch.load(savedir))
    exist = os.path.exists(resultsdir)
    if not exist:
        os.makedirs(resultsdir)
    for idx, samples in enumerate(testsets):
        img, lab = samples['image'], samples['mask']
        img, lab = img.to(device), lab.to(device)

        pred = model(img)

        np.save(resultsdir + r'/pred' + str(idx+1) + '.npy', pred.detach().cpu().numpy())
        np.save(resultsdir + r'/label' + str(idx+1) + '.npy', lab.detach().cpu().numpy())

if __name__ == '__main__':
    test()