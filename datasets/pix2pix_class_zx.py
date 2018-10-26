#encoding=utf8
import os
import os.path
import numpy as np
from PIL import Image

import torch.utils.data

def default_loader(path):
    return Image.open(path).convert('RGB')

class pix2pix(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, loader=default_loader, seed=None):
        self.label_dirname_list = []
        self.label_files_list = []
        self.imgs = []
        self.files_num = []
        for dirname in os.listdir(root):
            if dirname[0] == '.' or not os.path.isdir(os.path.join(root, dirname)):
                continue
            self.label_dirname_list.append(dirname)
            file_list = []
            for filename in os.listdir(os.path.join(root, dirname)):
                img_path = os.path.join(root, dirname, filename)
                if filename[0] == '.' or not os.path.isfile(img_path):
                    continue
                file_list.append(img_path)
                self.imgs.append(img_path)
            self.label_files_list.append(file_list)
            self.files_num.append(len(file_list))
        self.root = root
        self.transform = transform
        self.loader = loader
        self.label_num = len(self.label_dirname_list)
        if seed is not None:
            np.random.seed(seed)

    def __getitem__(self, index):
        label = np.random.randint(0, self.label_num)
        index = np.random.randint(0, self.files_num[label])
        path = self.label_files_list[label][index]
        img = self.loader(path)
        w, h = img.size
        # NOTE: split a sample into imgA and imgB
        imgA = img.crop((0, 0, int(w/2), h))
        imgB = img.crop((int(w/2), 0, w, h))


        if self.transform is not None:
            # NOTE preprocessing for each pair of images
            imgA, imgB = self.transform(imgA, imgB)

        return imgA, imgB, label

    def __len__(self):
        return len(self.imgs)
