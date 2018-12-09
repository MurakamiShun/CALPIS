from os.path import join
from torch.utils import data
from PIL import Image
import cv2
import numpy as np
import glob
import torch


class ImageDataset(data.Dataset):
	def __init__(self, file_path, transform=None):
		super(ImageDataset, self).__init__()
		self.taste = ["orange","mango","pine","grape", "calpis", "hokai","photo"]
		self.transform = transform
		self.paths = []
		label = 0
		self.labels = []
		for name in self.taste:
			path = glob.glob(join(file_path,name, "*"))
			self.paths += path
			self.labels += [label for i in range(len(path))]
			label+=1

	def __getitem__(self, index):
		#--- noise ---#
		img = Image.open(self.paths[index]).convert('RGB')
		img = self.transform(img)
		#--- origin ---#
		return img, torch.tensor([self.labels[index]],dtype=torch.long)

	def __len__(self):
		return len(self.labels)


def get_train_dataset(dataset_path, transform=None):
	return ImageDataset(join(dataset_path, "train"), transform)

def get_test_dataset(dataset_path, transform=None):
	return ImageDataset(join(dataset_path, "test"), transform)