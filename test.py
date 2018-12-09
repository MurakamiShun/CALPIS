import argparse
import os
import torch
import torchvision.transforms as transforms
import torchvision
import glob
from PIL import Image
import time
import numpy as np
from torch import nn

#--- Argment Setting ---#
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-t','--test_img', help='image path',type=str, required=True)
parser.add_argument('-m','--model', help='model path', type=str, required=True)
parser.add_argument('-g','--cuda', help='use cuda', action='store_true')
opt = parser.parse_args()


#--- Model Initialize ---#
torch.no_grad()
model = torchvision.models.vgg16()
model.classifier = nn.Sequential(
	nn.Linear(25088,4096),
	nn.ReLU(inplace=True),
	nn.Dropout(),
	nn.Linear(4096,4096),
	nn.ReLU(inplace=True),
	nn.Dropout(),
	nn.Linear(4096,7)
)
model.load_state_dict(torch.load(opt.model))
model = model.eval()
if opt.cuda:
	model = model.cuda()

#--- transform ---#
transform = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
])

class Images():
	def __init__(self, dir):
		self.paths = glob.glob(os.path.join(dir, "*"))
		self.paths.sort()
		self.images = []
		for path in self.paths:
			self.images.append(Image.open(path).convert('RGB'))

	def __getitem__(self, index):
		return self.images[index]

	def getname(self, index):
		return os.path.split(self.paths[index])[1]

	def __len__(self):
		return len(self.images)

def main():
	imgs = Images(opt.test_img)

	print("image_name\telapsed_time")
	sum_time = 0
	total_images = len(imgs)
	for n in range(len(imgs)):
		start = time.time()
		img = transform(imgs[n])
		c,h,w = img.shape
		img = img.reshape(1,c,h,w)
		image_name = imgs.getname(n)
		if opt.cuda:
			img = img.cuda()
		out = torch.nn.functional.softmax(model(img),1).reshape(7)
		#out = model(img)
		if opt.cuda:
			out = out.cpu()
		print("{0}\n{1}".format(image_name,out.max(0)))
		elapsed_time = time.time() - start
		print("{} [s]".format(elapsed_time))
		sum_time += elapsed_time


if __name__ == "__main__":
	main()
