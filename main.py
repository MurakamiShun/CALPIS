#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import random
from os.path import join
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
import time

from dataset import get_train_dataset, get_test_dataset
from util import weights_init, print_network, checkpoint

torch.set_num_threads(4)

#--- params ---#
epochs = 500
first_epoch = 1
batch_size = 8

#--- random seed ---#
np.random.seed(1)
random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

transform = transforms.Compose([
	transforms.Resize(256),
	transforms.RandomCrop(224),
	transforms.ToTensor()
])

#--- parent path ---#
parent_path = "./save"

#--- path of log ---#
log = SummaryWriter(join(parent_path,"log"))


#--- dataset path ---#
dataset_path = "./dataset"


#--- gpu(cuda) setting ---#
use_gpu = torch.cuda.is_available()
if use_gpu:
	print("cuda is available!")
	cudnn.benchmark = True
	cudnn.deterministic = True
	#print(torch.cuda.max_memory_allocated())

#--- model init ---#
model = torchvision.models.vgg16(pretrained=False)
param = torch.load("./vgg16-397923af.pth")
model.load_state_dict(param)
model.classifier = nn.Sequential(
	nn.Linear(25088,4096),
	nn.ReLU(inplace=True),
	nn.Dropout(),
	nn.Linear(4096,4096),
	nn.ReLU(inplace=True),
	nn.Dropout(),
	nn.Linear(4096,7)
)

#--- model load ---#
model_load = False
if model_load:
	param = torch.load("./model_epoch_.pkl")
	model.load_state_dict(param)
	first_epoch = 1

#-- optimizer and loss function setting --#
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=3e-5, momentum=0.9)


use_scheduler = False
if use_scheduler:
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.8)

save_epoch = 1

def train(model, train_loader):
	model.train()
	running_loss = 0
	total = 0
	correct = 0
	if use_scheduler:
		scheduler.step()
	for batch_idx, (img, label) in enumerate(train_loader):
		label = label.reshape(label.shape[0])
		if use_gpu:
			img = img.cuda()
			label = label.cuda()

		optimizer.zero_grad()
		out = model(img)
		loss = criterion(out, label)
		loss.backward()
		optimizer.step()
		running_loss += loss.item()
		_,predicted = torch.max(out.data,1)
		correct += (predicted == label).sum().item()
		total += label.shape[0]

		if batch_idx%10 == 0:
			sys.stdout.write("\r{0}/{1}".format(batch_idx, len(train_loader)))
			sys.stdout.flush()

	train_loss = running_loss / len(train_loader)
	return [train_loss,correct/total]



def test(model, test_loader):
	model.eval()
	running_loss = 0
	total = 0
	correct = 0
	for batch_idx, (img, label) in enumerate(test_loader):
		label = label.reshape(label.shape[0])
		if use_gpu:
			img = img.cuda()
			label = label.cuda()
		

		with torch.no_grad():
			out = model(img)
			loss = criterion(out, label)
			running_loss += loss.item()

			_,predicted = torch.max(out.data,1)
			correct += (predicted == label).sum().item()
			total += label.shape[0]

	test_loss = running_loss / len(test_loader)
	return [test_loss,correct/total]



def main():
	print_network(model)

	#if log is not None:
		#dummy_input = torch.rand(batch_size, 1, 128, 128)
		#log.add_graph(model, dummy_input)

	if use_gpu:
		model.cuda()


	train_set = get_train_dataset(dataset_path, transform)
	test_set = get_test_dataset(dataset_path, transform)

	train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
	test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

	print("train images: {}".format(len(train_set)))
	print("test images: {}".format(len(test_set)))
	print("epoch: {}".format(epochs))
	print("batch size: {}".format(batch_size))

	loss_list = []
	test_loss_list = []

	start_time = time.time()

	for epoch in range(first_epoch, epochs+1):
		loss,acc = train(model, train_loader)
		test_loss,test_acc = test(model, test_loader)

		elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time()-start_time))
		print("\repoch:{0}    loss:{1:.8f}    test_loss:{2:.8f}    elapsed_time:{3}".format(epoch, loss, test_loss, elapsed))
		print("Accuracy:{0:.4f}    test_Accuracy:{1:.4f}".format(acc, test_acc))

		if log is not None:
			log.add_scalars("loss",{"train":loss, "test":test_loss}, epoch)

		if epoch%save_epoch == 0:
			checkpoint(parent_path, epoch, model)



if __name__=='__main__':
	main()