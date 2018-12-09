import torch
from torch import nn
from torch.nn import init
from os.path import join, exists
from os import makedirs

def print_network(net):
	param_count = 0
	for param in net.parameters():
		param_count += param.numel()
	print(net)
	print("Total number of parameters: {0}".format(param_count))


def weights_init(m):
	if type(m) in (nn.Conv2d, nn.Linear):
		init.orthogonal_(m.weight.data)



def checkpoint(path, epoch, model):
	path = join(path, "checkpoint")
	if not exists(path):
		makedirs(path)
	save_path = join(path, "model_epoch_{}.pkl".format(epoch))
	torch.save(model.state_dict(), save_path)
	print("Checkpoint saved to {}".format(save_path))
