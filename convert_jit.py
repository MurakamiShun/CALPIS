import argparse
import torch
import torchvision
from torch import nn

#--- Argment Setting ---#
parser = argparse.ArgumentParser(add_help=True)
parser.add_argument('-m','--model', help='model path', type=str, required=True)
opt = parser.parse_args()


model = torchvision.models.vgg16(pretrained=False)
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
model.eval()
model.cpu()
dummy = torch.rand(1,3,224,224)

script_module = torch.jit.trace(model, dummy)
script_module.save("script_module.pt")