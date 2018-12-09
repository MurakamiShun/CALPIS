#define _GLIBCXX_USE_CXX11_ABI 0
#include <torch/torch.h>
#include <torch/script.h>
#include "calpis.hpp"
#include <iostream>
#include <memory>
#include <vector>

std::vector<double> calpis(unsigned char* ptr, const int w, const int h){
	using torch::Tensor;
	static auto module = torch::jit::load("../script_module.pt");
	Tensor img = torch::zeros({3,h*w});
	for (unsigned long idx = 0; idx < w*h; idx++) {
		img[0][idx] = *ptr;
		ptr+=1;
		img[1][idx] = *ptr;
		ptr+=1;
		img[2][idx] = *ptr;
		ptr+=2;
	}
	img.reshape({ 3,h,w });
	img /= 255;

	std::cout << img[0].mean() << std::endl;
	std::cout << img[1].mean() << std::endl;
	std::cout << img[2].mean() << std::endl;

//	img[0].normal_(0.485, 0.229);
//	img[1].normal_(0.456, 0.224);
//	img[2].normal_(0.406, 0.225);


	std::vector<torch::jit::IValue> input;
	input.push_back(img.reshape({1,3,h,w}));
	auto out = module->forward(input).toTensor().softmax(1);
	std::vector<double> rtn;
	for (int i = 0; i < 7; i++) {
		double data = out[0][i].item<double>();
		std::cout << data << std::endl;
		rtn.push_back(data);
	}
	return std::move(rtn);
}