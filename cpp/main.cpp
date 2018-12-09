# include <Siv3D.hpp> // OpenSiv3D v0.3.0
#include <algorithm>
#include <vector>
#include <iostream>
#include "calpis.hpp"

int max_label(const std::vector<double>& prob) {
	double max = prob[0];
	int label = 0;
	for (int idx = 1; idx < prob.size(); idx++) {
		if (prob[idx] > max) {
			max = prob[idx];
			label = idx;
		}
	}
	return label;
}

void Main(){
	Graphics::SetBackground(ColorF(0.8, 0.9, 1.0));
	Window::Resize(1280, 720);

	const Font font(40);

	Array<String> label_name = {
		U"みかん",
		U"マンゴー",
		U"パイン",
		U"巨峰",
		U"カルピス",
		U"北海道",
		U"No CALPIS"
	};
	Array<Texture> tex_taste;
	for (auto taste_name : label_name) {
		tex_taste.push_back(Texture(U"alpha/"+taste_name+U".png"));
	}
	std::vector<double> prob = { 0,0,0,0,0,0,1 };
	
	Texture img;

	auto label = max_label(prob);

	while (System::Update()){
		if (DragDrop::HasNewFilePaths()) {
			auto filepath = DragDrop::GetDroppedFilePaths();
			Image tex(filepath[0].path);
			if (tex) {
				std::cout << "Start..." << std::endl;
				img = Texture(tex);
				tex.scale(224,224);
				prob = calpis(tex.dataAsUint8());
				label = max_label(prob);
				std::cout << "End..." << std::endl;
			}
		}
		for (int idx = 0; idx < label_name.size(); idx++) {
			if (label == idx) {
				font(label_name[idx]).draw(1000, idx * 55+300, Palette::Green);
				auto w = font(Format(prob[idx] * 100) + U"%").region().w;
				font(Format(prob[idx] * 100) + U"%").draw(990 - w, idx * 55 + 300, Palette::Green);
			}
			else {
				font(label_name[idx]).draw(1000, idx * 55+300, Palette::Red);
				auto w = font(Format(prob[idx] * 100) + U"%").region().w;
				font(Format(prob[idx] * 100) + U"%").draw(990 - w, idx * 55 + 300, Palette::Red);
			}
		}
		tex_taste[label].resized(300,300).drawAt(1050, 150);
		if(img){
			double scale = 600.0/(img.height()>img.width()?img.height():img.width());
			img.scaled(scale).draw(50,50);
		}
	}
}