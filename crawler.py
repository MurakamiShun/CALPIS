from icrawler.builtin import GoogleImageCrawler

#taste = ["みかん","マンゴー","パイン","巨峰", "470ml", "北海道"]
taste = ["写真",]
for name in taste:
	crawler = GoogleImageCrawler(storage={"root_dir":name})
	crawler.crawl(keyword="{}".format(name),max_num=600)