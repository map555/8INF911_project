import scrapy
import csv
from pathlib import Path
from functools import reduce
import os


class GenerateSpider(scrapy.Spider):
    name = "popular-memes"
    save_path = os.getcwd()

    def start_requests(self):
        urls = [
            'https://api.imgflip.com/popular_meme_ids',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def get_memes(self, response, fieldnames):
        rows = response.css('#page table tr')[1:]
        self.log('rows %s' % rows)
        for row in rows:
            self.log('row %s' % row)
            _row = row.css('td::text').extract()
            self.log('_row %s' % _row)
            meme = {
                fieldnames[0]: _row[0],
                fieldnames[1]: _row[1]
            }
            if len(_row) > 2:
                meme[fieldnames[2]] = _row[2]
            yield meme

    def parse(self, response):
        # ['ID', 'Name', 'Alternate Text']
        fieldnames = response.css('#page table tr th::text').extract()

        filename = reduce(os.path.join, [self.save_path, "dataset", 'popular_100_memes.csv'])
        Path(os.path.dirname(filename)).mkdir(mode=0o655, parents=True, exist_ok=True)

        with open(filename, 'w', newline='') as file:
            self.log('Created File %s' % filename)
            writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
            writer.writeheader()

            for meme in self.get_memes(response, fieldnames):
                writer.writerow(meme)
