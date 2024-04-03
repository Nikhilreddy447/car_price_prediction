import scrapy
import sys
from src.logger import logging
from src.exception import CustomException

try:

    base_url = "https://api.spinny.com/v3/api/listing/v3/?city={}&product_type=cars&category=used&page={}&show_max_on_assured=true&custom_budget_sort=true&prioritize_filter_listing=true&high_intent_required=true&active_banner=true"
    class CollectorSpider(scrapy.Spider):
        name = "data_collector"
        allowed_domains = ["api.spinny.com"]
        cities = [
            'hyderabad',
            'delhi',
            'bangalore',
            'kolkata',
            'chennai',
            'pune'
        ]
        
        max_pages = 100
        
        def start_requests(self) :
            
            for city in self.cities:
                for page in range(1,self.max_pages+1):
                    url = f"https://api.spinny.com/v3/api/listing/v3/?city={city}&product_type=cars&category=used&page={page}&show_max_on_assured=true&custom_budget_sort=true&prioritize_filter_listing=true&high_intent_required=true&active_banner=true"
                    yield scrapy.Request(url,callback=self.parse)
                    
        
        logging.info("Data Scraping started")

        def parse(self, response):
            data = response.json()
            results = data.get('results',[])
            
            for car in results:
                id=car.get('id')
                city=car.get('city')
                make=car.get('make')
                model=car.get('model')
                body_type=car.get('body_type')
                transmission=car.get('transmission')
                no_of_owners=car.get('no_of_owners')
                color=car.get('color')
                milleage=car.get('round_off_mileage')
                fuel_type= car.get('fuel_type')
                make_year=car.get('make_year')
                price=car.get('price')
                yield{
                    'Id':id,
                    'City':city,
                    'Make':make,
                    'Model':model,
                    'Body_type':body_type,
                    'Transmission':transmission,
                    'No_of_owners':no_of_owners,
                    'color':color,
                    'Milleage':milleage,
                    'Fuel_type':fuel_type,
                    'Make_year':make_year,
                    'Price':price
                }
        logging.info("Scraping Ended")

except Exception as e:
    raise CustomException(e,sys)