import json
import requests

headers = {"Authorization": "Bearer fe1f7c27-65de-45c5-84d1-dfb12f4a2475"}

response = requests.get("https://office.smarttradzt.com:8001/buy-ecommerce-service/product/search?_page=0&_pageSize=10", headers=headers)

data = response.json()

#%% Import json data
# file = 'products.json'
# with open(file,'r') as f:
#     data = json.load(f)
