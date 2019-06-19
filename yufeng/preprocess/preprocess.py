import pandas as pd

data = pd.read_excel("electronics.xlsx")
data = data[['商品描述', '品类']]
data = data.rename(columns = {'商品描述':'title', "品类":'type'})
data.to_csv('electronics.csv', index=False)