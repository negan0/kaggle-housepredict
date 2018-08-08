# -*- coding: utf-8 -*-
"""
Created on Sun Jul 22 12:46:00 2018

@author: wangw
"""

import pandas as pd
import numpy as np
#读取数据
path = 'C://users/wangw/downloads/kaggle/housepredict20180722/'
house_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')

#定义目标变量和转换因子
y = house_data.SalePrice

#加入了mszoing，street,Alley,这三者均是非数值型数据，需进行encoding
house_predictors = ['MSSubClass','MSZoning','LotArea','Street','Alley','OverallQual','OverallCond',
                    'YearBuilt','KitchenAbvGr','BedroomAbvGr','FullBath','HalfBath','TotRmsAbvGrd',
                    'Fireplaces','WoodDeckSF','OpenPorchSF','ScreenPorch','PoolArea','MiscVal']
print(np.isnan(house_data['KitchenAbvGr'].any()))
#接下来尝试encoding
size_mapping1 = {'RL': 1.0,
               'RM': 2.0,
               'C(all)': 3.0,
               'FV': 4.0,
               'RH': 5.0
               }
size_mapping2 = {'Pave':1.0,
                 'Grvl':2.0
                 }
size_mapping3 = {'NA':1.0,
                 'Grvl':2.0,
                 'Pave':3.0
                 }
house_data['MSZoning'] = house_data['MSZoning'].map(size_mapping1)
test_data['MSZoning'] = test_data['MSZoning'].map(size_mapping1)
house_data['Street'] = house_data['Street'].map(size_mapping2)
test_data['Street'] = test_data['Street'].map(size_mapping2)
house_data['Alley'] = house_data['Alley'].map(size_mapping3)
test_data['Alley'] = test_data['Alley'].map(size_mapping3)

#检查转码之后的mszoning列是否有nan值、无穷大值,起初只检查了训练集，现在加上测试集
print(np.isnan(house_data['MSZoning'].any()))
print(np.isfinite(house_data['MSZoning'].all()))     #结果为true，证明有无穷大值
print(np.isnan(test_data['MSZoning'].any()))
print(np.isfinite(test_data['MSZoning'].all())) 

print(np.isnan(house_data['Street'].any()))
print(np.isfinite(house_data['Street'].all()))
print(np.isnan(test_data['Street'].any()))
print(np.isfinite(test_data['Street'].all()))

#下面尝试解决无穷数
house_data['MSZoning'][np.isnan(house_data['MSZoning'])] = np.mean(house_data['MSZoning'][~np.isnan(house_data['MSZoning'])])
test_data['MSZoning'][np.isnan(test_data['MSZoning'])] = np.mean(test_data['MSZoning'][~np.isnan(test_data['MSZoning'])])

house_data['Street'][np.isnan(house_data['Street'])] = np.mean(house_data['Street'][~np.isnan(house_data['Street'])])
test_data['Street'][np.isnan(test_data['Street'])] = np.mean(test_data['Street'][~np.isnan(test_data['Street'])])

house_data['Alley'][np.isnan(house_data['Alley'])] = np.mean(house_data['Alley'][~np.isnan(house_data['Alley'])])
test_data['Alley'][np.isnan(test_data['Alley'])] = np.mean(test_data['Alley'][~np.isnan(test_data['Alley'])])



X = house_data[house_predictors]
print(2)

#导入另一个库,但在这个代码中这几行完全没有任何作用，但并不影响程序执行
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0) #定义训练级与测试集
print(3)

#导入随机森林
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(X, y)
print(4)

#进行预测
testdata = test_data[house_predictors]  
melb_preds = forest_model.predict(testdata)
my_submission = pd.DataFrame({'Id':test_data.Id,'SalePrice':melb_preds})
my_submission.to_csv('my_submission0808.csv')
print(5)

#导入误差判断函数
from sklearn.metrics import mean_absolute_error
house_prices = forest_model.predict(X)
print(mean_absolute_error(y,house_prices))