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


#加入了mszoing，希望通过encoding来转码
house_predictors = ['MSSubClass','MSZoning','LotArea','OverallQual','OverallCond',
                    'YearBuilt']   #当加入mszoing的时候会出现错误

'''接下来尝试转码'''
size_mapping = {'RL': 1.0,
               'RM': 2.0,
               'C(all)': 3.0,
               'FV': 4.0,
               'RH': 5.0
               }
house_data['MSZoning'] = house_data['MSZoning'].map(size_mapping)
test_data['MSZoning'] = test_data['MSZoning'].map(size_mapping)
#检查转码之后的mszoning列是否有nan值、无穷大值,起初只检查了训练集，现在加上测试集
print(np.isnan(house_data['MSZoning'].any()))
print(np.isfinite(house_data['MSZoning'].all()))     #结果为true，证明有无穷大值
print(np.isnan(test_data['MSZoning'].any()))
print(np.isfinite(test_data['MSZoning'].all())) 

#下面尝试解决无穷数
house_data['MSZoning'][np.isnan(house_data['MSZoning'])] = np.mean(house_data['MSZoning'][~np.isnan(house_data['MSZoning'])])
test_data['MSZoning'][np.isnan(test_data['MSZoning'])] = np.mean(test_data['MSZoning'][~np.isnan(test_data['MSZoning'])])


print(house_data.dtypes.sample())
#下面可能出错,当house_predictors时可以输出4，为one---时输不出2
X = house_data[house_predictors]#没有one_hot_traing_pridictore依然可以运行
print(X['MSZoning'].describe)
print(2)

#导入另一个库
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0) #定义训练级与测试集
print(3)

#导入随机森林
from sklearn.ensemble import RandomForestRegressor

forest_model = RandomForestRegressor()
forest_model.fit(X, y)
print(4)
#使用待测数据
testdata = test_data[house_predictors]  #应该是one--时出错，当house_predictors时正常，程序不会出错
melb_preds = forest_model.predict(testdata)
my_submission = pd.DataFrame({'Id':test_data.Id,'SalePrice':melb_preds})
print(5)

#导入误差判断函数
from sklearn.metrics import mean_absolute_error
house_prices = forest_model.predict(X)
print(mean_absolute_error(y,house_prices))