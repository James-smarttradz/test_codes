# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 22:16:57 2022

@author: James Ang
"""

import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
print(iris.head())

# print(sns.get_dataset_names())

tips = sns.load_dataset("tips")
print(tips.head())
sns.set_style("whitegrid")
g=sns.lmplot(x='tip',
        y = "total_bill",
        data = tips,
        aspect=2)
g = (g.set_axis_labels("Tip","Total bill(USD")).set(xlim=(0,10),ylim=(0,100))
plt.title("title")
plt.show()

sns.pairplot(iris, hue='species', height=3, aspect=1)