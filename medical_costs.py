import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

data = pd.read_csv("insurance.csv")

age = data.groupby(by = 'age').size()
sex = data.groupby(by = 'sex').size()
bmi = data.groupby(by = 'bmi').size()
children = data.groupby(by = 'children').size()
smoker = data.groupby(by = 'smoker').size()
region = data.groupby(by = 'region').size()
charges = data.groupby(by = 'charges').size()

distPlot = sns.distplot(data['charges'])
plt.title("Distirbution of Charges")
plt.show(distPlot)


def individual_plots(data):
	'''
	Generate plots of charges versus each data point on its own
	'''
	g = sns.jointplot(x = "age", y = "charges", data=data)
	jointPlot1 = g.plot_joint(plt.scatter)
	plt.show(jointPlot1)

	meanGender = data.groupby(by = "sex")["charges"].mean()
	print(meanGender)
	print(meanGender["male"] - meanGender["female"])
	boxPlot1 = sns.violinplot(x = "sex", y = "charges", data = data)
	plt.show(boxPlot1)

	g2 = sns.jointplot(x = "bmi", y = "charges", data=data)
	jointPlot2 = g2.plot_joint(plt.scatter)
	plt.show(jointPlot2)

	g3 = sns.jointplot(x = "children", y="charges", data=data)
	jointPlot3 = g3.plot_joint(plt.scatter)
	plt.show(jointPlot3)

	meanSmoker = data.groupby(by = "smoker")["charges"].mean()
	print(meanSmoker)
	print(meanSmoker["yes"] - meanSmoker["no"])
	boxPlot2 = sns.violinplot(x = "smoker", y = "charges", data = data)
	plt.show(boxPlot2)

	meanRegion = data.groupby(by = "region")["charges"].mean()
	print(meanRegion)
	boxPlot3 = sns.violinplot(x = "region", y = "charges", data = data)
	plt.show(boxPlot3)

individual_plots(data)
