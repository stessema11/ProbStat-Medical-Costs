import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

data = pd.read_csv('insurance.csv')

age = data.groupby(by = 'age').size()
sex = data.groupby(by = 'sex').size()
bmi = data.groupby(by = 'bmi').size()
children = data.groupby(by = 'children').size()
smoker = data.groupby(by = 'smoker').size()
region = data.groupby(by = 'region').size()
charges = data.groupby(by = 'charges').size()

floatDataPoints = ['age', 'bmi', 'children', 'charges']
stringDataPoints = ['sex', 'smoker', 'region']

def get_distribution_plot(data, x):
	'''
	Generate distrbution of each data point
	'''
	distPlot = sns.distplot(data[x])
	title_x = get_title(x)
	plt.title('Distirbution of ' + title_x)
	plt.show()


def get_bar_chart(data, x):
	'''
	Generate count of each data point
	'''
	plt.hist(data[x])
	plt.title('Frequency of Person by ' + x)
	plt.show()


def make_plots(data, z = None):
	'''
	Generate plots of charges versus each data point on its own
	Inputs:
	data: from insurance.csv
	z: add a color scale
	Will generate either a scatter plot or violin plot depending on the data type
		Float/Integer: Scatter Plot
		String/Boolean: Violin Plot
	'''
	for point in floatDataPoints:
		if point != 'charges':
			make_scatter_plot(data, point, z = z)

	for point in stringDataPoints:
		if z == 'smoker' or z == None:
			make_violin_plot(data, point, z = z)


def multi_plots(data):
	'''
	Generate all individual plots with a colored z-axis for smokers, bmi and age
	'''
	make_plots(data, z = 'smoker')
	make_plots(data, z = 'bmi')
	make_plots(data, z = 'age')


def make_scatter_plot(data, x, y = 'charges', z = None):
	if z is not None:
		c = []
		if z == 'smoker':
			for i in data[z]:
				if i == 'yes':
					c.append('red')
				else:
					c.append('blue')
		else:
			c = data[z]
	else:
		c = None

	plt.scatter(data[x], data[y], c = c)
	if z is not None and z != 'smoker':
		cbar = plt.colorbar()
		cbar.ax.set_ylabel(z)
	title_x = get_title(x)
	plt.title(y.capitalize() + ' vs. ' + title_x)
	plt.xlabel(x)
	plt.ylabel(y)
	plt.show()


def make_violin_plot(data, x, y = 'charges', z = None):
	mean = data.groupby(by = x)[y].mean()
	hue = 'smoker' if z else None
	violinPlot = sns.violinplot(x = x, y = y, data = data, hue = hue)
	plt.show(violinPlot)


def get_title(x):
	if x == 'bmi':
		title_x = x.upper()
	elif x == 'children':
		title_x = '# of ' + x.capitalize()
	else:
		title_x = x.capitalize()
	return title_x


def main():
	for point in floatDataPoints:
		get_distribution_plot(data, point)
	for point in stringDataPoints:
		get_bar_chart(data, point)

	make_plots(data)
	multi_plots(data)


if __name__ == '__main__':
	main()