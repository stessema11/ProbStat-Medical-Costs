import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_squared_error
import os

floatDataPoints = ['age', 'bmi', 'children', 'charges']
stringDataPoints = ['sex', 'smoker', 'region']

mlModels = [LinearRegression(), Lasso(), ElasticNet()]

pdf = PdfPages('medical_costs.pdf')

COUNTER = 0

def get_distribution_plot(data, x):
	'''
	Generate distrbution of each data point
	'''
	fig = plt.figure(COUNTER)
	increment_counter()
	distPlot = sns.distplot(data[x])
	title_x = get_title(x)
	plt.title('Distirbution of ' + title_x)
	pdf.savefig(fig)
	plt.close()


def get_bar_chart(data, x):
	'''
	Generate count of each data point
	'''
	fig = plt.figure(COUNTER)
	increment_counter()
	plt.hist(data[x])
	plt.xlabel('Count')
	plt.title('Frequency of Person by ' + x)
	pdf.savefig(fig)
	plt.close()


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
		if point != 'charges' and z != point:
			make_scatter_plot(data, point, z = z)

	for point in stringDataPoints:
		if (z == 'smoker' or z == None) and z != point:
			make_violin_plot(data, point, z = z)


def three_axis_plot(data):
	'''
	Generate all individual plots with a colored z-axis for smokers, bmi and age
	'''
	make_plots(data, z = 'smoker')
	make_plots(data, z = 'bmi')
	make_plots(data, z = 'age')


def make_scatter_plot(data, x, y = 'charges', z = None):
	# Create plot, add labels and title
	fig = plt.figure(COUNTER)
	increment_counter()
	if z is not None:
		if z != 'smoker':
			plt.scatter(data[x], data[y], c = data[z])
			cbar = plt.colorbar()
			cbar.ax.set_ylabel(z)
		else:
			smokers = list(data[x][data[z] == 'yes'])
			charges_smokers = list(data[y][data[z] == 'yes'])
			nonsmokers = list(data[x][data[z] == 'no'])
			charges_nonsmokers = list(data[y][data[z] == 'no'])
			plt.scatter(smokers, charges_smokers)
			plt.scatter(nonsmokers, charges_nonsmokers)
			plt.legend(['smoker', 'nonsmoker'])
	else:
		plt.scatter(data[x], data[y])

	title_x = get_title(x)
	plt.title(y.capitalize() + ' vs. ' + title_x)
	plt.xlabel(x)
	plt.ylabel(y)

	pdf.savefig(fig)
	plt.close()


def make_violin_plot(data, x, y = 'charges', z = None):
	fig = plt.figure(COUNTER)
	increment_counter()
	hue = 'smoker' if z else None # add hue if desired
	violinPlot = sns.violinplot(x = x, y = y, data = data, hue = hue) # create plot
	pdf.savefig(fig) # save to pdf
	plt.close() # close plot object


def make_ml_test_comparison_plot(target_train, target_test, train_pred, test_pred, model):
	'''
	Make scatter plot of predicted charges for test data and target data
	'''
	fig = plt.figure(COUNTER)
	increment_counter()
	plt.scatter(train_pred, train_pred - target_train, label = 'Train Data')
	plt.scatter(test_pred, test_pred - target_test, label = 'Test Data')

	# plot title and legend
	title = str(model)
	title = title[:title.index('(')]
	plt.title(title)
	plt.legend()
	pdf.savefig(fig)
	plt.close()


def preprocess(data):
	'''
	Prepare data to be inputted into machine learning algorithms
	'''
	label_encoder = preprocessing.LabelEncoder()
	data['sex'] = label_encoder.fit_transform(data['sex'])
	data['smoker'] = label_encoder.fit_transform(data['smoker'])
	data['region'] = label_encoder.fit_transform(data['region'])

	cols = [col for col in data.columns if col is not 'charges']
	data = data[cols]
	target = data['charges']
	return data, target


def linear_models(model, data_train, target_train, data_test, target_test):
	lm = model.fit(data_train, target_train)

	train_pred = lm.predict(data_train)
	test_pred = lm.predict(data_test)

	rmse = {'train': mean_squared_error(target_train, train_pred)**(1/2), 'test': mean_squared_error(target_test, test_pred)**(1/2)} 

	r2_scores = {'train': r2_score(target_train, train_pred), 'test': r2_score(target_test, test_pred)}

	return train_pred, test_pred, rmse, r2_scores


def get_title(x):
	'''
	Format a value to be put into a plot title
	Inputs:
		x: String to be put into title
	'''
	if x == 'bmi':
		title_x = x.upper()
	elif x == 'children':
		title_x = '# of ' + x.capitalize()
	else:
		title_x = x.capitalize()
	return title_x


def increment_counter():
	'''
	Keeps a count of the figures created
	'''
	global COUNTER
	COUNTER += 1


def main():
	# Read Data
	data = pd.read_csv('insurance.csv')
	# Initial Data Analysis
	for point in floatDataPoints:
		get_distribution_plot(data, point)
	for point in stringDataPoints:
		get_bar_chart(data, point)

	make_plots(data)
	three_axis_plot(data)
	# Machine Learning
	ml_data, target = preprocess(data)
	data_train, data_test, target_train, target_test = train_test_split(ml_data, target, test_size = 0.20, random_state = 0)
	for model in mlModels:
		train_pred, test_pred, rmse, r2_scores = linear_models(model, data_train, target_train, data_test, target_test)
		modelDataTest = pd.DataFrame({'Tailings': test_pred, 'Predicted Charges': test_pred - target_test})
		modelDataTrain = pd.DataFrame({'Tailings': train_pred, 'Predicted Charges': train_pred - target_train})
		make_ml_test_comparison_plot(target_train, target_test, train_pred, test_pred, model)
		print(r2_scores)
		print(rmse)

	# Close pdf
	pdf.close()


if __name__ == '__main__':
	main()