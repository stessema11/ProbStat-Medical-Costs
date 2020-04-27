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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from scipy import stats
import os
import json

floatDataPoints = ['age', 'bmi', 'children', 'charges']
stringDataPoints = ['sex', 'smoker', 'region']

# Estimators that will be used
estimators = [LinearRegression(), Lasso(), ElasticNet(alpha = 0.01, l1_ratio = 0.9, max_iter = 20), Ridge(), RandomForestRegressor(n_estimators = 100, criterion = 'mse', random_state = 1, n_jobs = -1)]

# Open pdf to be generated
pdf = PdfPages('medical_costs.pdf')

# A counter for plot figures
COUNTER = 0

def get_distribution_plot(data, x):
    '''
    Generate distrbution of a data point
    Input:
        data: Pandas DataFrame of medical_costs
        x: data point
    '''
    fig = plt.figure(COUNTER)
    increment_counter()
    distPlot = sns.distplot(data[x])
    title_x = get_title(x)
    plt.title('Distribution of ' + title_x)
    pdf.savefig(fig)
    plt.close()


def get_bar_chart(data, x):
    '''
    Generate count of a data point
    Input:
        data: Pandas DataFrame of medical_costs
        x: data point
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
    Generate plots with a colored z-axis for smokers, bmi and age
    Input:
        data: Pandas DataFrame of medical_costs
    '''
    make_plots(data, z = 'smoker')
    make_plots(data, z = 'bmi')
    make_plots(data, z = 'age')


def make_scatter_plot(data, x, y = 'charges', z = None):
    '''
    Create scatter plot, default is charges vs. x
    Inputs:
        data: Pandas Dataframe of medical cost data
        x: String, x on plot
        y: String, y on plot, default charges
        z: color scale of plot
    '''
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
    '''
    Create violin plot, default is charges vs. x
    Inputs:
        data: Pandas Dataframe of medical cost data
        x: String, x on plot
        y: String, y on plot, default charges
        z: color scale of plot
    '''
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
    Input:
        data: Pandas DataFrame of medical_costs
    '''
    scaleMinMax = preprocessing.MinMaxScaler()
    data[["age", "bmi", "children"]] = scaleMinMax.fit_transform(data[["age", "bmi", "children"]])
    data = pd.get_dummies(data, prefix = ["sex", "smoker", "region"])
    ## retain sex = male, smoker = yes, and remove 1 region = northeast to avoid dummytrap
    data = data.drop(data.columns[[4,6,11]], axis = 1)

    target = data['charges']
    data = data.drop(data.columns[[3]], axis = 1)

    return data, target


def linear_models(model, data_train, data_test, target_train, target_test):
    '''
    Fit training data to model then check accuracy with the test data.
    Return: tuple of predictions for the train and test data, along with root-mean-square error and r^2 scores
    '''
    lm = model.fit(data_train, target_train)
    train_pred = lm.predict(data_train)
    test_pred = lm.predict(data_test)

    rmse = {'rmse': {'model': str(model)[:str(model).index('(')], 'train': mean_squared_error(target_train, train_pred)**(1/2), 'test': mean_squared_error(target_test, test_pred)**(1/2)}}

    r2_scores = {'r2_scores': {'model': str(model)[:str(model).index('(')], 'train': r2_score(target_train, train_pred), 'test': r2_score(target_test, test_pred)}}

    return train_pred, test_pred, rmse, r2_scores


def get_linear_summary(data_train, target_train):
    '''
    Get summary of the data
    '''
    X_train = sm.add_constant(data_train)
    linearModel = sm.OLS(target_train, X_train)
    linear = linearModel.fit()
    generate_text_page_pdf(linear.summary(), 8, 0.125)


def generate_text_page_pdf(txt, size, bottom):
    '''
    format text to be saved to pdf
    '''
    fig = plt.figure()
    fig.clf()
    fig.text(0.0625, bottom, txt, transform=fig.transFigure, size=size, ha="left")
    pdf.savefig()
    plt.close()


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

    txt_1 = 'Now that we understand the\ndata in the dataset, lets look at\ncharges vs. each data point to see\nwhich have an effect'
    generate_text_page_pdf(txt_1, 24, 0.625)

    make_plots(data)

    txt_2 = 'It is clear from these plots\nthat smokers, people with BMI\nover 30, and old people have\nhigher individual medical costs.\n'
    txt_3 = 'Let us now look at the other\nplots with these data points\nhighlighted to see what their\neffect is on charges'
    generate_text_page_pdf(txt_2 + txt_3, 24, 0.3)
    three_axis_plot(data)

    # Machine Learning
    ml_data, target = preprocess(data)
    data_train, data_test, target_train, target_test = train_test_split(ml_data, target, test_size = 0.20, random_state = 0)

    get_linear_summary(data_train, target_train)

    results = []

    for estimator in estimators:
        train_pred, test_pred, rmse, r2_scores = linear_models(estimator, data_train, data_test, target_train, target_test)
        modelDataTest = pd.DataFrame({'Tailings': test_pred, 'Predicted Charges': test_pred - target_test})
        modelDataTrain = pd.DataFrame({'Tailings': train_pred, 'Predicted Charges': train_pred - target_train})
        make_ml_test_comparison_plot(target_train, target_test, train_pred, test_pred, estimator)
        results.append([rmse, r2_scores])

    txt_4 = ''
    for i, j in results:
        txt_4 = txt_4 + json.dumps(i) + '\n' + json.dumps(j) + '\n'
    txt_4 = txt_4 + '\n\nThe Random Forest Regression is best suited for this dataset by far. Linear\nregression is slightly favored over the other regression models. This makes sense\nbecuase the other models are more robust the more features there are'
    generate_text_page_pdf(txt_4, 6, 0.625)

    # Close pdf
    pdf.close()


if __name__ == '__main__':
    main()