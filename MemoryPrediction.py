import pandas as pd
import scipy as sci
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
from sklearn.linear_model import LinearRegression

#reading dataset
dataset = pd.read_csv(r"H:\DSci\gpusdataset7.csv")

#extracting year from Release_Date
dataset['Release_Date']=pd.to_datetime(dataset['Release_Date'])
dataset['Release_Year']=dataset['Release_Date'].dt.year
plt.figure(figsize=(16,8))
#Plotting Release Year and Number of GPUs
sns.countplot(x="Release_Year", data=dataset)
plt.title('Grouping GPUs by Release Year', fontsize=20)
plt.ylabel('Number of GPUs', fontsize=15)
plt.xlabel('Release Year', fontsize=15)

plt.figure(figsize=(16,10))
sns.set_style("whitegrid")
plt.title('GPU Memory vs Year of Release', fontsize=20)
plt.xlabel('Year of Release', fontsize=15)
plt.ylabel('GPU Memory', fontsize=15)

years = dataset["Release_Year"].values
memory = dataset["Memory"].values
#Scatter plot for release year and memory
plt.scatter(years, memory, edgecolors='black')
#plt.show()

#extracting integer from Memory object in dataset
dataset['Memory'] = dataset['Memory'].str[:-3].fillna(0).astype(int)
# Array for holding unique release year values
year_arr = dataset.sort_values("Release_Year")['Release_Year'].unique()
#Array for holding mean values of GPUs memory for each year
memory_mean = dataset.groupby('Release_Year')['Memory'].mean().values
#Array for holding median values of GPUs memory for each year
memory_median = dataset.groupby('Release_Year')['Memory'].median().values

plt.figure(figsize=(16,8))
plt.title('GPU Memory vs Year of Release', fontsize=20, fontweight='bold')
plt.xlabel('Year of Release', fontsize=15)
plt.ylabel('GPU Memory', fontsize=15)

#Plot for year array and memory mean array
plt.plot(year_arr, memory_mean, label="Mean")
#Plot for year array and memory median array
plt.plot(year_arr, memory_median, label="Median")
plt.legend(loc=4, prop={'size': 12}, facecolor="white", edgecolor="black")
#show the plots
plt.show()
#Generate a new feature matrix consisting of all polynomial combinations of the features
poly_features = PolynomialFeatures()
resh = year_arr.reshape(-1, 1)
poly_fit = poly_features.fit_transform(resh)
#using linear regression as model for polynomial features (polynomial regression)
linear = LinearRegression()
linear.fit(poly_fit , memory_mean)
#predict using linear regressiong after fitting and transforming polynomial feauteres
lin_fit=poly_features.fit(year_arr.reshape(-1, 1))
prediction= linear.predict(lin_fit)
#calculating the accuracy
score = r2_score(prediction, memory_mean)
print("Accuracy= " + str(round(score, 3)))
#exponential curve for prediction of gpu size
def Exponential_Curve(x, a, b, c):
    return a*2**((x-c)*b)
#using curve fit to adjust the numerical values for the model so that it most closely matches some data
popt, pcov = sci.optimize.curve_fit(Exponential_Curve,  year_arr, memory_mean,  p0=(2, 0.5, 1998))
#input the year
yearint=input("Type future year: ")
memoryyear = Exponential_Curve(int(yearint), *popt)
print("Predicted size of GPU memory in ",yearint, "is " + str(round(int(memoryyear) / 1024, 2)) + " GB.")
