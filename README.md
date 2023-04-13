# Automobile-Mileage-Prediction
Objective of the project was to study the data of given cars, understand pattern and predict the mileage of cars based on input variables.

Objective of the project was to study the data of given cars, understand pattern and predict the mileage of cars based on input variables.

The data sample I used for this study had 12 variables and 400+ observation.

I used Linear Regression technique to work on this problem.

I started my work with importing the required libraries. I imported pandas library for data cleaning and preparation, matplotlib and seaborn for plotting the data, sklearn for random sampling and model building.

I imported the data from excel file to python environment, checked the sample, rows, columns, descriptive statistics, and data types.

In this data Horsepower column was stored an object though it is a numeric column. So I did the type conversion using pd.to_numeric().

I performed missing value check and found missing values in Horsepower column. I did missing value imputation to eliminate the missing values from the data.

I plotted boxplots to test outliers and found outliers in acceleration column.

I wrote a user defined function and applied it on the columns to treat outliers.

Then I performed EDA to check data quality, data-mix, and correlations. 

In this data we had object variables as well, so I did one-hot encoding (dummy conversion) to transform the data from object to numeric.

Then I did split the data into x and y and created training and test samples.

After that I used the training samples to fit the model and predicted mileage using test sample. 

My model gave an accuracy (adjusted R-squared value) of 0.86
