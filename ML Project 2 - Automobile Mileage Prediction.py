#!/usr/bin/env python
# coding: utf-8

# # Objective of Automobile Mileage Prediction
# Build a predictive modeling algorithm to predict mileage of cars based on given input variables

# In[91]:


#import libraries

##for data preparation and data cleaning
import pandas as pd
#for creating plots
import matplotlib.pyplot as plt
#for distribution plot and heatmap
import seaborn as sns
#for creating test and train samples
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[92]:


df=pd.read_csv(r"C:\Users\User\Desktop\Introtallent\Python\automobile mileage prediction\automobile data.csv")


# In[93]:


df.head()


# In[94]:


df.tail()


# In[95]:


df.shape


# In[96]:


df.describe()


# In[97]:


df.dtypes


# In[98]:


#horsepower is a numeric variablebut in the df it is stored as categorical so, we 
#need to change the datatype of numeric
df['Horsepower']=pd.to_numeric(df['Horsepower'],errors='coerce')


# In[99]:


df.dtypes


# In[100]:


df.isnull().sum()  #There are 6missing values


# # Missing value treatment

# In[101]:


df.columns


# In[102]:


df=df.dropna(axis=0)


# In[103]:


df.isnull().sum()


# In[104]:


plt.boxplot(df['MPG'])  # don't have outliers
plt.show()


# In[105]:


plt.boxplot(df['Weight'])  # don't have outliers
plt.show()


# In[106]:



plt.boxplot(df['Horsepower'])
plt.show()


# In[107]:


def remove_outlier(d,c):
    #find q1 and q3
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    
    #iqr
    iqr=q3-q1
    
    #upper bound and lower bound
    ub=q3+1.5*iqr    
    lb=q3-1.5*iqr
    
    
    #filtering data by removing outliers
    good_data=d[(d[c]>lb) & (d[c]<ub)]
    return good_data


# In[108]:


df=remove_outlier(df,'Horsepower')
plt.boxplot(df['Horsepower'])
plt.show()


# In[109]:


df=remove_outlier(df,'Acceleration')
plt.boxplot(df['Acceleration'])
plt.show()


# In[110]:


df.dtypes


# # EDA

# In[112]:


df.columns


# In[113]:


#'MPG', 'Displacement', 'Horsepower', 'Weight', 'Acceleration'


# In[114]:


sns.distplot(df['MPG'])


# In[115]:


sns.distplot(df['Displacement'])


# In[116]:


sns.distplot(df['Horsepower'])


# In[117]:


sns.distplot(df['Weight'])


# In[118]:


sns.distplot(df['Acceleration'])


# In[119]:


#Check datamix
#Cylinders, Model_year, Origin, Car_Name


# In[120]:


df.groupby('Cylinders')['Cylinders'].count().plot(kind='bar')


# In[121]:


df.groupby('Model_year')['Model_year'].count().plot(kind='bar')


# In[122]:


df.groupby('Origin')['Origin'].count().plot(kind='bar')


# In[123]:


df.groupby('Car_Name')['Car_Name'].count().plot(kind='bar')


# # Pearson correlation test

# In[124]:


#create a set of numeric columns
df_numeric=df.select_dtypes(include=['int64','float64'])
df_numeric.head()


# In[125]:


#in df_numeric has categorical variables that we need to drop(Cylinders, Model_year, Origin)


# In[131]:


df_numeric=df_numeric.drop(['Cylinders','Model_year','Origin'],axis=1)
df_numeric.head()


# In[57]:


#create heatmap
sns.heatmap(df_numeric.corr(),cmap='YlGnBu', annot=True)


# In[127]:


df['Cylinders'].unique()


# In[128]:


df['Origin'].unique()


# In[60]:


df['Model_year'].unique()


# In[129]:


df['Car_Name'].unique()


# In[61]:


sns.pairplot(df)


# ----------------------------------------End of EDA------------------------------------------------------

# # Dummy Conversion

# In[62]:


#one-hot-encoding(dummy conversion)


# In[132]:


#remove model_year as it doesn't signify anything in terms of impact on target
df=df.drop('Model_year', axis=1)


# In[133]:


df.dtypes


# In[111]:


df['Origin']=df['Origin'].astype('object')
df['Model_year']=df['Model_year'].astype('object')
df['Cylinders']=df['Cylinders'].astype('object')
df.dtypes


# In[136]:


#create a new df to store categorical variables for dummy conversion
df_categorical=df.select_dtypes(include='object')
df_categorical.head()


# In[135]:


#dummy conversion
df_dummy=pd.get_dummies(df_categorical)
df_dummy.head()


# In[65]:


#create final data by combining df_numeric and df_categorical
df_final=pd.concat([df_numeric,df_dummy], axis=1)
df_final.head()


# In[30]:


df_final.to_excel(r"C:\Users\User\Desktop\Introtallent\Python\automobiledata.xlsx")


# In[137]:


#create x and y
x=df_final.drop('MPG', axis=1)

y=df_final['MPG']

print(x.shape, y.shape)


# In[138]:


xtest, xtrain, ytest, ytrain=train_test_split(x,y, test_size=0.3, random_state=0)


# In[139]:


print(xtrain.shape, xtest.shape, ytest.shape, ytrain.shape)


# # Buil Linear Regression Model

# In[69]:


#Instantiate Linear Regression function
linreg=LinearRegression()

#fit the model (i.e ,train the alogorithm using training sample)
linreg.fit(xtrain, ytrain)


# In[143]:


#check training accuracy
linreg.score(xtrain, ytrain)


# In[144]:


#predict the car_name using xtest
predicted_mpg=linreg.predict(xtest)


# In[145]:


#print predicted price
predicted_mpg


# In[142]:


#check model accuracy using test data
linreg.score(xtest, ytest)


# In[141]:


#print beta-not value
linreg.intercept_


# In[140]:


#print beta-values
linreg.coef_


# In[ ]:





# In[ ]:




