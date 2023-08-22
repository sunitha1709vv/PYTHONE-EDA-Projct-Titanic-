#!/usr/bin/env python
# coding: utf-8

# In[49]:


import seaborn as sns  


# In[50]:


data = sns.load_dataset("titanic")
data


# In[51]:


sns.lineplot(x='sex', y='fare', data=data)


# In[52]:


sns.boxplot(x='fare', y='class', data=data)


# In[53]:


sns.scatterplot(x='sex', y='fare', data=data, hue='survived')


# In[54]:


sns.countplot(x='pclass', data=data)


# In[55]:


sns.barplot(x='sex', y='age', data=data)


# starting with meaningfull analysis  

# In[56]:


sns.distplot(data.age)


# distrubution plot is a type of data visualisation that shows the distribution of counitinues variable.
# it provides insights in to the underline data distribution including central tendencies, spread the presence of any outliers .there are several ways for identifying the distribution one such visualisation is ditbn plot.

# In[57]:


data.shape


# Box Plot:
# 
# x: Categorical variable or grouping variable
# y: Continuous variable

# In[58]:


data.columns


# In[59]:


sns.boxplot(x='parch',y='age',data=data , palette= 'rainbow')
#one continuous and one categorical variable


# In[60]:


sns.countplot(x='survived',data=data , palette = 'rainbow')
#data imbalancing


# In[61]:


sns.countplot(x='sex',data=data , palette = 'rainbow')


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


data.corr()


# In[64]:


plt.figure(figsize= [10,12])
sns.heatmap(data.corr(),annot = True)


# heatmap is a visualisation that displace the metrix like representation of the data.where individual values are reprsented as colour.it is commonly used visuaise the corr ,pattern,replationship between two variable ina adata set.heat map s are particulary useful dealing with or when we want to identify the patterns quickly

# In[65]:


sns.histplot(data=data, x='age', bins=20)
plt.title('Age Distribution')
plt.show()


# Histogram
# 
# The histogram is a visual representation of the distribution of a data set.it provides a visual summery of the frequency or count of data values falling in to varieous interval or bins.histogram is usefful understand te underlying the dbtn and identfy the pateerns such as centtral tentencies ,spread,presence of outliers .the x axis represents the range of the values from the data set divided in to equal sized interval or bins.the y axis represents the frequency or count of data values falling into each bin .the hight of each bar corresponds to the numebr of data points falling within the particuar intervals.
# 
# here we have generated a histogram showing the age dbtn of titanic paaseenger with x axiz representing age intervals y axis representing the frequencies (the number of passengers)in each age intervals.
# bins=20 parameters sets the number of the bins or intervals for the histogram.

# In[66]:


sns.barplot(data=data, x='pclass', y='survived', hue='sex')
plt.title('Survival Rate by passenger Class and sex')
plt.show()


# In[67]:


sns.countplot(data=data, x='sex')
plt.title('Passenger Class Distribution based on Males and Females')
plt.show()


# In[68]:


sns.boxplot(data=data, x='pclass', y='fare')
plt.title('Fare Distribution by Passenger Class')
plt.show()


# In[69]:


sns.jointplot(data=data, x='age', y='fare', kind='scatter')
plt.title('Jointplot: Age vs. Fare')
plt.show()


# joint plot 
# 
# it is a type of data visualisation that combines multiple plots to show the retlship between two variables in adata set.it is particulerly useful when we explore the corelation bertween two counitinues variables and the univariate the ditbn of each variable simultanieously.it is mainly used as ascatterplot which displace te ndivisual on one variable on the x axis and the other on the y axis.te scatterplot helps to visualise the pattern of te data point.indicating te replatioship between the two variables.

# In[70]:


sns.jointplot(data=data, x='age', y='fare', kind='reg')
plt.title('Jointplot: Age vs. Fare')
plt.show()


# In[71]:


sns.jointplot(data=data, x='age', y='fare', kind='resid')
plt.title('Jointplot: Age vs. Fare')
plt.show()


# In[72]:


sns.jointplot(data=data, x='age', y='fare', kind='kde')
plt.title('Jointplot: Age vs. Fare')
plt.show()


# In[73]:


sns.jointplot(data=data, x='age', y='fare', kind='hex')
plt.title('Jointplot: Age vs. Fare')
plt.show()


# the kind parameter allows you to specify the type of plot to be used for visualizing the joint relationship between two variables. The available options for the kind parameter are as follows:
# 
# 'scatter' (default): This is the default option and creates a scatter plot with individual data points, showing the joint distribution of two variables.
# 
# 'reg': This option creates a scatter plot with a linear regression line and a 95% confidence interval. It is useful for visualizing the linear relationship between two variables and checking for trends.
# 
# 'resid': This option creates a scatter plot of the residuals after performing linear regression on the data. It is helpful for checking the goodness of fit of the linear regression model.
# 
# 'kde': This option creates a 2D kernel density estimate plot, showing the density of data points in the joint distribution. It can be useful when dealing with large datasets.
# 
# 'hex': This option creates a hexbin plot, which uses hexagonal bins to display the joint distribution. It is useful for visualizing large datasets and identifying regions of high data density.

# In[74]:


plt.figure(figsize= [10,12])
sns.heatmap(data.corr(),annot = True)


# In[75]:


#missing values check
data.isnull().sum()


# In[76]:


#dropping reduntant column from data set
#pclass,fare,class,who,adult_male,deck,embark_town,alive,alone-these column to be droped.
data1=data.drop(["pclass","fare","class","who","adult_male","deck","embark_town","alive","alone"],axis=1,inplace=True)


# In[77]:


data


# In[78]:


data.isnull().sum()


# In[83]:


data


# In[81]:


sns.boxplot(x='parch',y='age',data=data ,palette= 'rainbow')
#one cont and one cat variable


# In[84]:


#hence we have choosed substitute te missing values of age.
parch = data.groupby(data['parch'])
parch.mean()


# In[85]:


def age1(col):
    age=col[0]
    parch=col[1]
    if pd.isnull(age):
        if parch == 0:
            return 32
        elif  parch == 1: 
            return 24
        elif  parch == 2: 
            return 17
        elif  parch == 3: 
            return 33
        elif  parch == 4:
            return 44
        elif parch == 5:
            return 39
        else:
            return 43
    else:
        return age


# In[87]:


import pandas as pd


# In[89]:


data['age'] = data[['age','parch']].apply(age1, axis=1)


# In[90]:


data


# In[91]:


data.isnull().sum()


# In[92]:


data.dropna(inplace=True)


# In[93]:


data.isnull().sum()


# In[94]:


data


# In[95]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
gender =  data['sex']
gender1= le.fit_transform(gender)


# In[96]:


gender1


# In[97]:


genderdf =  pd.DataFrame(gender1 , columns = ['gender'])
genderdf


# In[98]:


embarked = data['embarked']
embarked1 = le.fit_transform(embarked)


from sklearn.preprocessing import OneHotEncoder 
binary = OneHotEncoder(categories='auto')
cols = binary.fit_transform(embarked1.reshape(-1,1))
matrix = cols.toarray()


emb_df = pd.DataFrame(matrix , columns = ['C','Q','S'])



# In[99]:


emb_df


# In[100]:


titanicnewdata = pd.concat([data , genderdf , emb_df] , axis =1 )
titanicnewdata.head()


# In[102]:


titanicnewdata.drop(['sex' , 'embarked','S'], axis=1 , inplace = True)


# In[103]:


titanicnewdata.dropna(inplace=True)


# In[104]:


titanicnewdata.isnull().sum()


# In[105]:


titanicnewdata


# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# gender =  data['sex']
# gender1= le.fit_transform(gender)
# 
# step 1 
# the first line of the code aove imports te  LabelEncoder class from the scikit learn librarory ,which is used for encoding categorical features into numerical values .
# 
# step 2 
# this line creates an intances of the label encoded class and stores it in the variabl le = LabelEncoder()
# 
# step 3
# 
# this line gender1 = le.fit_transform(gender): This line uses the fit_transform method of the LabelEncoder to transform the 'gender' data from categorical labels to numerical values. The fit_transform method first fits the encoder to the data and then transforms it. The transformed numerical values are stored in the variable gender1.
# 
# the 'gender1' variable will contain numerical representations of the original gender data. The 'male' category will be encoded as one integer value, and the 'female' category will be encoded as another integer value. The specific integer values assigned to each category will depend on the alphabetical order of the categories, i.e., 'female' might be encoded as 0 and 'male' as 1, or vice versa.
# 

# 
# Missing value imputation n regression model .
# it is a prrocess or replacing values in a data set with estimated or imputed values.in regression models missing data can lead to biased or inefficient paramtres estimate and reduced predictive accuracy.therefore imputation techniques are commonly used for handling missing values before fiti
# 
# ng regression models.
# 
# 1.Mean, Median, or Mode Imputation
# 
# Replace missing values with the mean, median, or mode of the non-missing values for the respective variable.
# 
# 2.regression imputation we use oter variables in the dataset to predict the missing values through regression modeling.
# 
# for example we can fit a regression model using the non missing values of the variable with missing data as the dependent variable and other variables as predicted then we use the model predict the missing values 
# 
# 3.K-Nearest (KNN) Imputation:
# 
# identifying K-Nearest data points with  complte infn to the obsvn with missing values.then we take the average or the weighted average  of the values from K-Nearest neighboures as the imputed values.
# 
# 4  <h3 style="color:purple">multiple imputation</h3> -we geerate mutple imputation by simultaneousely simulating the missing values multiple times.each imputed dataset is then used to fit separate regression models and results are compbined using specific rules to obtain unbaised estimetes or appropriate standared errors.
# 
# 5 <p><strong><u>time series imputation</u></strong></p>
# 
# for time series data we use interpolation methods to fill in missing values based on th e values of neighbouring time points 
# 
# 
# 
# 
# 
# 
# 
# 

# In[1]:


get_ipython().system('pip install markdown')


# <p><strong><u>time series imputation</u></strong></p>

# steps 
# 
# 1.check the data column to understand each data column meaning 
# 2.check visualisation and drop non important columns from the data ,which does not impact the dependeant or target variable
# 3.check if missing values are there or not 
# 4 we will now treat the missing values here we have reconstruct te data to findout which column related to my missing values related column age
# 5.we will treat te missing values accordingly
# 6.we will ensure there is n further missing values in the dataset
# 7.we have done encoding to change each categorical variable to numerical.
# 8.we will check any other reduntant columns are there or not.
# 9.then our data preprocessing is done nd the data is ready to build the model
# 10.then we fit te model which logistic regression as our target variable as survived becs survive column  has only two values 0 and 1
# 11.we will check how our model has performed and as predicted the future values for survive column.
