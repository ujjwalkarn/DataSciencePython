"""
Created on Wed Sep 09 12:38:16 2015
@author: ujjwal.karn
"""

import pandas as pd                #for handling datasets
import statsmodels.api as sm       #for statistical modeling
import pylab as pl                 #for plotting
import numpy as np                 #for numerical computation

# read the data in
dfTrain = pd.read_csv("C:\\Users\\ujjwal.karn\\Desktop\\Python\\train.csv")
dfTest = pd.read_csv("C:\\Users\\ujjwal.karn\\Desktop\\Python\\test.csv")

# take a look at the dataset
print dfTrain.head()
#   admit  gre   gpa prestige
#0      0  380  3.61     good
#1      1  660  3.67     good
#2      1  800  4.00     best
#3      1  640  3.19       ok
#4      0  520  2.93       ok

print dfTest.head()
#   gre   gpa  prestige
#0  640  3.30  veryGood
#1  660  3.60      good
#2  400  3.15  veryGood
#3  680  3.98  veryGood
#4  220  2.83      good


# summarize the data
print dfTrain.describe()
#            admit         gre         gpa
#count  300.000000  300.000000  300.000000
#mean     0.306667  590.866667    3.386233
#std      0.461880  117.717630    0.374880
#min      0.000000  300.000000    2.260000
#25%      0.000000  515.000000    3.130000
#50%      0.000000  600.000000    3.390000
#75%      1.000000  680.000000    3.642500
#max      1.000000  800.000000    4.000000

# take a look at the standard deviation of each column
print dfTrain.std()
#admit      0.46188
#gre      117.71763
#gpa        0.37488

# frequency table cutting presitge and whether or not someone was admitted
print pd.crosstab(dfTrain['admit'], dfTrain['prestige'], rownames=['dmit'])
#prestige  best  good  ok  veryGood
#admit                             
#0           20    73  47        68
#1           25    19   9        39

#explore data
dfTrain.groupby('admit').mean()
#              gre       gpa
#admit                      
#0      573.461538  3.336587
#1      630.217391  3.498478

# plot one column
dfTrain['gpa'].hist()
pl.title('Histogram of GPA')
pl.xlabel('GPA')
pl.ylabel('Frequency')
pl.show()

# barplot of gre score grouped by admission status (True or False)
pd.crosstab(dfTrain.gre, dfTrain.admit.astype(bool)).plot(kind='bar')
pl.title('GRE score by Admission Status')
pl.xlabel('GRE score')
pl.ylabel('Frequency')
pl.show()

# dummify prestige
dummy_ranks = pd.get_dummies(dfTrain['prestige'], prefix='prestige')
print dummy_ranks.head()
#      prestige_best  prestige_good  prestige_ok  prestige_veryGood
#0              0              1            0                  0
#1              0              1            0                  0
#2              1              0            0                  0
#3              0              0            1                  0
#4              0              0            1                  0

# create a clean data frame for the regression
cols_to_keep = ['admit', 'gre', 'gpa']
data = dfTrain[cols_to_keep].join(dummy_ranks.ix[:, 'prestige_good':])
print data.head()
#     admit  gre   gpa  prestige_good  prestige_ok  prestige_veryGood
#0      0  380  3.61              1            0                  0
#1      1  660  3.67              1            0                  0
#2      1  800  4.00              0            0                  0
#3      1  640  3.19              0            1                  0
#4      0  520  2.93              0            1                  0

# manually add the intercept
data['intercept'] = 1.0

print data.head()

train_cols = data.columns[1:]
print data.columns[1:]
# Index([u'gre', u'gpa', u'prestige_good', u'prestige_ok', u'prestige_veryGood', u'intercept'], dtype='object')

#Logistic Regression
logit = sm.Logit(data['admit'], data[train_cols])

# fit the model
result = logit.fit()
print result.summary()

# recreate the dummy variables
dummy_ranks_test = pd.get_dummies(dfTest['prestige'], prefix='prestige')
print dummy_ranks_test

#create intercept column
dfTest['intercept'] = 1.0

# keep only what we need for making predictions
cols_to_keep = ['gre', 'gpa', 'prestige', 'intercept']
dfTest = dfTest[cols_to_keep].join(dummy_ranks_test.ix[:, 'prestige_good':])

dfTest.head()
# make predictions on the enumerated dataset
dfTest['admit_pred'] = result.predict(dfTest[train_cols])

#see probabilities
print dfTest.head()

#convert probabilities to 'yes' 'no'
dfTest['admit_yn']= np.where(dfTest['admit_pred'] > 0.5,'yes','no')
print dfTest.head()

cols= ['gre', 'gpa', 'admit_yn']
dfTest[cols].groupby('admit_yn').mean()
#                 gre       gpa
#admit_yn                      
#no        556.585366  3.324268
#yes       676.666667  3.750000

cols= ['gre', 'gpa', 'admit_yn']
dfTest[cols].groupby('admit_yn').mean()
#                 gre       gpa
#admit_yn                      
#no        556.585366  3.324268
#yes       676.666667  3.750000

dfTest.to_csv('C:\\Users\\ujjwal.karn\\Desktop\\Python\\output.csv', sep=',')
