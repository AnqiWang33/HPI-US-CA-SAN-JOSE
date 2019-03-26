
# Relationship of Annual House Price Index for the United States, California and San Jose-Sunnyvale -Santa Clara Region

### Data Sources:
 - U.S. Federal Housing Finance Agency, All-Transactions House Price Index for the United States [USSTHPI], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/USSTHPI, March 26, 2019.
 - U.S. Federal Housing Finance Agency, All-Transactions House Price Index for California [CASTHPI], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/CASTHPI, March 26, 2019.
 - U.S. Federal Housing Finance Agency, All-Transactions House Price Index for San Jose-Sunnyvale-Santa Clara, CA (MSA) [ATNHPIUS41940Q], retrieved from FRED, Federal Reserve Bank of St. Louis; https://fred.stlouisfed.org/series/ATNHPIUS41940Q, March 26, 2019.

### Data Info:

| Region | the United States | California  | San Jose-Sunnyvale-Santa Clara |
| :---: | :----:| :---: | :---: |
| Unit  | Index 1980:Q1=100, Not Seasonally Adjusted | Index 1980:Q1=100, Not Seasonally Adjusted | Index 1995:Q1=100, Not Seasonally Adjusted|
| Frequency | Quarterly | Quarterly | Quarterly |
| Index | HPI | HPI | HPI|
| House Style | Single Family House | Single Family House | Single Family House |
| Date Range | 1975Q1 - 2018Q4 | 1975Q1 - 2018Q4 | 1975Q4 - 2018Q4 |

## Research Question:
## Is There a Linear Regression Relationship among Annual House Price Index for the United States, California and San Jose-Sunnyvale -Santa Clara Region?

### Step 1:
 Import numpy, pandas, matplotlib.pyplot, etc.


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import statsmodels
import statsmodels.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
```

### Step 2:
load the three csv.files to jupyter notebook.  
Use pd.read_csv()  
We can use DataFrame.head() to check the first 5 rows of a dataframe.


```python
nhpi = pd.read_csv('All-Transactions House Price Index for the United States.csv')
nhpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>USSTHPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975-01-01</td>
      <td>59.83</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1975-04-01</td>
      <td>61.13</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1975-07-01</td>
      <td>61.25</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1975-10-01</td>
      <td>62.35</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-01-01</td>
      <td>62.85</td>
    </tr>
  </tbody>
</table>
</div>




```python
cahpi = pd.read_csv('All-Transactions House Price Index for California.csv')
cahpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>CASTHPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975-01-01</td>
      <td>41.61</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1975-04-01</td>
      <td>42.76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1975-07-01</td>
      <td>44.24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1975-10-01</td>
      <td>45.84</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-01-01</td>
      <td>47.73</td>
    </tr>
  </tbody>
</table>
</div>




```python
ssshpi = pd.read_csv('All-Transactions House Price Index for San Jose.csv')
ssshpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>ATNHPIUS41940Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975-10-01</td>
      <td>18.85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-01-01</td>
      <td>19.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-04-01</td>
      <td>20.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-07-01</td>
      <td>21.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-10-01</td>
      <td>23.02</td>
    </tr>
  </tbody>
</table>
</div>



### Step 3:
Since we want to find out the relationship of HPI in these three files, we use meger() to join them together.  
We merge them on 'DATE' column.


```python
multi_hpi = nhpi.merge(cahpi, on = 'DATE').merge(ssshpi, on ='DATE')
multi_hpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>USSTHPI</th>
      <th>CASTHPI</th>
      <th>ATNHPIUS41940Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975-10-01</td>
      <td>62.35</td>
      <td>45.84</td>
      <td>18.85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976-01-01</td>
      <td>62.85</td>
      <td>47.73</td>
      <td>19.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1976-04-01</td>
      <td>65.51</td>
      <td>50.22</td>
      <td>20.87</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1976-07-01</td>
      <td>66.57</td>
      <td>53.54</td>
      <td>21.92</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1976-10-01</td>
      <td>67.26</td>
      <td>55.47</td>
      <td>23.02</td>
    </tr>
  </tbody>
</table>
</div>



### Step 4:
We want to know the relationship annually instead of quaterly.
Group data by using groupby() function.  
Here we group data by the first 4 characters in 'DATE' column and calculate the mean of each group, calling mean() at the same time.  
Then, we could get a new data frame 'mean_hpi2' grouped by year with means in each cell.


```python
mean_hpi = multi_hpi.groupby(multi_hpi.DATE.str[0:4]).mean().reset_index()
mean_hpi.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>DATE</th>
      <th>USSTHPI</th>
      <th>CASTHPI</th>
      <th>ATNHPIUS41940Q</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975</td>
      <td>62.3500</td>
      <td>45.8400</td>
      <td>18.850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976</td>
      <td>65.5475</td>
      <td>51.7400</td>
      <td>21.395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1977</td>
      <td>73.4125</td>
      <td>64.8675</td>
      <td>27.560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1978</td>
      <td>83.7450</td>
      <td>77.3400</td>
      <td>31.750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1979</td>
      <td>95.1050</td>
      <td>90.7050</td>
      <td>37.255</td>
    </tr>
  </tbody>
</table>
</div>



### Step 5:
The column names in the new data frame is hard to understand, we can rename these columns.  
Pyhthon has rename() function for us.  
We can change columns nanes together by creating a dictionary.


```python
mean_hpi2 = mean_hpi.rename(columns = {'DATE':'YEAR', 'USSTHPI':'National HPI','CASTHPI':'California HPI','ATNHPIUS41940Q':'San Jose-Sunnyvale-Santa Clara HPI'})
mean_hpi2.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>YEAR</th>
      <th>National HPI</th>
      <th>California HPI</th>
      <th>San Jose-Sunnyvale-Santa Clara HPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1975</td>
      <td>62.3500</td>
      <td>45.8400</td>
      <td>18.850</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1976</td>
      <td>65.5475</td>
      <td>51.7400</td>
      <td>21.395</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1977</td>
      <td>73.4125</td>
      <td>64.8675</td>
      <td>27.560</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1978</td>
      <td>83.7450</td>
      <td>77.3400</td>
      <td>31.750</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1979</td>
      <td>95.1050</td>
      <td>90.7050</td>
      <td>37.255</td>
    </tr>
  </tbody>
</table>
</div>



### Step 6:
Check info of data frame 'mean_hpi2'.  
Get some statistical measurements using 'mean_hpi2.describe()'.


```python
mean_hpi2.info()
mean_hpi2.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 44 entries, 0 to 43
    Data columns (total 4 columns):
    YEAR                                  44 non-null object
    National HPI                          44 non-null float64
    California HPI                        44 non-null float64
    San Jose-Sunnyvale-Santa Clara HPI    44 non-null float64
    dtypes: float64(3), object(1)
    memory usage: 1.5+ KB





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>National HPI</th>
      <th>California HPI</th>
      <th>San Jose-Sunnyvale-Santa Clara HPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>44.000000</td>
      <td>44.000000</td>
      <td>44.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>223.350114</td>
      <td>292.338807</td>
      <td>165.535227</td>
    </tr>
    <tr>
      <th>std</th>
      <td>106.204233</td>
      <td>181.432888</td>
      <td>117.000256</td>
    </tr>
    <tr>
      <th>min</th>
      <td>62.350000</td>
      <td>45.840000</td>
      <td>18.850000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>133.428125</td>
      <td>134.401875</td>
      <td>62.500000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>196.653750</td>
      <td>226.595000</td>
      <td>113.891250</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>321.802500</td>
      <td>424.371250</td>
      <td>240.284375</td>
    </tr>
    <tr>
      <th>max</th>
      <td>425.085000</td>
      <td>656.820000</td>
      <td>452.202500</td>
    </tr>
  </tbody>
</table>
</div>



> Conclusion:
> - Every column has 44 values.  
> - National average HPI is higher than that for San Jose-Sunnyvale-Santa Clara, but lower than California HPI. 

### Step7 :
Before predictive analysis, we need to makesure there is no outlier in our data frame.  
We could use boxplots to examine outliers.  
Here we use seaborn to draw boxplot with 3 vairables.


```python

```


```python

```


```python
sns.boxplot(data=mean_hpi2.ix[:,1:4])
#sns.plt.show()
```

    /Users/liurongxuan/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:1: DeprecationWarning: 
    .ix is deprecated. Please use
    .loc for label based indexing or
    .iloc for positional indexing
    
    See the documentation here:
    http://pandas.pydata.org/pandas-docs/stable/indexing.html#ix-indexer-is-deprecated
      """Entry point for launching an IPython kernel.





    <matplotlib.axes._subplots.AxesSubplot at 0x1c22b6feb8>




![png](output_18_2.png)


>Conclusion:  
> From the boxplot we could conclude that there is no outlier in each column.   
> California HPI data fluctuate greater than the other two.   
> All these three are skewed to right.

### Step 8:
Draw a line plot.   
x-axis = 'YEAR'  
y-axis = HPI value


```python
plt.plot(mean_hpi2['YEAR'], mean_hpi2['National HPI'], label = 'National')
plt.plot(mean_hpi2['YEAR'], mean_hpi2['California HPI'], label = 'California')
plt.plot(mean_hpi2['YEAR'], mean_hpi2['San Jose-Sunnyvale-Santa Clara HPI'], label = 'San Jose-Sunnyvale-Santa Clara')
plt.legend(loc = 'upper left')
pl.xticks(rotation = 45)
plt.title('All-transactions House Price Index', fontsize = 16, fontweight = 'bold')
plt.suptitle('National, California, San Jose-Sunnyvale-Santa Clara', fontsize = 10)
fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 4
print(fig_size)
plt.rcParams["figure.figsize"] = fig_size
plt.grid(True)
plt.show()
```

    [15, 4]



![png](output_21_1.png)


> Conclusion   
> - From the line plot, we could see that HPI in California is always higher than the national form 1975 to 2018.  
> - HPI for the US and San Jose region has a more similar tendency.  
> - As a whole, HPI is increasing.

### Step 9：
we want to know the distribution of data check the relationship of every two variables.  
pairplot() in seaborn help us getting the outcome we want.  
Based on the line plot we created above, suppose they have linear regression. Thus, we set kind = 'reg'.    
Additionally, the values in first column in mean_hpi2 are not numeric values, we use .iloc[] to choose the right column.


```python
sns.pairplot(mean_hpi2.iloc[:,1:4], kind = 'reg')
```




    <seaborn.axisgrid.PairGrid at 0x1c22c3bb38>




![png](output_24_1.png)


> Conclusion  
> - The scatters are almost around the straight line.
> - Every two variables follow positive linear regression.

### Step 10:
To make the coefficient of determination more clear, we could create a heatmap.   
First, we call .corr() to get the coefficient of determination.
Next, Use sns.heatmap().  
We could set the color, the shape of heatmap in the parenthesis.


```python
corr = mean_hpi.corr()
sns.heatmap(corr, cmap='GnBu_r', square=True, annot=True)
plt.savefig('heatmap.png')
```


![png](output_27_0.png)


### Step 11:
Since we have already know the linear relationship, we want to create a predictive model of every two of them.   
We have already import statsmodels.formula.api as smf.   
We expect to create OLS model of one independent variable and one dependent vaiable. Thus, we use y~x.  
The code is shown below.


```python
a = mean_hpi2['National HPI']
b = mean_hpi2['California HPI']
c = mean_hpi2['San Jose-Sunnyvale-Santa Clara HPI']
linear_1 = smf.ols('a~b', data = mean_hpi2).fit()
linear_2 = smf.ols('a~c', data = mean_hpi2).fit()
linear_3 = smf.ols('b~c', data = mean_hpi2).fit()
print(linear_1.params)
print(linear_2.params)
print(linear_3.params)
print(linear_1.summary())
print(linear_2.summary())
print(linear_3.summary())
```

    Intercept    55.531722
    b             0.574054
    dtype: float64
    Intercept    76.271950
    c             0.888501
    dtype: float64
    Intercept    41.584937
    c             1.514807
    dtype: float64
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      a   R-squared:                       0.962
    Model:                            OLS   Adj. R-squared:                  0.961
    Method:                 Least Squares   F-statistic:                     1056.
    Date:                Tue, 26 Mar 2019   Prob (F-statistic):           2.16e-31
    Time:                        14:02:01   Log-Likelihood:                -195.41
    No. Observations:                  44   AIC:                             394.8
    Df Residuals:                      42   BIC:                             398.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     55.5317      6.060      9.164      0.000      43.302      67.761
    b              0.5741      0.018     32.489      0.000       0.538       0.610
    ==============================================================================
    Omnibus:                        0.106   Durbin-Watson:                   0.330
    Prob(Omnibus):                  0.948   Jarque-Bera (JB):                0.271
    Skew:                          -0.094   Prob(JB):                        0.873
    Kurtosis:                       2.665   Cond. No.                         656.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      a   R-squared:                       0.958
    Model:                            OLS   Adj. R-squared:                  0.957
    Method:                 Least Squares   F-statistic:                     960.1
    Date:                Tue, 26 Mar 2019   Prob (F-statistic):           1.46e-30
    Time:                        14:02:01   Log-Likelihood:                -197.42
    No. Observations:                  44   AIC:                             398.8
    Df Residuals:                      42   BIC:                             402.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     76.2719      5.791     13.172      0.000      64.586      87.958
    c              0.8885      0.029     30.986      0.000       0.831       0.946
    ==============================================================================
    Omnibus:                        0.369   Durbin-Watson:                   0.165
    Prob(Omnibus):                  0.832   Jarque-Bera (JB):                0.510
    Skew:                          -0.182   Prob(JB):                        0.775
    Kurtosis:                       2.619   Cond. No.                         353.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      b   R-squared:                       0.954
    Model:                            OLS   Adj. R-squared:                  0.953
    Method:                 Least Squares   F-statistic:                     875.8
    Date:                Tue, 26 Mar 2019   Prob (F-statistic):           9.30e-30
    Time:                        14:02:01   Log-Likelihood:                -222.91
    No. Observations:                  44   AIC:                             449.8
    Df Residuals:                      42   BIC:                             453.4
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    Intercept     41.5849     10.337      4.023      0.000      20.724      62.445
    c              1.5148      0.051     29.593      0.000       1.412       1.618
    ==============================================================================
    Omnibus:                       16.181   Durbin-Watson:                   0.231
    Prob(Omnibus):                  0.000   Jarque-Bera (JB):               19.835
    Skew:                           1.244   Prob(JB):                     4.93e-05
    Kurtosis:                       5.152   Cond. No.                         353.
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.


### Conclusion:  
> - coefficient of determination = 0.962, 0.958, 0.954, which are really close to 1. The models are with high fitting degrees.  
> - Based on the result:  
> -   National HPI = 0.574054 * California HPI + 55.531722  
> -   National HPI = 0.888501 * San Jose-Sunnyvale-Santa Clara HPI + 76.271950  
> -   California HPI = 1.514807 * San Jose-Sunnyvale-Santa Clara HPI + 41.584937  


### Step 12:
Now we know how to predive the house price index. We may want to buy a house for investment. We want to know the growth rate of HPI to make the decision.  
> - Here we use .pct_change() on the last three columns of the data frame and create a new index to show us the year. Because we use .pct_change(), we could never get the data for 1975. Thus, we call iloc agian to drop NA values in '1975' row.


```python
pct_change = mean_hpi2.iloc[:,1:4].pct_change().iloc[1:,:]
pct_change.index += 1975
pct_change.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>National HPI</th>
      <th>California HPI</th>
      <th>San Jose-Sunnyvale-Santa Clara HPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1976</th>
      <td>0.051283</td>
      <td>0.128709</td>
      <td>0.135013</td>
    </tr>
    <tr>
      <th>1977</th>
      <td>0.119989</td>
      <td>0.253721</td>
      <td>0.288151</td>
    </tr>
    <tr>
      <th>1978</th>
      <td>0.140746</td>
      <td>0.192277</td>
      <td>0.152032</td>
    </tr>
    <tr>
      <th>1979</th>
      <td>0.135650</td>
      <td>0.172808</td>
      <td>0.173386</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>0.079991</td>
      <td>0.151756</td>
      <td>0.218964</td>
    </tr>
  </tbody>
</table>
</div>



### Step 13:
Angain, we use .describe() and .corr() for descriptive analysis.


```python
pct_change.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>National HPI</th>
      <th>California HPI</th>
      <th>San Jose-Sunnyvale-Santa Clara HPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>43.000000</td>
      <td>43.000000</td>
      <td>43.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.046503</td>
      <td>0.068257</td>
      <td>0.080895</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.042461</td>
      <td>0.096327</td>
      <td>0.096624</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-0.055766</td>
      <td>-0.198840</td>
      <td>-0.123716</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.030811</td>
      <td>-0.000748</td>
      <td>0.008502</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.051272</td>
      <td>0.080119</td>
      <td>0.084858</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.064157</td>
      <td>0.126375</td>
      <td>0.133271</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.140746</td>
      <td>0.253721</td>
      <td>0.302213</td>
    </tr>
  </tbody>
</table>
</div>




```python
pct_change.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>National HPI</th>
      <th>California HPI</th>
      <th>San Jose-Sunnyvale-Santa Clara HPI</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>National HPI</th>
      <td>1.000000</td>
      <td>0.870285</td>
      <td>0.727606</td>
    </tr>
    <tr>
      <th>California HPI</th>
      <td>0.870285</td>
      <td>1.000000</td>
      <td>0.867981</td>
    </tr>
    <tr>
      <th>San Jose-Sunnyvale-Santa Clara HPI</th>
      <td>0.727606</td>
      <td>0.867981</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Step 14:
To observe the fluctuation of growth rate, we create an area chart, using plt.stackplot().


```python
x = list(pct_change.index)
y = list([pct_change.iloc[:,0],pct_change.iloc[:,1],pct_change.iloc[:,2]])
plt.stackplot(x, y, labels = ['National','California','San Jose-Sunnyvale-Santa Clara'], alpha = 0.6)
plt.legend()
plt.title('Annual Growth Rate of House Price Index', fontsize = 16, fontweight = 'bold')
plt.suptitle('Natioanl, California, San Jose-Sunnyvale-Santa Clara', fontsize = 10)
plt.grid(True)
plt.xlabel('Growth Rate')
plt.ylabel('Year')
plt.xticks(np.linspace(1976,2018,43),rotation = 45)
```




    ([<matplotlib.axis.XTick at 0x1c22e750f0>,
      <matplotlib.axis.XTick at 0x1c22ab8f60>,
      <matplotlib.axis.XTick at 0x1c22ab8da0>,
      <matplotlib.axis.XTick at 0x1c228ec198>,
      <matplotlib.axis.XTick at 0x1c228ec5f8>,
      <matplotlib.axis.XTick at 0x1c228ecac8>,
      <matplotlib.axis.XTick at 0x1c228ece80>,
      <matplotlib.axis.XTick at 0x1c228f24a8>,
      <matplotlib.axis.XTick at 0x1c228f29b0>,
      <matplotlib.axis.XTick at 0x1c228f2ef0>,
      <matplotlib.axis.XTick at 0x1c22908470>,
      <matplotlib.axis.XTick at 0x1c228f2a90>,
      <matplotlib.axis.XTick at 0x1c228ec2b0>,
      <matplotlib.axis.XTick at 0x1c229089b0>,
      <matplotlib.axis.XTick at 0x1c22908ef0>,
      <matplotlib.axis.XTick at 0x1c228d9470>,
      <matplotlib.axis.XTick at 0x1c228d99b0>,
      <matplotlib.axis.XTick at 0x1c228d9ef0>,
      <matplotlib.axis.XTick at 0x1c22909470>,
      <matplotlib.axis.XTick at 0x1c229099b0>,
      <matplotlib.axis.XTick at 0x1c22909ef0>,
      <matplotlib.axis.XTick at 0x1c228d9a90>,
      <matplotlib.axis.XTick at 0x1c228ec0b8>,
      <matplotlib.axis.XTick at 0x1c2290b390>,
      <matplotlib.axis.XTick at 0x1c2290b8d0>,
      <matplotlib.axis.XTick at 0x1c2290be10>,
      <matplotlib.axis.XTick at 0x1c228df390>,
      <matplotlib.axis.XTick at 0x1c228df8d0>,
      <matplotlib.axis.XTick at 0x1c228dfe10>,
      <matplotlib.axis.XTick at 0x1c228df6d8>,
      <matplotlib.axis.XTick at 0x1c2290b9b0>,
      <matplotlib.axis.XTick at 0x1c22aee2b0>,
      <matplotlib.axis.XTick at 0x1c22aee8d0>,
      <matplotlib.axis.XTick at 0x1c22aeee10>,
      <matplotlib.axis.XTick at 0x1c22af5390>,
      <matplotlib.axis.XTick at 0x1c22af58d0>,
      <matplotlib.axis.XTick at 0x1c22af5e10>,
      <matplotlib.axis.XTick at 0x1c22afe390>,
      <matplotlib.axis.XTick at 0x1c22afe8d0>,
      <matplotlib.axis.XTick at 0x1c22af59b0>,
      <matplotlib.axis.XTick at 0x1c22908cf8>,
      <matplotlib.axis.XTick at 0x1c22afecc0>,
      <matplotlib.axis.XTick at 0x1c22b06240>],
     <a list of 43 Text xticklabel objects>)




![png](output_37_1.png)


### Conclusion:
 > - The growth rate of annaual HPI for national is the most stable one. 
 > - The growth rate in San Jose-Sunnyvale-Santa Clara is with the biggist fluctuation.
 > - If we buy a house in San Jose area, we may get high rates of return, but also with high risks.


```python

```
