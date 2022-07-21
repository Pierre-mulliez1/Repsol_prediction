---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.2
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
#import the necessary packages 
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import statsmodels.api as sm
import scipy
import re
from datetime import datetime as dttime
from datetime import date
import calendar
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif,SelectKBest
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor 
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from dateutil.relativedelta import *
from sklearn import metrics
from sklearn.neighbors import BallTree
pd.set_option('display.max_columns', None)
```

<div ><img src="Repsol-Logo.png" width=300 height=200  style="border-radius:5%"></div>

# Fuel Demand & Pricing - Repsol

**Description:** Data competition -- Repsol Fuel pricing

**Time:** 3 Days 
 
**Client:** Repsol 

**Code created:** 2021-03-15 <br>

**Last updated:** 2021-03-18


# Data Loading and Preprocessing
We have two main datasets:

**DATASET_REP**
The dataset provided of Repsol data contains the following timeseries with daily granularity.
- Date: Reference date %m/%d/%y (2016-2019)
- Code: Code of Provice of Spain (PoS)
- Province: Name of the province ("A Coruña")
- Longitude: Location of the POS
- Latitute: Location of the POS
- Red: Supplier/Brand
- Gasoline Demand: Normalized (0-1) volume sold
- Gasoline Price: Gasoline pump price
- Diesel Demand: Normalized (0-1) volume sold
- Diesel: Diesel pump price

**DATASET_COMP**
- Similar to the other dataset but for the POS of competitors and only containing prices (not demand)


Let's load and inspect the datasets.

```python
#Import the dataset
df = pd.read_excel('Dataset_REP.xlsx')
```

```python
#assigning df to another var / avoid reloading the dataset if an eroor arises 
df1 = df.copy()
name_graph = "Fuel prices"
target = "Diesel price"
target_2 = "Gasoline price"
gd = "Gasoline demand"
dd = "Diesel demand"
```

# Data Inspection
## Understanding the dataset

```python
#datetime to unix for performance
import datetime
df1['Date']  = df1['Date'].astype('datetime64[ns]')
df1['dateunix'] = (df1['Date'].apply(lambda x: x.toordinal()) - datetime.date(1970, 1, 1).toordinal()) * 24*60*60
```

```python
print(df1.head())
#grouping price and demand for visualization
df2 = df1.groupby( ['Date']).mean()
df2.reset_index(inplace=True)
print("")
print("Size of grouped dataset: " + str(len(df2)))
```

```python
#Describe qualitative features of the dataset
df.describe(include=[object]).T
```

```python
#Describe quantitative features of the dataset
df.describe(exclude=[np.object]).T
```

```python
# shape of the dataframe
size = df1.shape
print("Size of the Dataframe -> {}".format(size))
```

```python
# column data types
df1.dtypes
```

```python
df1.head()
```

```python
df1.tail()
```

```python
df1.sample(25)
```

```python
df1.isnull().sum()
```

## Understanding the variables


```python
#! pip install pandas-profiling

#from pandas_profiling import ProfileReport
#report = ProfileReport(df1, minimal=False)
#report
```

### Correlation Matrix

```python
rs = np.random.RandomState(0)
corr = df1.corr()
corr.style.background_gradient(cmap='RdBu_r')
```

## Data visualizations

```python
from matplotlib import pyplot 
#size of rba
print(len(df1[dd]))
#number of nulls
print(df1[dd].isnull().sum())
#other stat
print(df1[dd].describe())


bins = 10
rbahist = df1[dd].hist(bins = bins, figsize = (10, 10),
                                   grid = False,facecolor='r',range=(0,1))
#set x axis legend to the median of the bin
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
rbahist.set_xlabel(dd,fontsize=18)
rbahist.set_ylabel(('Frequency') ,fontsize=20)
rbahist.set_title('Distribution of  ' + dd,fontsize=20,weight='bold')
plt.show()

print("skeweness test ")
data = pd.to_numeric(df1[dd], downcast='integer')
print(str(scipy.stats.skew(data)) + ", we have a normally dirtibutted target variable!")

```

```python
from matplotlib import pyplot 
#size of rba
print(len(df1[gd]))
#number of nulls
print(df1[gd].isnull().sum())
#other stat
print(df1[gd].describe())


bins = 10
rbahist = df1[gd].hist(bins = bins, figsize = (10, 10),
                                   grid = False,facecolor='r',range=(0, 1))
#set x axis legend to the median of the bin
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=15)
rbahist.set_xlabel(gd,fontsize=18)
rbahist.set_ylabel(('Frequency') ,fontsize=20)
rbahist.set_title('Distribution of  ' + gd,fontsize=20,weight='bold')
plt.show()

print("skeweness test ")
data = pd.to_numeric(df1[gd], downcast='integer')
print(str(scipy.stats.skew(data)) + ", we have a normally dirtibutted target variable!")
```

```python
#gasoline and Diesel demand over time 
x = df2['Date']
y = df2[gd]
y2 = df2[dd]
plt.rcParams["figure.figsize"] = (8,8)
plt.plot(x, y, color= "r",label = gd,lw = 0.3)
plt.plot(x, y2, color= "b",label = dd,lw = 0.3)
plt.xlabel("Gasoline vs Diesel",fontsize = 16)
plt.ylabel('Demand',fontsize = 16)
plt.legend(bbox_to_anchor=(1.3, 0.80), loc='center', borderaxespad=0., fontsize = 15)
plt.title("Demand evolution",fontsize = 18)
plt.show()
```

```python
#Gasoline / Diesel price over time 
x = df2['Date']
y = df2[target]
y2 = df2[target_2]
plt.rcParams["figure.figsize"] = (8,8)
plt.plot(x, y, color= "y",label = target,lw = 0.5)
plt.plot(x, y2, color= "g",label = target_2,lw = 0.5)
plt.xlabel("Gasoline vs Diesel",fontsize = 16)
plt.ylabel('Price',fontsize = 16)
plt.legend(bbox_to_anchor=(1.3, 0.80), loc='center', borderaxespad=0., fontsize = 15)
plt.title("Price Evolution",fontsize = 18)
plt.show()
```

```python
## Inspecting elasticity 
#Gasoline
x = df2[gd]
y = df2[target_2]
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(x, y, color= "r",label = None,linewidths = 0.5)
plt.xlabel(gd,fontsize = 16)
plt.ylabel('Price',fontsize = 16)
plt.title(gd + " price elasticity",fontsize = 18)
plt.show()
```

```python
#Diesel 
x = df2[dd]
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(x, y, color= "r",label = None,linewidths = 0.5)
plt.xlabel(dd,fontsize = 16)
plt.ylabel('Price',fontsize = 16)
plt.title(dd + " price elasticity",fontsize = 18)
plt.show()
```

## Location areas

```python
#region capital 
#la coruna, santiago de compostela and ferrol respectively to their indexes in the array
x2 = [43.3623,42.8782,43.4896]
y2 = [-8.396,-8.5448,-8.2193]
```

```python
#using longitude and latitude 
#color scale by gasoline price
x = df1['Latitude']
y= df1['Longitude']
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(x, y,label = "station locations",linewidths = 1, c= df1[target_2],s= 400 ,cmap="brg")
plt.scatter(x2, y2, color= "b",label = "large cities locations",linewidths = 1,marker = 'D',s=100)
plt.xlabel("Latitude",fontsize = 16)
plt.ylabel('Longitude',fontsize = 16)
plt.legend(bbox_to_anchor=(1.3, 0.80), loc='center', borderaxespad=0., fontsize = 15)
plt.title( " Location clusters",fontsize = 18)
plt.show()
```

```python
#using longitude and latitude 
#color scale by gasoline demand
x = df1['Latitude']
y= df1['Longitude']
plt.rcParams["figure.figsize"] = (8,8)
plt.scatter(x, y,label = "station locations",linewidths = 1, c= df1[gd],s= 400 ,cmap="brg")
plt.scatter(x2, y2, color= "b",label = "large cities locations",linewidths = 1,marker = 'D',s=100)
plt.xlabel("Latitude",fontsize = 16)
plt.ylabel('Longitude',fontsize = 16)
plt.legend(bbox_to_anchor=(1.3, 0.80), loc='center', borderaxespad=0., fontsize = 15)
plt.title( " Location clusters",fontsize = 18)
plt.show()
```

```python
df1['Latitude']
```

```python
df2['Date'].dt.year.apply(int)
```

# Feature Creation


## Creating variables from the original ones

```python
df1['Month'] = df1['Date'].dt.month.apply(int)
df1['Weekday'] = df1['Date'].dt.weekday.apply(int)
df1['Day'] = df1['Date'].dt.day.apply(int)
df1['Year'] = df1['Date'].dt.year.apply(int)
#Grouped dataset date 
df2['Month'] = df2['Date'].dt.month.apply(int)
df2['Weekday'] = df2['Date'].dt.weekday.apply(int)
df2['Day'] = df2['Date'].dt.day.apply(int)
df2['Year'] = df2['Date'].dt.year.apply(int)
```

## Adding external variables

```python
# adding dataset to consider weather
weather = pd.read_csv("New Temp.csv", sep = ";")
weather.drop_duplicates(subset='FECHA').reset_index(drop=True)
weather = weather.rename(columns={"FECHA": "Date"})
weather['Date']  = weather['Date'].astype('datetime64[ns]')
df1 = df1.merge(weather, on='Date', how='left' )
print("Source: http://www.aemet.es/en/datos_abiertos/AEMET_OpenData")
weather
```

```python
# Brent Index data
crude = pd.read_csv("Brent_Crude.csv")
crude['Date']  = crude['Date'].astype('datetime64[ns]')
df1 = df1.merge(crude, on='Date', how='left' )
print("Source: https://es.investing.com/commodities/brent-oil-historical-datab")
crude
```

```python
# adding feature for holidays
!pip install holidays-es

from holidays_es import Province
holi = pd.DataFrame.from_dict(Province(name="la-coruna", year=2016).holidays(), orient='index').T
for i in range (2017,2020):
    holidays = Province(name="la-coruna", year=i).holidays()
    hello = pd.DataFrame.from_dict(holidays, orient='index').T
    holi = pd.concat([holi, hello], axis=1)
    
holi = pd.concat([holi, holi.T.stack().reset_index(name='new')['new']], axis=1)
holi = pd.DataFrame(holi.iloc[:,12])
```

```python
holi["holiday"] = 1
```

```python
# encoding data from holidays df to datetime and merging with main dataframe
holi["Date"] = pd.to_datetime(holi["new"])
df1 = pd.merge(df1,holi, how = 'left',  on="Date")
df1
```

```python
df1 = df1.drop(columns=["new"])
```

```python
df1['holiday'] = df1['holiday'].replace(np.nan, '0')
df1
```

```python
# adding feature to consider season
def season(x):
    if x == 12 or x == 1 or x == 2:
        return 'Winter'
    elif x == 3 or x == 4 or x == 5:
        return 'Spring'
    elif x == 6 or x == 7 or x== 8:
        return 'Summer'
    else:
        return 'Fall'
    
df1['season'] = df1['Month'].apply(season)
df2['season'] = df2['Month'].apply(season)
```

```python
# adding feature for sunday because trucks aren't allowed to transit on this day 
def sunday(x):
    if x == 6:
        return 1
    else:
        return 0

df1['Sunday'] = df1['Weekday'].apply(sunday)
df2['Sunday'] = df2['Weekday'].apply(sunday)
```

# code to create the geopoint
!pip install geopandas
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
gdf = geopandas.GeoDataFrame(
    df1, geometry=geopandas.points_from_xy(df1.Longitude, df1.Latitude))
gdf

```python

```

```python
# CODE TO FIND THE NN (nearest neighbour)

# def get_nearest(src_points, candidates, k_neighbors=1):
#     """Find nearest neighbors for all source points from a set of candidate points"""

#     # Create tree from the candidate points
#     tree = BallTree(candidates, leaf_size=15, metric='haversine')

#     # Find closest points and distances
#     distances, indices = tree.query(src_points, k=k_neighbors)

#     # Transpose to get distances and indices into arrays
#     distances = distances.transpose()
#     indices = indices.transpose()

#     # Get closest indices and distances (i.e. array at index 0)
#     # note: for the second closest points, you would take index 1, etc.
#     closest = indices[0]
#     closest_dist = distances[0]

#     # Return indices and distances
#     return (closest, closest_dist)


# def nearest_neighbor(left_gdf, right_gdf, return_dist=False):
#     """
#     For each point in left_gdf, find closest point in right GeoDataFrame and return them.

#     NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
#     """

#     left_geom_col = left_gdf.geometry.name
#     right_geom_col = right_gdf.geometry.name

#     # Ensure that index in right gdf is formed of sequential numbers
#     right = right_gdf.copy().reset_index(drop=True)

#     # Parse coordinates from points and insert them into a numpy array as RADIANS
#     left_radians = np.array(left_gdf[left_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
#     right_radians = np.array(right[right_geom_col].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

#     # Find the nearest points
#     # -----------------------
#     # closest ==> index in right_gdf that corresponds to the closest point
#     # dist ==> distance between the nearest neighbors (in meters)

#     closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

#     # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
#     closest_points = right.loc[closest]

#     # Ensure that the index corresponds the one in left_gdf
#     closest_points = closest_points.reset_index(drop=True)

#     # Add distance if requested
#     if return_dist:
#         # Convert to meters from radians
#         earth_radius = 6371000  # meters
#         closest_points['distance'] = dist * earth_radius

#     return closest_points
```

```python
# # Find closest station to each station and get also the distance based on haversine distance
# # Note: haversine distance which is implemented here is a bit slower than using e.g. 'euclidean' metric
# # but useful as we get the distance between points in meters
# closest_stops = nearest_neighbor(data2,data2, return_dist=True)

# closest_stops
```

# Feature Engineering


## Handling nulls

```python
# Percentage of missing values identifyied with "null"
print(100*df1.isnull().sum()/df1.isnull().count())
print("")
print("For the grouped dataset now:")
print(100*df2.isnull().sum()/df2.isnull().count())
df2.fillna(df2.ffill(),inplace=True)
df1.fillna(df1.ffill(),inplace=True)
```

```python
df1.fillna(df1.mean(),inplace=True)
print(100*df1.isnull().sum()/df1.isnull().count())
df1
```

```python
#dropping unique values column 
print({col: df1[col].nunique() for col in df1.columns})
df1 = df1.drop(['Red','Province'], axis = 1)
#drop unecessary grouped columns
df2 = df2.drop(['Longitude','Latitude','dateunix'], axis = 1)
```

## Dummifying Features

```python
df1 = pd.get_dummies(df1, columns=['season'], drop_first=False, prefix='season')

```

```python
#distance to large cities
# vectorized haversine function
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

        a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2
        element = earth_radius * 2 * np.arcsin(np.sqrt(a))
    return element 
#la coruna, santiago de compostela and ferrol 
df1['coruna_l'] = 43.3623
df1['coruna_lo'] = -8.396
df1['santiago_l'] = 42.8782
df1['santiago_lo'] = -8.5448
df1['ferrol_l'] = 43.4896
df1['ferrol_lo'] = -8.2193
df1['Coruna'] =  haversine(df1.loc[:, 'Latitude'], df1.loc[:, 'Longitude'], 
                             df1.loc[:, 'coruna_l'], df1.loc[:,'coruna_lo'])
df1['Santiago'] =  haversine(df1.loc[:, 'Latitude'], df1.loc[:, 'Longitude'], 
                             df1.loc[:, 'santiago_l'], df1.loc[:,'santiago_lo'])
df1['Ferrol'] =  haversine(df1.loc[:, 'Latitude'], df1.loc[:, 'Longitude'], 
                             df1.loc[:, 'ferrol_l'], df1.loc[:,'ferrol_lo'])
```

```python
df1 = df1.drop(['coruna_l','coruna_lo','santiago_l','santiago_lo','ferrol_l','ferrol_lo'], axis = 1)
```

```python
def flagedcities(val):
    if float(val) <= 1:
        val = 1
    else:
        val = 0
    return val  
for col in df1.iloc[:,-3:].columns:
    df1[col] = np.vectorize(flagedcities)(df1[col])
```

```python
df1["city"] = np.maximum.reduce(df1[['Coruna', 'Santiago','Ferrol']].values, axis=1)
```

```python
len(df1[df1.city != 0])/1400
```

```python
df1 = df1.drop(['Coruna','Santiago','Ferrol'], axis = 1)
```

## Analysis by Service Station 

```python
# Demand by service station
pos_demand = df1.pivot(index="Date", columns="Code", values="Gasoline demand")
pos_demand
```

```python
# Price by service station
pos_price = df1.pivot(index="Date", columns="Code", values="Gasoline price")
pos_price
```

```python
pos_price =pd.DataFrame(pos_price.to_records())
```

```python
pos = pos_price.loc[:, pos_price.columns != 'Date']
pos
```

```python
# pos = pos_price.loc[:, pos_price.columns != 'Date']

# poscolnames = []
# for col in pos_price.loc[:, pos_price.columns != 'Date']:
#     poscolnames.append(col)
    
# sns.set_palette("Paired",7)

# for i in poscolnames:
#     sns.lineplot(x="Date", y=i, data=pos, label=i, linewidth = 4.5)

# sns.set(rc={'figure.figsize':(20,10)})
# sns.despine()
# sns.set_style("ticks")


# plt.legend(bbox_to_anchor=(1, 0.8), loc='upper left', prop={'size': 16}, title = 'Musical Features', title_fontsize='18')


# plt.title('Average Score on Musical Features Time Trend',fontsize=30, weight='bold')
# plt.xlabel('Years',fontsize=20, weight='bold')
# plt.ylabel('Average Score',fontsize=20, weight='bold')
# plt.tick_params(labelsize=20)

```

## Array of station, respective elasticities

```python
#Get all stations within an array to fit a model for each one of them 
unique_code = np.unique(df1['Code'])
code_arr = []
for codes in unique_code:
    code_arr.append(df1[df1.Code == codes])
code_arr_el = []
for stations in range(0,len(code_arr)): 
    df4 = code_arr[stations].copy()
    df4.fillna(df4.ffill(),inplace=True, axis = 1)
    df4.loc[(df4['Diesel demand'] == 0) & (df4['Gasoline demand'] == 0), ['Diesel demand','Gasoline demand']] = np.NaN
    df4 = df4.dropna()
    change_priceD = (((df4['Diesel price']*10)- (df4['Diesel price']*10).shift(1)-1)/ ((df4['Diesel price']*10).shift(1)-1))*100
    change_demandD = (((df4['Diesel demand']*10)- (df4['Diesel demand']*10).shift(1)-1)/ ((df4['Diesel demand']*10).shift(1)-1))*100
    change_priceG = (((df4['Gasoline price']*10)- (df4['Gasoline price']*10).shift(1)-1)/((df4['Gasoline price']*10).shift(1)-1))*100
    change_demandG = (((df4['Gasoline demand']*10)- (df4['Gasoline demand']*10).shift(1)-1)/ ((df4['Gasoline demand']*10).shift(1)-1))*100
    df4['Gas elasticity'] = change_demandG/change_priceG;
    df4['Diesel elasticity'] = change_demandD/ change_priceD;
    print("Station ",df4.iloc[1,1], " "  ,len(df4) , " open days")
    code_arr_el.append(df4)
```

```python
code_arr_el[1]
```

```python
#Get the elasticity of each station as a dataframe and dictionnary 
elasticity_stationG = {}
elasticity_stationD = {}
elasticity_df = pd.DataFrame(columns = {"Stations",'Gas elasticity','Diesel elasticity'})
for stations in range(0,len(code_arr)):
    print("Station number", stations, "station: ", code_arr_el[stations].iloc[1,1])
    print(code_arr_el[stations].iloc[:,4:8].describe())
    print('')
    elasticity_stationG[code_arr_el[stations].iloc[1,1]] = np.mean(code_arr_el[stations].loc[:,'Gas elasticity'])
    elasticity_stationD[code_arr_el[stations].iloc[1,1]] = np.mean(code_arr_el[stations].loc[:,'Diesel elasticity'])
    elasticity_df = elasticity_df.append({"Stations": code_arr_el[stations].iloc[1,1],
                                         "Gas elasticity": np.mean(code_arr_el[stations].loc[:,'Gas elasticity']),
                                          'Diesel elasticity':np.mean(code_arr_el[stations].loc[:,'Diesel elasticity'])
                                         },ignore_index=True) 
```

```python
#Reorder station order, observe the elasticity statistics 
cols = elasticity_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
elasticity_df = elasticity_df[cols] 
print(elasticity_df)
print(elasticity_df.iloc[:,1:3].describe())
```

```python
df1.iloc[:,0:28]
```

## Scaling variable 

```python
code_arr_scaled = []
for stations in range(0,len(code_arr)):
    df6 = code_arr_el[stations]
    df1_scaled = df6.copy()
    df6.drop(['Longitude','Latitude','city'],axis =1)
    for col in df6.columns: 
        if (df1_scaled[col].nunique() > 31) & (col != 'Date') & (col != 'Gasoline demand') & (col != 'Gasoline price') & (col != 'Diesel demand') & (col != 'Diesel price'):
            dat = df1_scaled.loc[:,col]
            try:
                dat = pd.DataFrame(dat.apply(pd.to_numeric))
                print(col)
            except:
                continue
            scaler = StandardScaler()
            scaler.fit(dat)
            df_scaling = scaler.transform(dat)
            df1_scaled.loc[:,col] = df_scaling
        else: 
            continue
    code_arr_scaled.append(df1_scaled)
```

```python
code_arr_scaled[1]
```

<div style="color:blue;  font-size: 30px; text-align:center;  font-weight: bold;"> First model  </div> 

```python
# Blocked Time Series Split
class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)
        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.8 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]
```

```python
#Predict gasoline demand using global dataset (not considering location)
df1_regr = pd.DataFrame()
for col in df2.columns:
    try:
        if (col == "Date"):
            continue
        df1_regr[col] = pd.to_numeric(df2[col], downcast="float")
    except:
        print(col)
        continue
X = df1_regr.loc[:, (df1_regr.columns != dd) & (df1_regr.columns != gd)  ]
y = df1_regr.loc[:,gd]
print(X)
score_dict = {}
score_dictA = {}
mod = ["Linear Regression ","Ridge ","Lasso ","Bayesian Ridge "]
count = 0
for model in [LinearRegression(),Ridge(),Lasso(),BayesianRidge()]:
    print('Our model is ' + str(mod[count]))
    for i in range(2,10):
        btscv = BlockingTimeSeriesSplit(n_splits=i)
        print("Number of splits: " + str(i))
        for train_index, test_index in btscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        scores = cross_val_score(model, X_train, y_train, cv=btscv, scoring='neg_mean_squared_error')
        score_dict[str(i)+ " " + str(mod[count])] = scores.mean()
        score_dictA[str(i)+ " " + str(mod[count])] = abs(scores.mean())
        print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))
        print('')
    count += 1
```

```python
print(score_dict)
```

```python
minval = min(score_dictA.values())
result = list(filter(lambda x: score_dictA[x]==minval, score_dictA))
result = result[0]
```

```python
print(result + "score is: " + str(score_dict[result]))
```

```python
#model importance using ridge 
count = 0
rmse = {}
btscv = BlockingTimeSeriesSplit(n_splits=9)
for train_index, test_index in btscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    scores = cross_val_score(model, X_train, y_train, cv=btscv, scoring='neg_mean_squared_error')
    # predicting
    model_b = Ridge()
    model_b.fit(X_train,y_train)
    predictions = model_b.predict(X_test)
    # evaluating the model with RMSE metric
    rmse["fold " + str(count)] = np.sqrt(mean_squared_error(y_test,predictions))
    count =+ 1
```

```python
rmse
```

```python
def get_feature_importance(clf, feature_names):
    return pd.DataFrame({'variable': feature_names, # Feature names
                         'coefficient': clf.coef_# Feature Coeficients
                    }) \
    .round(decimals=2) \
    .sort_values('coefficient', ascending=False) \
    .style.bar(color=['red', 'green'], align='zero')
get_feature_importance(model_b, X_train.columns)
```

```python
#Find the model for the divided df 
```

```python
df1.head()
```

## Models for each station

```python
#Predict gasoline demand using global dataset (not considering location)
score_dict = {}
score_dictA = {}
result_df = pd.DataFrame(columns = {"Splits",'model','station','score_mean', 'score_deviation'})
for stations in range(0,len(code_arr_scaled)): 
    df3 = code_arr_scaled[stations]
    df1_regr = pd.DataFrame()
    for col in df3.columns:
        try:
            if (col == "Date"):
                continue
            df1_regr[col] = pd.to_numeric(df3[col], downcast="float")
        except:
            #print(col)
            continue
    X = df1_regr.loc[:, (df1_regr.columns != dd) & (df1_regr.columns != gd) & (df1_regr.columns != 'Diesel elasticity') & (df1_regr.columns != 'Gas elasticity')]
    y = df1_regr.loc[:,gd]
    mod = ["Linear Regression ","Ridge ","Lasso ","Bayesian Ridge "]
    count = 0
    for model in [LinearRegression(),Ridge(normalize=True),Lasso(),BayesianRidge()]:
        #print('Our model is ' + str(mod[count]))
        for i in range(2,10):
            btscv = BlockingTimeSeriesSplit(n_splits=i)
            #print("Number of splits: " + str(i))
            for train_index, test_index in btscv.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            scores = cross_val_score(model, X_train, y_train, cv=btscv, scoring='neg_mean_squared_error')
            score_dict[str(i)+ "/ " + str(mod[count]) + "/ station code/" + str(stations)] = scores.mean()
            score_dictA[str(i)+ "/ " + str(mod[count]) + "/ station code/" + str(stations)] = abs(scores.mean())
            result_df = result_df.append({"Splits": i,
                                         "model": mod[count],
                                         'station':df3.iloc[1,1],
                                          "score_mean": abs(scores.mean()),
                                          'score_deviation': scores.std()
                                         },ignore_index=True) 
            #print("Loss: {0:.3f} (+/- {1:.3f})".format(scores.mean(), scores.std()))
            #print('')
        count += 1
```

```python
result_df.groupby("station").agg({"score_mean":"min"})
```

```python
result_df
```

```python
#model importance using ridge 
count = 0
rmse = {}
btscv = BlockingTimeSeriesSplit(n_splits=9)
for train_index, test_index in btscv.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    scores = cross_val_score(model, X_train, y_train, cv=btscv, scoring='neg_mean_squared_error')
    # predicting
    model_b = Ridge()
    model_b.fit(X_train,y_train)
    predictions = model_b.predict(X_test)
    # evaluating the model with RMSE metric
    rmse["fold " + str(count)] = np.sqrt(mean_squared_error(y_test,predictions))
    count =+ 1
```

```python
def get_feature_importance(clf, feature_names):
    return pd.DataFrame({'variable': feature_names, # Feature names
                         'coefficient': clf.coef_# Feature Coeficients
                    }) \
    .round(decimals=2) \
    .sort_values('coefficient', ascending=False) \
    .style.bar(color=['red', 'green'], align='zero')
get_feature_importance(model_b, X_train.columns)
```

```python

```
