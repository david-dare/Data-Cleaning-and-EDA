**File will show basic EDA techniques in Python and also basic data cleaning on dataset.


**Importing libraries and dependencies


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
```

**Importing, inspecting head of dataframe:


```python
df = pd.read_csv(r"C:\Users\User\Downloads\airlines_final.csv")
df.head()
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
      <th>Unnamed: 0</th>
      <th>id</th>
      <th>day</th>
      <th>airline</th>
      <th>destination</th>
      <th>dest_region</th>
      <th>dest_size</th>
      <th>boarding_area</th>
      <th>dept_time</th>
      <th>wait_min</th>
      <th>cleanliness</th>
      <th>safety</th>
      <th>satisfaction</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1351</td>
      <td>Tuesday</td>
      <td>UNITED INTL</td>
      <td>KANSAI</td>
      <td>Asia</td>
      <td>Hub</td>
      <td>Gates 91-102</td>
      <td>2018-12-31</td>
      <td>115.0</td>
      <td>Clean</td>
      <td>Neutral</td>
      <td>Very satisfied</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>373</td>
      <td>Friday</td>
      <td>ALASKA</td>
      <td>SAN JOSE DEL CABO</td>
      <td>Canada/Mexico</td>
      <td>Small</td>
      <td>Gates 50-59</td>
      <td>2018-12-31</td>
      <td>135.0</td>
      <td>Clean</td>
      <td>Very safe</td>
      <td>Very satisfied</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2820</td>
      <td>Thursday</td>
      <td>DELTA</td>
      <td>LOS ANGELES</td>
      <td>West US</td>
      <td>Hub</td>
      <td>Gates 40-48</td>
      <td>2018-12-31</td>
      <td>70.0</td>
      <td>Average</td>
      <td>Somewhat safe</td>
      <td>Neutral</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>1157</td>
      <td>Tuesday</td>
      <td>SOUTHWEST</td>
      <td>LOS ANGELES</td>
      <td>West US</td>
      <td>Hub</td>
      <td>Gates 20-39</td>
      <td>2018-12-31</td>
      <td>190.0</td>
      <td>Clean</td>
      <td>Very safe</td>
      <td>Somewhat satsified</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2992</td>
      <td>Wednesday</td>
      <td>AMERICAN</td>
      <td>MIAMI</td>
      <td>East US</td>
      <td>Hub</td>
      <td>Gates 50-59</td>
      <td>2018-12-31</td>
      <td>559.0</td>
      <td>Somewhat clean</td>
      <td>Very safe</td>
      <td>Somewhat satsified</td>
    </tr>
  </tbody>
</table>
</div>



**Checking datatypes and shape of dataframe:



```python
df.info()
df.shape
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2477 entries, 0 to 2476
    Data columns (total 13 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Unnamed: 0     2477 non-null   int64  
     1   id             2477 non-null   int64  
     2   day            2477 non-null   object 
     3   airline        2477 non-null   object 
     4   destination    2477 non-null   object 
     5   dest_region    2477 non-null   object 
     6   dest_size      2477 non-null   object 
     7   boarding_area  2477 non-null   object 
     8   dept_time      2477 non-null   object 
     9   wait_min       2477 non-null   float64
     10  cleanliness    2477 non-null   object 
     11  safety         2477 non-null   object 
     12  satisfaction   2477 non-null   object 
    dtypes: float64(1), int64(2), object(10)
    memory usage: 251.7+ KB
    




    (2477, 13)



**Unnamed 0: column is duplication of axis and thus can be dropped


```python
df = df.drop(columns="Unnamed: 0")
```

id can act as unique identifier and can be set as the index:



```python
df = df.set_index("id")
```

**wait_min is only numeric field - so descriptive statistics can be generated



```python
df["wait_min"].describe()
```




    count    2477.000000
    mean      166.084376
    std       107.643625
    min        15.000000
    25%       105.000000
    50%       145.000000
    75%       185.000000
    max      1365.000000
    Name: wait_min, dtype: float64


**Useful to run
value counts on other axis to get insight into data:

```python
df["airline"].value_counts(normalize=True).head(10)



```




    UNITED            0.212354
    ALASKA            0.127170
    SOUTHWEST         0.074283
    AMERICAN          0.070650
    DELTA             0.068631
    UNITED INTL       0.062576
    JETBLUE           0.058135
    AIR CANADA        0.053694
    AIR FRANCE/KLM    0.048446
    CATHAY PACIFIC    0.032701
    Name: airline, dtype: float64




```python
df["destination"].value_counts(normalize=True).head(10)
```




    LOS ANGELES        0.074687
    PARIS-DE GAULLE    0.050868
    DENVER             0.038757
    HONG KONG          0.038353
    LONDON HEATHROW    0.036738
    SAN DIEGO          0.034719
    NEW YORK-JFK       0.032297
    CHICAGO-O'HARE     0.031086
    NEWARK             0.030682
    LAS VEGAS          0.030279
    Name: destination, dtype: float64




```python
df["boarding_area"].value_counts(normalize=True).head()
```




    Gates 1-12      0.245862
    Gates 91-102    0.191361
    Gates 50-59     0.188535
    Gates 70-90     0.154623
    Gates 20-39     0.090836
    Name: boarding_area, dtype: float64



**cleanliness, safety and satisfaction are currently stored as object - better practice to store as category:


```python
df["cleanliness"] = df["cleanliness"].astype("category")
df["safety"] = df["safety"].astype("category")
df["satisfaction"] = df["satisfaction"].astype("category")                
```

**dept_time currently stored as object need to be stored as data


```python
df["dept_time"] = pd.to_datetime(df["dept_time"])
```

**I'm interested to visualise wait_min to see possible presence of outliers



```python
sns.boxplot(df["wait_min"])
plt.show()
```


![png](output_22_0.png)



Assign values from the 95th percentile for wait_min to outliers(+95%) by creating new column 'wait_min_modified'


```python
percentile_95 = np.quantile(df["wait_min"],.95)
```


```python
df["wait_min_modified"] = df["wait_min"]
```


```python
df["wait_min_modified"] = np.where(df["wait_min_modified"] > percentile_95,percentile_95,df["wait_min"])
```

**Check entries for duplicates and typos in dest_region


```python
df["dest_region"].unique()
```




    array(['Asia', 'Canada/Mexico', 'West US', 'East US', 'Midwest US',
           'EAST US', 'Middle East', 'Europe', 'eur', 'Central/South America',
           'Australia/New Zealand', 'middle east'], dtype=object)



**Replace 'eur' with 'Europe','EAST US' with 'East US', 'middle east' with 'Middle East' to clean this column


```python
df["dest_region"] = df["dest_region"].str.replace("eur","Europe")
df["dest_region"] = df["dest_region"].str.replace("EAST US","East US")
df["dest_region"] = df["dest_region"].str.replace("middle east","Middle East")

```

**Check changes have been effective


```python
df["dest_region"].unique()
```




    array(['Asia', 'Canada/Mexico', 'West US', 'East US', 'Midwest US',
           'Middle East', 'Europe', 'Central/South America',
           'Australia/New Zealand'], dtype=object)



**Create mapping from day of week to weekday/end split:


```python
wday_mappings = {"Monday":"weekday","Tuesday":"weekday","Wednesday":"weekday","Thursday":"weekday","Friday":"weekday","Saturday":"weekend","Sunday":"weekend"}
```


```python
df["weekday/weekday"] = df["day"].map(wday_mappings)
```

**Remove leading/trailing spaces in 'dest_size' column.


```python
df["dest_size"] = df["dest_size"].str.strip()
```

**Check changes have been effective:


```python
df["dest_size"].unique()
```




    array(['Hub', 'Small', 'Medium', 'Large'], dtype=object)


**Remove any duplicated rows


```python

```
