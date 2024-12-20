# Exploratory Data Analysis on Vehicle Sales(Cars24)

```python
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
import scipy.stats as stat
import pylab
import statsmodels.api as sm
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
```


```python
# Vehicle Sales Features --> 'Car Name', 'Reg Month', 'Make Year', 'Engine Capacity', 'Insurance', 'Spare key', 
#'Transmission', 'KM Driven', 'Ownership', 'Fuel Type', 'Price_in_lakhs', 'EMI/month', 'Brand', 'Model', 'Mileage', 'Registered State' 

data = pd.read_csv("/content/mileage_veh.csv")
data.head(10)
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
      <th>Car Name</th>
      <th>Reg month	</th>
      <th>Make Year	</th>
      <th>Engine Capacity</th>
      <th>Insurance	</th>
      <th>Spare key</th>
      <th>Transmission	</th>
      <th>KM Driven	</th>
      <th>Ownership</th>
      <th>Fuel Type	</th>
      <th> Price_in_lakhs </th>
      <th> EMI/month	</th>
      <th> Brand </th>
      <th> Model </th>
      <th> Mileage </th>
      <th> Registered State </th> 
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016 Maruti Celerio VXI AMT	</td>
      <td>Nov-16</td>
      <td>2016</td>
      <td>998.0</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Automatic</td>
      <td>53167</td>
      <td>1st owner	</td>
      <td>Petrol	</td>
      <td>4.13</td>
      <td>8,074</td>
      <td>Maruti</td>
      <td>Celerio</td>
      <td>20.866233</td>
      <td>Telangana</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020 Hyundai GRAND I10 NIOS SPORTZ 1.0 TURBO G...		</td>
      <td>Feb-21</td>
      <td>2020</td>
      <td>NaN</td>
      <td>No</td>
      <td>Yes</td>
      <td>Manual</td>
      <td>10622</td>
      <td>2nd owner	</td>
      <td>Petrol	</td>
      <td>7.17</td>
      <td>14,017</td>
      <td>Hyundai</td>
      <td>GRAND</td>
      <td>NaN</td>
      <td>Telangana</td>
    </tr>
  </tbody>
</table>
</div>

```python
data.describe()
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
      <th>Make Year	</th>
      <th>Engine Capacity</th>
      <th>KM Driven	</th>
      <th>Price_in_lakhs</th>
      <th>Mileage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>700.000000</td>
      <td>614.000000</td>
      <td>700.000000</td>
      <td>700.000000</td>
      <td>614.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2017.624286</td>
      <td>1334.749186</td>
      <td>52449.572857</td>
      <td>6.763271</td>
      <td>17.610681</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.082599</td>
      <td>358.448873</td>
      <td>27310.655683</td>
      <td>3.822461</td>
      <td>1.985020</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2010.000000</td>
      <td>796.000000</td>
      <td>1413.000000</td>
      <td>1.540000</td>
      <td>12.532217</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2016.000000</td>
      <td>1197.000000</td>
      <td>30236.250000</td>
      <td>4.105000</td>
      <td>16.558820</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2018.000000	</td>
      <td>1199.000000</td>
      <td>49241.000000	</td>
      <td>5.840000</td>
      <td>17.229186</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2020.000000</td>
      <td>1497.000000</td>
      <td>71761.000000</td>
      <td>8.270000</td>
      <td>17.830075</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2023.000000</td>
      <td>2982.000000</td>
      <td>124116.000000</td>
      <td>22.830000</td>
      <td>21.982091</td>
    </tr>
  </tbody>
</table>
</div>

```python
#shape of the dataframe
data.shape
```

(700, 16)

```python
data.nunique()
```

Car Name            489<br>
Reg month           153<br>
Make Year            14<br>
Engine Capacity      40<br>
Insurance             2<br>
Spare key             2<br>
Transmission          2<br>
KM Driven           576<br>
Ownership             3<br>
Fuel Type             3<br>
Price_in_lakhs      432<br>
EMI/month           491<br>
Brand                15<br>
Model                80<br>
Mileage             614<br>
Registered State     16<br>
dtype: int64<br>

```python
# Missing Values
data.isnull().sum()
```

Car Name             0<br>
Reg month            0<br>
Make Year            0<br>
Engine Capacity     86<br>
Insurance            0<br>
Spare key            0<br>
Transmission         0<br>
KM Driven            0<br>
Ownership            0<br>
Fuel Type            0<br>
Price_in_lakhs       0<br>
EMI/month            0<br>
Brand                0<br>
Model                0<br>
Mileage             86<br>
Registered State     0<br>
dtype: int64<br>


## Missing Value Analysis

## msno.bar plot:
```python
msno.bar(data)
```

## msno.matrix :
```python
msno.matrix(data)
```

## msno.dendrogram :
```python
msno.dendrogram(data)
```

# Feature Engneering Tasks :
## Application of Imputation,Deletion,Encoding Technique :
```python3
data['Reg month'] = data['Reg month'].astype(str)

# Extract the month from 'Reg Year' and update the column
data['Reg month'] = data['Reg month'].str.extract('([A-Za-z]+)')

# Convert 'Insurance' and 'Spare key' from 'Yes'/'No' to boolean
data['Insurance'] = data['Insurance'].map({'Yes': True, 'No': False})
data['Spare key'] = data['Spare key'].map({'Yes': True, 'No': False})

data['EMI/month'] = data['EMI/month'].str.replace(',', '').astype(float)

cols = ['Reg month','Transmission','Ownership','Fuel Type','Brand','Model','Make Year','Registered State']
data[cols] = data[cols].astype('category')
data['Make Year'] = data['Make Year'].astype(int)

data.head()

data.info()
```

Data columns (total 16 columns):
 #   Column            Non-Null Count  Dtype   
---  ------            --------------  -----   
 0   Car Name          700 non-null    object  <br>
 1   Reg month         700 non-null    category <br>
 2   Make Year         700 non-null    int64   <br>
 3   Engine Capacity   614 non-null    float64 <br>
 4   Insurance         700 non-null    bool    <br>
 5   Spare key         700 non-null    bool    <br>
 6   Transmission      700 non-null    category<br>
 7   KM Driven         700 non-null    int64   <br>
 8   Ownership         700 non-null    category<br>
 9   Fuel Type         700 non-null    category<br>
 10  Price_in_lakhs    700 non-null    float64 <br>
 11  EMI/month         700 non-null    float64 <br>
 12  Brand             700 non-null    category<br>
 13  Model             700 non-null    category<br>
 14  Mileage           614 non-null    float64 <br>
 15  Registered State  700 non-null    category<br>
dtypes: bool(2), category(7), float64(4), int64(2), object(1)<br>

```python
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Engine Capacity
data['Engine Capacity'].hist(ax=ax[0], bins=30, color='skyblue', edgecolor='black')
ax[0].set_title('Histogram of Engine Capacity')
ax[0].set_xlabel('Engine Capacity (cc)')
ax[0].set_ylabel('Frequency')

# Mileage
data['Mileage'].hist(ax=ax[1], bins=30, color='lightgreen', edgecolor='black')
ax[1].set_title('Histogram of Mileage')
ax[1].set_xlabel('Mileage (km/l)')
ax[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

```python
data['Reg month'] = data['Reg month'].astype(str)
data['Reg month'] = data['Reg month'].str.extract('([A-Za-z]+)')# Extract month from 'Reg Year'and update column
# cols = ['Reg month','Transmission','Ownership','Fuel Type','Brand','Model','Car Name','Make Year','Reg Number']
# data[cols] = data[cols].astype('category')
```

```python
data.info()
```
RangeIndex: 700 entries, 0 to 699<br>
Data columns (total 16 columns):<br>
    Column            Non-Null Count  Dtype   <br>
---  ------            --------------  -----   <br>
 0   Car Name          700 non-null    object  <br>
 1   Reg month         700 non-null    object  <br>
 2   Make Year         700 non-null    int64   <br>
 3   Engine Capacity   614 non-null    float64 <br>
 4   Insurance         700 non-null    bool    <br>
 5   Spare key         700 non-null    bool    <br>
 6   Transmission      700 non-null    category <br>
 7   KM Driven         700 non-null    int64   <br>
 8   Ownership         700 non-null    category <br>
 9   Fuel Type         700 non-null    category <br>
 10  Price_in_lakhs    700 non-null    float64 <br>
 11  EMI/month         700 non-null    float64 <br>
 12  Brand             700 non-null    category <br>
 13  Model             700 non-null    category <br>
 14  Mileage           614 non-null    float64 <br>
 15  Registered State  700 non-null    category <br>
dtypes: bool(2), category(6), float64(4), int64(2), object(2) <br>

```python
data.head()
```

```python
# Impute missing values
data['Engine Capacity'].fillna(data['Engine Capacity'].mean(), inplace=True)
data['Mileage'].fillna(data['Mileage'].median(), inplace=True)
```

```python
data.isnull().sum()

```

```python
data.shape
```

```python
data['Age_of_vehicle'] = 2024 - data['Make Year']
```

```python
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# One-hot encoding for categorical variables without a natural order
one_hot_cols = ['Transmission', 'Fuel Type']
data_encoded = pd.get_dummies(data, columns=one_hot_cols)
```

```python
# Ordinal encoding for 'Ownership' as it has a natural order
ownership_order = ['1st owner', '2nd owner', '3rd owner']
ordinal_encoder = OrdinalEncoder(categories=[ownership_order])
data_encoded['Ownership'] = ordinal_encoder.fit_transform(data[['Ownership']])
```

```python
data_encoded
```

```python
data_encoded.shape
data['Age_of_vehicle'] = 2024 - data['Make Year']
```
# Data Visualization of Trends

```python
plt.figure(figsize=(8, 8))
data['Fuel Type'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Proportion of Different Fuel Types')
plt.ylabel('')
plt.show()
```

```python
plt.figure(figsize=(10,5))
sns.countplot(data=data, x='Make Year',hue='Fuel Type')
#sns.set(rc={'figure.figsize':(10,15)})
plt.title("Type of fuels cars sold over the years")
```

```python
# Check if 'KM Driven' needs cleaning and conversion
if data['KM Driven'].dtype == object:
    data['KM Driven'] = data['KM Driven'].str.replace(' km', '').str.replace(',', '').astype(float)

# Check if 'EMI/month' needs cleaning and conversion
if data['EMI/month'].dtype == object:
    data['EMI/month'] = data['EMI/month'].str.replace(',', '').astype(float)

# Generating histograms for key numeric variables
fig, axes = plt.subplots(3, 2, figsize=(14, 15))
fig.suptitle('Distribution of Key Numeric Variables')

axes[0, 0].hist(data['Price_in_lakhs'], bins=20, color='skyblue')
axes[0, 0].set_title('Distribution of Price in Lakhs')
axes[0, 0].set_xlabel('Price (in Lakhs)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(data['Engine Capacity'].dropna(), bins=20, color='lightgreen')
axes[0, 1].set_title('Distribution of Engine Capacity')
axes[0, 1].set_xlabel('Engine Capacity (cc)')
axes[0, 1].set_ylabel('Frequency')

axes[1, 0].hist(data['Make Year'], bins=range(data['Make Year'].min(), data['Make Year'].max() + 1), color='salmon')
axes[1, 0].set_title('Distribution of Make Year')
axes[1, 0].set_xlabel('Year')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(data['Reg month'].astype(str), bins=30, color='blue', alpha=0.7)
axes[1, 1].set_title('Distribution of Registration month')
axes[1, 1].set_xlabel('Registration month')
axes[1, 1].set_ylabel('Frequency')

axes[2, 0].hist(data['EMI/month'], bins=30, color='purple', alpha=0.7)
axes[2, 0].set_title('Distribution of EMI/month')
axes[2, 0].set_xlabel('EMI/month')
axes[2, 0].set_ylabel('Frequency')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

```

```python
plt.figure(figsize=(10, 10))

# List of categorical features to plot
categorical_features = ['Transmission', 'Fuel Type', 'Ownership', 'Insurance', 'Spare key']

for i, feature in enumerate(categorical_features, 1):
    plt.subplot(3, 2, i)
    data[feature].value_counts().plot(kind='bar', color='skyblue')
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

```python 
def plot_top_10(column, title):
    top_10 = data[column].value_counts().nlargest(10)
    plt.figure(figsize=(10, 8))
    top_10.plot(kind='barh', color='skyblue')
    plt.title(f'Top 10 {title}')
    plt.xlabel('Frequency')
    plt.gca().invert_yaxis()  # Invert y axis to have the largest bar on top
    plt.show()

# Generate plots
plot_top_10('Brand', 'Brands')
plot_top_10('Model', 'Models')
plot_top_10('Make Year', 'Make Years')
```

```python
# Calculate the average price per year
average_price_trends = data.groupby('Make Year')['Price_in_lakhs'].mean()

# Plotting
plt.figure(figsize=(12, 6))
average_price_trends.plot(kind='line', marker='o', linestyle='-', color='green')
plt.title('Average Vehicle Price Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Price (in Lakhs)')
plt.grid(True)
plt.show()
```

```python
# Calculate the average mileage per year
mileage_trends = data.groupby('Make Year')['Mileage'].mean()

# Plotting
plt.figure(figsize=(12, 6))
mileage_trends.plot(kind='line', marker='o', linestyle='-', color='red')
plt.title('Average Mileage Trend Over Years')
plt.xlabel('Year')
plt.ylabel('Average Mileage (km per liter)')
plt.grid(True)
plt.show()
```

```python
corr_matrix = data[['Price_in_lakhs', 'EMI/month', 'Make Year', 'Engine Capacity', 'Mileage', 'KM Driven']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
```

```python
# Function to create scatter plots
def plot_scatter(x, y, title):
    plt.figure(figsize=(10, 6))
    plt.scatter(data[x], data[y], alpha=0.5)
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

plot_scatter('Reg month', 'Price_in_lakhs', 'Registration Year vs. Price')
plot_scatter('Make Year', 'Price_in_lakhs', 'Manufacturing Year vs. Price')
plot_scatter('Engine Capacity', 'Price_in_lakhs', 'Engine Capacity vs. Price')
plot_scatter('Engine Capacity', 'KM Driven', 'Engine Capacity vs. KM Driven')
plot_scatter('KM Driven', 'Price_in_lakhs', 'Kilometers Driven vs. Price')
```

```python
sns.set_style("whitegrid")
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# Boxplot for Transmission vs Price_in_lakhs
sns.boxplot(data=data, x='Transmission', y='Price_in_lakhs', ax=ax[0])
ax[0].set_title('Price Distribution by Transmission Type')
ax[0].set_xlabel('Transmission')
ax[0].set_ylabel('Price in Lakhs')

# Boxplot for Fuel Type vs Price_in_lakhs
sns.boxplot(data=data, x='Fuel Type', y='Price_in_lakhs', ax=ax[1])
ax[1].set_title('Price Distribution by Fuel Type')
ax[1].set_xlabel('Fuel Type')
ax[1].set_ylabel('Price in Lakhs')

plt.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Ownership', y='Price_in_lakhs', order=['1st owner', '2nd owner', '3rd owner'])
plt.title('Price Distribution by Ownership')
plt.xlabel('Ownership')
plt.ylabel('Price in Lakhs')
plt.xticks(rotation=45)
plt.show()

```

```python
plt.figure(figsize=(10, 6))  # Set the figure size
sns.violinplot(x='Transmission', y='Price_in_lakhs', data=data)
plt.title('Price Distribution by Transmission Type')
plt.xlabel('Transmission')
plt.ylabel('Price in Lakhs')
plt.show()
```

```python
# Select a subset of popular brands and their models for mileage analysis
# popular_brands = ['Maruti', 'Hyundai', 'KIA', 'Toyota']
popular_brands_1 = ['Maruti']
data_models_1 = data[data['Brand'].isin(popular_brands_1)]
```

```python
# Bar plot for average mileage by model within brands
# plt.subplot(2, 2, 2)
plt.figure(figsize=(10, 3))
model_mileage_means = data_models_1.groupby(['Brand', 'Model'])['Mileage'].mean().sort_values()[:7]
model_mileage_means.plot(kind='bar')
plt.title('Average Mileage by Car Model')
plt.xlabel('Brand, Model')
plt.ylabel('Average Mileage (km/l)')
```

```python
# Histogram of mileage for selected brands
for brand in popular_brands_1:
    subset = data_models_1[data_models_1['Brand'] == brand]
    sns.histplot(subset['Mileage'], label=brand, kde=True, element='step', stat='density')
plt.title('Histogram of Mileage for Selected Brands')
plt.xlabel('Mileage (km/l)')
plt.ylabel('Density')
plt.legend(title='Brand')
```
```python
popular_brands_2 = ['Hyundai']
data_models_2 = data[data['Brand'].isin(popular_brands_2)]
```

```python
# Histogram of mileage for selected brands
for brand in popular_brands_2:
    subset = data_models_2[data_models_2['Brand'] == brand]
    sns.histplot(subset['Mileage'], label=brand, kde=True, element='step', stat='density')
plt.title('Histogram of Mileage for Selected Brands')
plt.xlabel('Mileage (km/l)')
plt.ylabel('Density')
plt.legend(title='Brand')
```

```python
# Select a subset of popular brands and their models for mileage analysis
# popular_brands = ['Maruti', 'Hyundai', 'KIA', 'Toyota']
popular_brands_3 = ['KIA']
data_models_3 = data[data['Brand'].isin(popular_brands_3)]
```

```python
plt.figure(figsize=(10, 3))
model_mileage_means = data_models_3.groupby(['Brand', 'Model'])['Mileage'].mean().sort_values()[:4]
model_mileage_means.plot(kind='bar')
plt.title('Average Mileage by Car Model')
plt.xlabel('Brand, Model')
plt.ylabel('Average Mileage (km/l)')
```

```python
# Histogram of mileage for selected brands
for brand in popular_brands_3:
    subset = data_models_3[data_models_3['Brand'] == brand]
    sns.histplot(subset['Mileage'], label=brand, kde=True, element='step', stat='density')
plt.title('Histogram of Mileage for Selected Brands')
plt.xlabel('Mileage (km/l)')
plt.ylabel('Density')
plt.legend(title='Brand')
```

```python
# Select a subset of popular brands and their models for mileage analysis
# popular_brands = ['Maruti', 'Hyundai', 'KIA', 'Toyota']
popular_brands_4 = ['Toyota']
data_models_4 = data[data['Brand'].isin(popular_brands_4)]
```

```python
# Bar plot for average mileage by model within brands
plt.figure(figsize=(10, 3))
model_mileage_means = data_models_4.groupby(['Brand', 'Model'])['Mileage'].mean().sort_values()[:7]
model_mileage_means.plot(kind='bar')
plt.title('Average Mileage by Car Model')
plt.xlabel('Brand, Model')
plt.ylabel('Average Mileage (km/l)')
```

```python
# Histogram of mileage for selected brands
for brand in popular_brands_4:
    subset = data_models_4[data_models_4['Brand'] == brand]
    sns.histplot(subset['Mileage'], label=brand, kde=True, element='step', stat='density')
plt.title('Histogram of Mileage for Selected Brands')
plt.xlabel('Mileage (km/l)')
plt.ylabel('Density')
plt.legend(title='Brand')
```

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Make Year', y='Price_in_lakhs', size='Engine Capacity', hue='Engine Capacity', palette='viridis', sizes=(20, 200))
plt.title('Price Variation with Manufacturing Year and Engine Capacity')
plt.xlabel('Manufacturing Year')
plt.ylabel('Price in Lakhs')
plt.grid(True)
plt.show()
```

```python
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='KM Driven', y='Price_in_lakhs', hue='Fuel Type')
plt.title('Relationship Between Car Price and Kilometers Driven by Fuel Type')
plt.xlabel('Kilometers Driven')
plt.ylabel('Price in Lakhs')
plt.grid(True)
plt.show()
```

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Transmission', y='Price_in_lakhs', hue='Ownership')
plt.title('Price Difference by Transmission Type and Ownership Status')
plt.xlabel('Transmission Type')
plt.ylabel('Price in Lakhs')
plt.show()
```

```python
plt.figure(figsize=(10, 6))
sns.boxplot(data=data, x='Spare key', y='Price_in_lakhs', hue='Insurance')
plt.title('Price Comparison of Cars with and Without Spare Keys by Insurance Type')
plt.xlabel('Spare Key Available')
plt.ylabel('Price in Lakhs')
plt.show()
```

```python
plt.figure(figsize=(12, 8))
sns.scatterplot(data=data, x='Brand', y='Price_in_lakhs', size='Engine Capacity', hue='Engine Capacity', palette='coolwarm', sizes=(20, 200))
plt.title('Price Differences Across Brands by Engine Capacity')
plt.xticks(rotation=45)
plt.xlabel('Brand')
plt.ylabel('Price in Lakhs')
plt.grid(True)
plt.show()
```

```python
g = sns.FacetGrid(data, col="Insurance", row="Transmission", margin_titles=True)
g.map_dataframe(sns.histplot, x="Price_in_lakhs")
g.set_axis_labels("Price in Lakhs", "Count")
g.set_titles(col_template="{col_name} Insurance", row_template="{row_name} Transmission")
g.tight_layout()
plt.show()
```

```python
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Make Year', y='Price_in_lakhs', hue='Fuel Type', marker='o')
plt.title('Relationship Between Car Price and Manufacturing Year by Fuel Type')
plt.xlabel('Manufacturing Year')
plt.ylabel('Price in Lakhs')
plt.grid(True)
plt.show()
```

```python 
sns.lmplot(data=data, x='Engine Capacity', y='Price_in_lakhs', col='Transmission', fit_reg=True)
plt.show()
```

```python
sns.lmplot(data=data, x='Make Year', y='Price_in_lakhs', hue='Ownership', palette='Set1', aspect=1.5)
plt.title('Interaction Effect Between Make Year and Ownership Type on Car Prices')
plt.xlabel('Make Year')
plt.ylabel('Price in Lakhs')
plt.show()
```

```python
from sklearn.feature_selection import VarianceThreshold

numerical_data = data_encoded.select_dtypes(include=['int64', 'float64'])

# Apply Variance Threshold with a threshold of 0.08
selector = VarianceThreshold(threshold=0.08)
selector.fit(numerical_data)

high_variance_features = numerical_data.columns[selector.get_support()]

high_variance_features
```

```python
data_encoded
```

```python
X = data_encoded.drop(columns=['Brand','Model','Car Name','Price_in_lakhs','EMI/month','Reg month','Registered State'])
y = data_encoded['Price_in_lakhs']

mi_scores = mutual_info_regression(X, y)
mi_results = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

# Display the top 7 features based on mutual information
top_7_mi_features = mi_results.head(7)
```

```python
top_7_mi_features
```

```python
# Correlation of numerical features with the target
correlation_data = numerical_data.join(data_encoded[['Transmission_Automatic', 'Transmission_Manual']])
correlation_scores = correlation_data.corr()['Price_in_lakhs'].sort_values(ascending=False)

# Display correlation scores of the top features
correlation_scores.drop('Price_in_lakhs').head(7)
```

```python
def plot_data(df, feature):
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    df[feature].hist()
    plt.subplot(1, 2, 2)
    stat.probplot(df[feature], dist='norm', plot=pylab)
    plt.show()

for feature in ['Engine Capacity', 'KM Driven', 'Mileage']:
    plot_data(data_encoded, feature)
```

```python
from scipy import stats
import pylab

transformed_data = data_encoded.copy()

transformed_data['Log_KM Driven'] = np.log1p(transformed_data['KM Driven'])
transformed_data['Log_Mileage'] = np.log1p(transformed_data['Mileage'])

transformed_data['BoxCox_Engine Capacity'], fitted_lambda = stats.boxcox(transformed_data['Engine Capacity'])

for feature in ['BoxCox_Engine Capacity', 'Log_KM Driven', 'Log_Mileage']:
    plot_data(transformed_data, feature)
```

# Problem - 1 : Price Prediction
Problem Statement: Predict the price of a used car based on its features such as make year, brand, model, mileage, engine capacity, and other available features.

# Model Type: Regression model (e.g., Linear Regression, Random Forest).

Procedure :

In this Section Of analysied "Model-Fitting" by use of Feature Selcection method like Variance Threshold, Mutual -Information,Pearson Coe-relation through which we choosen top K features,in our case we have taken top 7 features and tried to apply Feature transformation technique inoreder to transform feature distribution to Gaussian Distribution.

In Model Fitting for price prediction we are bound to use Regressor Model like Linear regression , RandomForest Regressor, and also applied boosting technique like XGboost regressor to train our problem. to solve prediction cases.

Next, we tried to implement model fitting without Feature selection and transformation and see how it differ from above Transformations.

At last , we comapre different model using evulation metric for regressor like MAE,R^2 score,MSE etc.

```python
from sklearn.model_selection import train_test_split

features = ['BoxCox_Engine Capacity', 'Log_KM Driven', 'Log_Mileage', 'Insurance',
            'Spare key', 'Ownership', 'Make Year','Transmission_Automatic','Transmission_Manual']

X_transformed = transformed_data[features]
y_transformed = transformed_data['Price_in_lakhs']

X_train, X_test, y_train, y_test = train_test_split(X_transformed, y_transformed, test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

```python
y_pred = model.predict(X_test)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test,y_pred)
rmse = mean_squared_error(y_test,y_pred, squared=False)
r2 = r2_score(y_test,y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
```

```python
data.head()
```

```python
X = data_encoded.drop(['Brand', 'Price_in_lakhs','Model','Registered State'], axis=1)
y = data_encoded['Price_in_lakhs']
```
```python     
if 'Car Name' in X.columns:
    X = X.drop('Car Name', axis=1)

categorical_cols = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=categorical_cols)

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
model1 = LinearRegression()
model1.fit(X_train, y_train)
```

```python
y_pred_lr = model1.predict(X_test)

```

```python
import matplotlib.pyplot as plt

# Create a scatter plot for y_test
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='red', label='Actual Values')

# Add a scatter plot for y_pred
plt.scatter(y_test, y_pred_lr, color='blue', label='Predicted Values')

# Customize plot labels and title
plt.xlabel('Actual Price prediction')
plt.ylabel('Predicted Price prediction')
plt.title('Predicted vs Actual Price prediction')
plt.legend()
plt.grid(True)
plt.show()
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test,y_pred_lr)
mse = mean_squared_error(y_test,y_pred_lr)
rmse = mean_squared_error(y_test,y_pred_lr, squared=False)
r2 = r2_score(y_test,y_pred_lr)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

```

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


reg_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

reg_model.fit(X_train, y_train)

y_pred_Xgreg = reg_model.predict(X_test)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test,y_pred_Xgreg)
mse = mean_squared_error(y_test,y_pred_Xgreg)
rmse = mean_squared_error(y_test,y_pred_Xgreg, squared=False)
r2 = r2_score(y_test,y_pred_Xgreg)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

```

```python
import matplotlib.pyplot as plt

# Create a scatter plot for y_test
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='red', label='Actual Values')

# Add a scatter plot for y_pred_reg
plt.scatter(y_test, y_pred_Xgreg, color='blue', label='Predicted Values')

# Customize plot labels and title
plt.xlabel('Price prediction')
plt.ylabel('Predicted Fuel Efficiency')
plt.title('Predicted vs Actual Price prediction')
plt.legend()
plt.grid(True)
plt.show()
```

```python
model3 = RandomForestRegressor(n_estimators=100, random_state=42)
model3.fit(X_train, y_train)

```

```python
y_predRf = model3.predict(X_test)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test,y_predRf)
mse = mean_squared_error(y_test,y_predRf)
rmse = mean_squared_error(y_test,y_predRf, squared=False)
r2 = r2_score(y_test,y_predRf)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

```

```python 
import matplotlib.pyplot as plt

# Create a scatter plot for y_test
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test, color='red', label='Actual Values')

# Add a scatter plot fpr y_pred
plt.scatter(y_test,y_predRf, color='blue', label='Predicted Values')

# Customize plot labels and title
plt.xlabel('Actual Fuel Efficiency')
plt.ylabel('Predicted Fuel Efficiency')
plt.title('Predicted vs Actual Price prediction')
plt.legend()
plt.grid(True)
plt.show()
```

```python

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Metrics for Random Forest Regressor
mae_rf = mean_absolute_error(y_test, y_predRf)
mse_rf = mean_squared_error(y_test, y_predRf)
rmse_rf = mean_squared_error(y_test, y_predRf, squared=False)
r2_rf = r2_score(y_test, y_predRf)

# Metrics for XGBoost Regressor
mae_xgb = mean_absolute_error(y_test, y_pred_Xgreg)
mse_xgb = mean_squared_error(y_test, y_pred_Xgreg)
rmse_xgb = mean_squared_error(y_test, y_pred_Xgreg, squared=False)
r2_xgb = r2_score(y_test, y_pred_Xgreg)

# Metrics for Linear Regression
mae_linear = mean_absolute_error(y_test, y_pred_lr)
mse_linear = mean_squared_error(y_test, y_pred_lr)
rmse_linear = mean_squared_error(y_test, y_pred_lr, squared=False)
r2_linear = r2_score(y_test, y_pred_lr)

# Display the results in a table-like format
print("| Metric                  | Random Forest | XGBoost    | Linear Regression |")
print("|-------------------------|---------------|------------|-------------------|")
print(f"| Mean Absolute Error     | {mae_rf:.4f}       | {mae_xgb:.4f}    | {mae_linear:.4f}           |")
print(f"| Mean Squared Error      | {mse_rf:.4f}       | {mse_xgb:.4f}    | {mse_linear:.4f}           |")
print(f"| Root Mean Squared Error | {rmse_rf:.4f}       | {rmse_xgb:.4f}    | {rmse_linear:.4f}           |")
print(f"| R-squared               | {r2_rf:.4f}       | {r2_xgb:.4f}    | {r2_linear:.4f}           |")

```

# Problem -2 : Fuel Efficiency Prediction
Problem Statement: Predict the fuel efficiency (mileage) of cars based on their engine specifications, make year, and other features.

# Model Type: Regression analysis.

# Procedure :

In this Section of "Fuel Efficiency Prediction", we consider "Mileage" as our main Target variable,and took independent features like Model, brand,'Transmission', 'Fuel Type','Ownership' etc for better analysis of Fuel Consumpution.

Firstly, we need our Feature variable and target variable in hand, so for feature variable there were some column which is directly effect the target varible, so to consider them we have applied One hot Encoding for better model interpretation.

Next,we consider the feature and target part seperately and applied Train-test split for better model training adn overcome over-fitting.

Now , we have implemented regressor models like linear regreesion, RandomForest Regressor, and also applied boosting technique like XGboost regressor to train our problem.

At last , we comapre different model using evulation metric for regressor like MAE,R^2 score,MSE etc.

```python
from sklearn.preprocessing import OneHotEncoder

# One-hot encoding for categorical variables without a natural order
one_hot_cols = ['Brand','Model','Transmission', 'Fuel Type','Ownership']
data_encoded_fuel = pd.get_dummies(data, columns=one_hot_cols)

```

```python
# features = ['Fuel Type_CNG',	'Fuel Type_Diesel',	'Fuel Type_Petrol', 'Transmission_Automatic', 'Engine Capacity','Transmission_Manual','Mileage']
features = data_encoded_fuel.drop(['Car Name','Reg month','Registered State'],axis = 1)
target = 'Mileage'

```

```python
X_fuel = features.drop(['Mileage'], axis=1)
y_fuel = features['Mileage']
X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(X_fuel, y_fuel, test_size=0.2, random_state=42)

```


```python
data_encoded_fuel.shape

```

```python
model = LinearRegression()
model.fit(X_train_fuel, y_train_fuel)

y_pred = model.predict(X_test_fuel)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_fuel,y_pred)
mse = mean_squared_error(y_test_fuel,y_pred)
rmse = mean_squared_error(y_test_fuel,y_pred, squared=False)
r2 = r2_score(y_test_fuel,y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

```

```python
reg_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

reg_model.fit(X_train_fuel, y_train_fuel)

y_pred_reg = reg_model.predict(X_test_fuel)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_fuel, y_pred_reg)
mse = mean_squared_error(y_test_fuel, y_pred_reg)
rmse = mean_squared_error(y_test_fuel, y_pred_reg, squared=False)
r2 = r2_score(y_test_fuel, y_pred_reg)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")
```

```python
model_fuel = RandomForestRegressor(n_estimators=100, random_state=42, max_features='sqrt', max_depth=10)
model_fuel.fit(X_train_fuel, y_train_fuel)
```

```python
y_pred_fuel = model_fuel.predict(X_test_fuel)
```

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test_fuel, y_pred_fuel)
mse = mean_squared_error(y_test_fuel, y_pred_fuel)
rmse = mean_squared_error(y_test_fuel, y_pred_fuel, squared=False)
r2 = r2_score(y_test_fuel, y_pred_fuel)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

```
Mean Absolute Error: 0.6140019754484282<br>
Mean Squared Error: 0.6107239266892724<br>
Root Mean Squared Error: 0.7814882767446178 <br>
R-squared: 0.8308343849204499<br>


```python
data.info()
```

Data columns (total 17 columns):
 #   Column            Non-Null Count  Dtype  <br> 
---  ------            --------------  -----   <br>
 0   Car Name          700 non-null    object  <br>
 1   Reg month         700 non-null    object  <br>
 2   Make Year         700 non-null    int64   <br>
 3   Engine Capacity   700 non-null    float64 <br>
 4   Insurance         700 non-null    bool    <br>
 5   Spare key         700 non-null    bool    <br>
 6   Transmission      700 non-null    category <br>
 7   KM Driven         700 non-null    int64   <br>
 8   Ownership         700 non-null    category <br>
 9   Fuel Type         700 non-null    category <br>
 10  Price_in_lakhs    700 non-null    float64 <br>
 11  EMI/month         700 non-null    float64 <br>
 12  Brand             700 non-null    category <br>
 13  Model             700 non-null    category <br>
 14  Mileage           700 non-null    float64 <br>
 15  Registered State  700 non-null    category <br>
 16  Age_of_vehicle    700 non-null    int64   <br>
dtypes: bool(2), category(6), float64(4), int64(3), object(2) <br>
