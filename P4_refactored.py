#%% md

# I. Introduction

#%% md

**Problématique**


La ville de Seattle aimerait réduire ses émissions de C02 et diminuer sa consommation en énergie, en particulier au niveau de ses bâtiments.

Via des mesures de chaleurs, on peut estimer le dégagement carbonné des bâtiments de la ville. Problème : ces mesures de chaleurs sont coûteuses, et il est inenvisageable de les déployer sur l'ensemble des bâtiments de la ville.

**Buts et intérêts**


L'objectif du projet est de prédire les émissions et consommations des bâtiments de la ville. Le programme va tout d'abord fournir quelques analyses du jeu de données à disposition, normaliser les données utiles, puis finalement utiliser quelques algorithmes de Machine Learning pour établir des prédictions.

L'intérêt d'un tel programme est de nous affranchir de réaliser les mesures de chaleur onéreuses. Le programme fournit également le niveau de fiabilité qu'on peut accorder aux différentes prédictions.

#%% md

# II. Imports

#%%

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import seaborn as sns
import statistics as stat
import sklearn.model_selection
import time

from collections import Counter
from matplotlib.pyplot import figure
from scipy.stats import uniform, randint

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV, KFold, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import xgboost as xgb

import std_eda

#%%

print(sklearn.__version__)

#%% md

# III. Data presentation

#%% md

## 1 Présentation

#%%

CHEMIN = 'drive/My Drive/Colab Notebooks/ocr_data_scientist/P4 Consommation électrique des bâtiments/'
FILENAME_2015 = '2015-building-energy-benchmarking.csv'
FILENAME_2016 = '2016-building-energy-benchmarking.csv'

#%%

main2015_df = pd.read_csv(CHEMIN + FILENAME_2015, sep=',')
main2016_df = pd.read_csv(CHEMIN + FILENAME_2016, sep=',')
my_explorator = std_eda.EdaExplorator(main2015_df, main2016_df)

#%%

main2015_df.info()

#%%

main2016_df.info()

#%%

def NaN_proportion(df):
    '''Returns the proportion of NaN values in the whole dataframe'''
    nan_proportion = df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100
    return nan_proportion

#%%

NaN_proportion(main2015_df)

#%% md

## 2 Fusion des tableaux 2015 & 2016

#%% md

Identification des colonnes différentes d'une année à l'autre



#%%

columns_2015 = [column for column in main2015_df.columns]
columns_2016 = [column for column in main2016_df.columns]

#%%

print('Columns in common: \n')
commons_list = list(set(columns_2015) & set(columns_2016))
commons_list

#%%

specific_2016 = list(set(columns_2016) - set(columns_2015))
specific_2016

#%%

specific_2015 = list(set(columns_2015) - set(columns_2016))
specific_2015

#%% md

Ici, nous repérons deux couples de colonnes ayant un nom similaire d'une année à l'autre :


*   GHGEmissions(MetricTonsCO2e) & TotalGHGEmissions
*   GHGEmissionsIntensity(kgCO2e/ft2) & GHGEmissionsIntensity

Nous allons voir si ces colonnes aux noms similaires désignent la même caractéristique.



#%%

main2015_df['GHGEmissionsIntensity(kgCO2e/ft2)'].describe()

#%%

main2016_df['GHGEmissionsIntensity'].describe()

#%% md

Nous trouvons des colonnes de noms différents, mais similaires. Donnons-leur le même nom dans les deux dataframes, 2015 et 2016.

#%%

# Nettoyage des données du tableau 2015
main2015_df = main2015_df.rename(columns={'Comment':'Comments',
                                          'GHGEmissions(MetricTonsCO2e)':'GHGEmissionsTotal',
                                          'GHGEmissionsIntensity(kgCO2e/ft2)':'GHGEmissionsIntensity',
                                          'Zip Codes':'ZipCode'})

# Nettoyage des données du tableau 2016
main2016_df = main2016_df.drop(['Address', 'City', 'State'], axis=1)
main2016_df = main2016_df.rename(columns={'TotalGHGEmissions':'GHGEmissionsTotal'})

columns_2015 = [column for column in main2015_df.columns]
columns_2016 = [column for column in main2016_df.columns]

#%%

print('Columns in common: \n')
commons_list = list(set(columns_2015) & set(columns_2016))
commons_list

#%%

print('Columns contained in 2016 dataframe but not in 2015')
specific_2016 = list(set(columns_2016) - set(columns_2015))
specific_2016

#%%

print('Columns contained in 2015 dataframe but not in 2016')
specific_2015 = list(set(columns_2015) - set(columns_2016))
specific_2015

#%% md

Détermination des clés de chaque tableau

#%%

def columns_with_unique_values(df):
    df_length = len(df)
    temp_dict={}
    for column in df.columns:
        temp_dict[column] = len(df[column].unique())
    temp_list = [element[0] for element in temp_dict.items() if element[1]==temp_value]
    return temp_list

columns_with_unique_values(main2015_df)

#%%

columns_with_unique_values(main2016_df)

#%% md

La colonne OSEBuilding est la clé pour 2015 et pour 2016.

#%% md

**Estimation des différences colonne par colonne entre les deux années**

Pour cela, nous allons comparer les éléments de 2015 et 2016 clé par clé.

#%%

# Fusion des deux tableaux, en prenant les clés pour indice
merge_df = main2015_df.merge(main2016_df,
                             left_on='OSEBuildingID',
                             right_on='OSEBuildingID',
                             suffixes=('_2015', '_2016'))

#%%

# Vérification des résultats sur quelques colonnes :
# YearBuilt : env. 100%
# PropertyGFABuilding(s) : env. 85%
# ZipCode : env. 0%
merge_df['ZipCode_2015'][:10], merge_df['ZipCode_2016'][:10]

#%% md

**Fusion des deux tableaux**

#%%

print('Dimensions du tableau de 2015 : {}'.format(main2015_df.shape))
print('Dimensions du tableau de 2016 : {}'.format(main2016_df.shape))
print('Dimensions du tableau fusionné : {}'.format(merge_df.shape))

#%% md

On voit que le tableau fusionné contient moins de lignes que les deux tableaux de départ. Cela veut dire que certains bâtiments ont été passés en revue en 2015, mais pas en 2016, et vice-versa.

Il serait dommage de laisser de côté ces bâtiments. Pour une modélisation de Machine Learning, il est préférable d'avoir le plus de données possible.

#%%

# Récupérer la liste des bâtiments propres à 2015
buildings2015only = set(main2015_df['OSEBuildingID']) - set(merge_df['OSEBuildingID'])
buildings2015only = list(buildings2015only)

# Récupérer la liste des bâtiments propres à 2016
buildings2016only = set(main2016_df['OSEBuildingID']) - set(merge_df['OSEBuildingID'])
buildings2016only = list(buildings2016only)

#%%

# Lister les bâtiments de 2015 et leurs caractéristiques
temp2015_df = main2015_df[main2015_df['OSEBuildingID'].isin(buildings2015only)]
# Eliminer la colonne OSEBuildingID
temp2015_df.drop('OSEBuildingID', axis=1, inplace=True)
for column_ in temp2015_df.columns:
    if column_ in commons_list:
        temp2015_df = temp2015_df.rename(columns={column_: column_ + '_2015'})

# Lister les bâtiments de 2016 et leurs caractéristiques
temp2016_df = main2016_df[main2016_df['OSEBuildingID'].isin(buildings2016only)]
temp2016_df.drop('OSEBuildingID', axis=1, inplace=True)
for column_ in temp2016_df.columns:
    if column_ in commons_list:
        temp2016_df = temp2016_df.rename(columns={column_: column_ + '_2016'})

# Concaténer les tableaux au tableau fusionné
merge_df = pd.concat([merge_df, temp2015_df, temp2016_df])

merge_df.shape

#%%

# Ranger les colonnes par ordre alphabétique
merge_df = merge_df.reindex(sorted(merge_df.columns), axis=1)
merge_df.head(3)

#%%

merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

# IV. Exploratory analysis

#%% md

## 1 Introduction

#%% md

Nous allons explorer les données qualitatives, puis les données quantitatives.

Nous voulons :
- estimer la pertinence de chaque catégorie de donnée,
- comprendre le contenu de chaque catégorie,
- ne retenir que les catégories vraiment utiles à notre analyse, c'est-à-dire uniquement les grandeurs dont nous pensons qu'elles peuvent avoir une influence sur les valeurs cibles (les *targets*).



#%% md

Analyses séparées des variables quantitives et des variables qualitatives

#%%

# Données qualitatives
qualitative_columns = [column for column in merge_df.columns if merge_df[column].dtype == object]
qualitative_df = merge_df[qualitative_columns]

# Données quantitatives
quantitative_columns = [column for column in merge_df.columns if merge_df[column].dtype != object]
quantitative_df = merge_df[quantitative_columns]

#%%

merge_df.shape, quantitative_df.shape, qualitative_df.shape

#%% md

Fonction de représentation des données non-NaN pour chaque colonne

#%%

def non_NaN_histogram(df):
    """Returns the proportion of non-NaN values for each column of a dataframe.
    Results are given under the form of an histogram."""
    columns_nan = {}
    for column in df.columns:
        columns_nan[column] = df[column].notna().sum()
    columns_nan = sorted(columns_nan.items(), key=lambda item: item[1], reverse=False)
    columns = [item[0] for item in columns_nan]
    nan_proportions = [item[1]/df.shape[0] for item in columns_nan]
    # Determine the color of the bars according to the type of the data
    if df[column].dtype != object:
        color = 'blue'
        title = 'Proportion de données non-Nan quantitatives'
    else:
        color = 'orange'
        title = 'Proportion de données non-Nan qualitatives'
    plt.figure(figsize=(5, math.sqrt(3*len(columns))))
    plt.title('{} ({})'.format(title, len(columns_nan)))
    plt.barh(columns, nan_proportions, color=color, edgecolor='k')

#%% md

## 2 Qualitative columns

#%% md

### a. Cleaning : how clean are our data ?

#%%

non_NaN_histogram(qualitative_df)

#%% md

La colonne Location, de 2015, contient la latitude et la longitude. Ces deux données sont contenues dans le tableau de 2016.

#%% md

Eliminons les colonnes Comments et Outlier, qui contiennent trop peu de données.

#%%

merge_df.drop(['Comments_2015', 'Outlier_2015', 'Outlier_2016'], axis=1, inplace=True)
merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

### b. Exploration : what do our data contain ?

#%% md

Et en particulier, quelles sont les colonnes susceptibles de révéler les bâtiments destinés à l'habitation ?

#%%

qualitative_df.head(3)

#%%

qualitative_df.info()

#%% md

Analyse colonne par colonne

#%%

get_pareto(merge_df, 'BuildingType_2015')

#%% md

Colonne PropertyName : Pareto des mots les plus récurrents

#%%
def most_common_words(df, column, separator, first_words=10):
    raw_series = df[column]
    words = []
    for element in raw_series:
        if type(element) == float:
            words.append(np.nan)
        else:
            for word in element.split(separator):
                words.append(word)
    short_list = Counter(words).most_common(first_words)
    words = [item[0] for item in short_list]
    df_length = df.shape[0]
    count = [item[1]/df_length*100 for item in short_list]
    return words, count
    
most_common_words(merge_df, 'PropertyName_2015', ' ')

#%% md

Colonne ListOfAllPropertyUseTypes : nettoyage

#%%

most_common_words(merge_df, 'ListOfAllPropertyUseTypes_2016', ',')

#%% md

Ces colonnes contiennent-elles des données utiles pour notre étude ?
Ce qui nous intéresse surtout, c'est l'utilisation principale du bâtiment, qui est déjà contenue dans la colonne LargestPropertyUseType. Nous pouvons donc supprimer ces deux colonnes.

#%%

merge_df.drop(['ListOfAllPropertyUseTypes_2015',
               'ListOfAllPropertyUseTypes_2016'], axis=1, inplace=True)

#%% md

Colonne YearsENERGYSTARCertified : nettoyage

Certaines cellules contiennent non pas une, mais plusieurs valeurs. Cela correspond à un bâtiment ayant été certifié à plusieurs reprises.

#%%

Attention, ça ne va pas du tout

def years_energystar_certified(df, column, separator):
    years_series = df[column]
    complete_list = []
    for element in years_series:
        if type(element) == str:
            try:
                len(element.split(separator)) > 1
                for year in element.split(separator):
                    complete_list.append(year)
            except AttributeError:
                complete_list.append(element)
    complete_list = [int(year) for year in complete_list]

complete_list = years_energystar_certified(main2016_df,
                                           'YearsENERGYSTARCertified',
                                           ',')

# Certaines "années" sont en réalité une liste d'années, ou une concaténation d'années. Séparons-les :
for _ in range(5):
    for element in complete_list:
        if type(element) == int:
            str_element = str(element)
            if len(str_element) > 4:
                new_elements = [int(str_element[i: i+4]) for i in range(0, len(str_element), 4)]
                complete_list.remove(element)
                for year in new_elements:
                    complete_list.append(year)
# Create the counter
years_dict = {}
for key, value in Counter(complete_list).items():
    years_dict[key] = value
# Plot the graph
x = range(1999, 2019)
plt.xticks(x, x, rotation = 45)
plt.xlim((1999, 2019))
#plt.ylim((0, 60))
plt.xlabel('Years')
plt.ylabel('Number of certifications')
plt.title('YearsENERGYSTARCertified by year')
plt.bar(list(years_dict.keys()), list(years_dict.values()), color='orange', edgecolor='k')

#%% md

On voit que certains bâtiments ont déjà reçu en 2016 leur certification pour 2017. S'agit-il de valeurs aberrantes, ou bien le score peut-il être décerné par avance ? Dans le doute, nous allons garder ces données.

#%% md

Nous allons remplacer cette simple colonne par autant de colonnes qu'il y a eu d'années de certification. Ce afin d'avoir des données exploitables pour la modélisation.

#%%
# Année 2015
def add_certif_year_cumsum(df, column, years_dict, prefix='ENERGYSTAR_'):
    min_year = min(list(years_dict.keys()))
    max_year = max(list(years_dict.keys()))
    years_range = range(min_year, max_year)
    temp_series = df[column]
    for year in years_range:
        year_list = []
        for element in temp_series:
            # Prévention du cas NaN
            try:
                if str(year) in element:
                    year_list.append(1)
                else:
                    year_list.append(0)
            # Cas NaN
            except TypeError:
                year_list.append(0)
        temp_df = df.copy()
    temp_df[prefix + str(year)] = year_list
    return temp_df

merge_df = add_certif_year_cumsum(merge_df, 'YearsENERGYSTARCertified_2015', years_dict)

# Année 2016
# L'année 2016 est beaucoup moins remplie, nous ne la gardons pas.

#%%

merge_df.drop('YearsENERGYSTARCertified_2015', axis=1, inplace=True)
merge_df.drop('YearsENERGYSTARCertified_2016', axis=1, inplace=True)

#%% md

Colonne Location : aperçu visuel de la répartition



#%%

# Données de 2015
def represent_location_data(df, column):
    locations_series = df[column]
    coordinates_list = []
    for element in locations_series:
        latitude = element.split(',')[0].split(':')[-1].replace(' ', '')
        latitude = latitude.split(',')[0].split(':')[-1].replace('\'', '')
        latitude = float(latitude)
        longitude = element.split(',')[1].split(':')[-1].replace(' ', '')
        longitude = longitude.split(',')[0].split(':')[-1].replace('\'', '')
        longitude = float(longitude)
        coordinates_list.append((latitude, longitude))
    latitudes = [coordinate[0] for coordinate in coordinates_list]
    longitudes = [coordinate[1] for coordinate in coordinates_list]
    plt.title('Coordonnées des bâtiments')
    plt.xlabel('Longitudes')
    plt.ylabel('Latitudes')
    plt.xticks(rotation=45)
    plt.axis('equal')
    plt.scatter(longitudes, latitudes, color='orange', edgecolors='k', )

represent_location_data(main2015_df, 'Location')

#%%

# Données de 2016

locations_df = main2016_df[['Latitude', 'Longitude']]

plt.title('Coordonnées des bâtiments')
plt.xlabel('Longitudes')
plt.ylabel('Latitudes')
plt.xticks(rotation=45)
plt.axis('equal')
plt.scatter(locations_df['Longitude'], locations_df['Latitude'], color='orange', edgecolors='k', )

#%% md

Par la suite nous utiliserons uniquement les données géographiques de 2016.

#%%

merge_df = merge_df.drop('Location', axis=1)

#%%

merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

### c. Analysis : which of our data are useful?

#%% md

Commençons par enlever les colonnes dont nous sommes sûrs qu'elles ne seront pas utiles.

#%%

merge_df.drop(['ComplianceStatus_2015',
               'ComplianceStatus_2016',
               'PropertyName_2015',
               'PropertyName_2016',
               'SecondLargestPropertyUseType_2015',
               'SecondLargestPropertyUseType_2016',
               'ThirdLargestPropertyUseType_2015',
               'ThirdLargestPropertyUseType_2016',
               'TaxParcelIdentificationNumber_2015', 
               'TaxParcelIdentificationNumber_2016'], axis=1, inplace=True)

qualitative_df.drop(['ComplianceStatus_2015',
               'ComplianceStatus_2016',
               'PropertyName_2015',
               'PropertyName_2016',
               'SecondLargestPropertyUseType_2015',
               'SecondLargestPropertyUseType_2016',
               'ThirdLargestPropertyUseType_2015',
               'ThirdLargestPropertyUseType_2016',
               'TaxParcelIdentificationNumber_2015', 
               'TaxParcelIdentificationNumber_2016'], axis=1, inplace=True)

#%% md

Les colonnes DefaultData contiennent des données similaires, mais de formats différents : Yes/No pour 2015, False/True pour 2016.

#%%

merge_df['DefaultData_2015'].value_counts()

#%%

merge_df['DefaultData_2016'].value_counts()

#%%

merge_df['DefaultData_2015'].replace({'No': 0, 'Yes':1}, inplace=True)
merge_df['DefaultData_2016'].replace({False: 0, True:1}, inplace=True)

#%%

merge_df.shape

#%% md

Objectif : n'étudier que les bâtiments non destinés à l'habitation. Nous allons identifier les colonnes pouvant donner cette information, puis filtrer par ligne.

Listons les colonnes en questions :

#%%

buildingtype_list = main2016_df['BuildingType'].unique()
primarypropertytype_list = main2016_df['PrimaryPropertyType'].unique()
largestpropertyusetype_list = list(main2016_df['LargestPropertyUseType'].unique())

#%%

largestpropertyusetype_list

#%% md

Pour chaque colonne, nous allons identifier les catégories **non liées** à l'habitation, et que nous allons retenir pour notre étude.

#%%

# Building type
buildingtype2015_list = ['NonResidential', 'Nonresidential COS', 'SPS-District K-12', 'Campus']
buildingtype2016_list = ['NonResidential', 'Nonresidential COS', 'SPS-District K-12',
                         'Campus', 'Nonresidential WA']

# Primary property type
primarypropertytype2015_list = ['Hotel', 'Other', 'Mixed Use Property', 'K-12 School',
                                'College/University', 'Small- and Mid-Sized Office',
                                'Self-Storage Facility\n', 'Distribution Center',
                                'Large Office', 'Retail Store', 'Medical Office',
                                'Hospital', 'Non-Refrigerated Warehouse', 'Distribution Center\n',
                                'SPS-District K-12', 'Worship Facility', 'Supermarket/Grocery Store',
                                'Laboratory', 'Self-Storage Facility', 'Refrigerated Warehouse',
                                'Restaurant\n', 'Restaurant']
primarypropertytype2016_list = ['Hotel', 'Other', 'Mixed Use Property', 'K-12 School',
                                'University', 'Small- and Mid-Sized Office', 'Self-Storage Facility',
                                'Warehouse', 'Large Office', 'Medical Office', 
                                'Retail Store', 'Hospital', 'Distribution Center',
                                'Worship Facility', 'Supermarket / Grocery Store',
                                'Laboratory', 'Refrigerated Warehouse', 'Restaurant',
                                'Office']

largestpropertyusetype2015_list = ['Hotel', 'Police Station', 'Other - Entertainment/Public Assembly',
                                   np.nan, 'Library', 'Fitness Center/Health Club/Gym', 'Social/Meeting Hall',
                                   'Courthouse', 'Other', 'K-12 School', 'College/University', 'Automobile Dealership',
                                   'Office', 'Self-Storage Facility', 'Retail Store', 'Senior Care Community',
                                   'Medical Office', 'Hospital (General Medical & Surgical)', 'Museum',
                                   'Repair Services (Vehicle, Shoe, Locksmith, etc)', 'Other/Specialty Hospital',
                                   'Financial Office', 'Non-Refrigerated Warehouse', 'Distribution Center',
                                   'Parking', 'Worship Facility', 'Laboratory', 'Supermarket/Grocery Store',
                                   'Convention Center', 'Urgent Care/Clinic/Other Outpatient', 'Other - Services',
                                   'Strip Mall', 'Wholesale Club/Supercenter', 'Refrigerated Warehouse',
                                   'Other - Recreation', 'Lifestyle Center', 'Other - Public Services',
                                   'Data Center', 'Other - Mall', 'Manufacturing/Industrial Plant', 'Restaurant',
                                   'Other - Education', 'Fire Station', 'Performing Arts', 'Bank Branch',
                                   'Other - Restaurant/Bar', 'Food Service', 'Adult Education', 'Other - Utility',
                                   'Movie Theater', 'Outpatient Rehabilitation/Physical Therapy',
                                   'Personal Services (Health/Beauty, Dry Cleaning, etc)', 'Pre-school/Daycare']
largestpropertyusetype2016_list = ['Hotel', 'Police Station', 'Other - Entertainment/Public Assembly',
                                   'Library', 'Fitness Center/Health Club/Gym',
                                   'Social/Meeting Hall', 'Courthouse', 'Other',
                                   'K-12 School', 'College/University', 'Automobile Dealership',
                                   'Office', 'Self-Storage Facility', 'Non-Refrigerated Warehouse',
                                   'Other - Mall', 'Medical Office', 'Retail Store',
                                   'Hospital (General Medical & Surgical)', 'Museum',
                                   'Repair Services (Vehicle, Shoe, Locksmith, etc)',
                                   'Other/Specialty Hospital', 'Financial Office',
                                   'Distribution Center', 'Parking', 'Worship Facility',
                                   'Restaurant', 'Data Center', 'Laboratory',
                                   'Supermarket/Grocery Store', 'Convention Center',
                                   'Urgent Care/Clinic/Other Outpatient', np.nan,
                                   'Other - Services', 'Strip Mall', 'Wholesale Club/Supercenter',
                                   'Refrigerated Warehouse', 'Manufacturing/Industrial Plant',
                                   'Other - Recreation', 'Lifestyle Center', 'Other - Public Services',
                                   'Other - Education', 'Fire Station', 'Performing Arts',
                                   'Bank Branch', 'Other - Restaurant/Bar', 'Food Service',
                                   'Adult Education', 'Other - Utility', 'Movie Theater',
                                   'Personal Services (Health/Beauty, Dry Cleaning, etc)',
                                   'Pre-school/Daycare', 'Prison/Incarceration']

#%% md

Filtrage des données concernant les bâtiments non destinés à l'habitation

#%%

# Building type
merge_df = merge_df[merge_df['BuildingType_2015'].isin(buildingtype2015_list)]
merge_df = merge_df[merge_df['BuildingType_2016'].isin(buildingtype2016_list)]

#%%

merge_df.shape

#%%

# Primary property type
merge_df = merge_df[merge_df['PrimaryPropertyType_2015'].isin(primarypropertytype2015_list)]
merge_df = merge_df[merge_df['PrimaryPropertyType_2016'].isin(primarypropertytype2016_list)]

#%%

merge_df.shape

#%%

# Property use type
merge_df = merge_df[merge_df['LargestPropertyUseType_2015'].isin(largestpropertyusetype2015_list)]
merge_df = merge_df[merge_df['LargestPropertyUseType_2016'].isin(largestpropertyusetype2016_list)]

#%%

merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

Le nombre de suppressions effectuées par les deux derniers nettoyages confirme le premier : plus de la moitié des bâtiments enregistrés étaient des bâtiments d'habitation.

#%% md

## 3 Quantitative columns

#%% md

### a. Cleaning : how clean are our data ?

#%%

non_NaN_histogram(quantitative_df)

#%%

quantitative_df.describe()

#%% md

Quelques données sont négatives, là où elles devraient être positives. Elles sont aberrantes et doivent être rejetées.

Colonnes concernées : PropertyGFAParking, PropertyGFABuilding(s), SourceEUI(kBtu/sf), SourceEUIWN(kBtu/sf)


Cela dit, il ne suffit pas de simplement rejeter ces données aberrantes en dessous de zéro. Il y a sûrement d'autres données erronées, certes au-dessus de zéro, mais sujettes à la même probabilité d'erreur. Il faudrait enquêter sur les raisons qui font que ces données sont apparues, et éradiquer la possibilité qu'à l'avenir, de telles données erronées ré-apparaissent.

#%%

erratic_list = ['Electricity(kBtu)_2016', 'Electricity(kWh)_2016', 'GHGEmissionsIntensity_2016',
                'GHGEmissionsTotal_2016', 'PropertyGFABuilding(s)_2015', 'PropertyGFAParking_2015',
                'PropertyGFAParking_2016', 'SourceEUIWN(kBtu/sf)_2015', 'SourceEUIWN(kBtu/sf)_2015',
                'SourceEUI(kBtu/sf)_2015']

for column_name in erratic_list:
    quantitative_df = quantitative_df[quantitative_df[column_name] >= 0]
    merge_df = merge_df[merge_df[column_name] >= 0]

#%%

merge_df.shape

#%%

# Certains éléments sont des NaN. Ils ne sont pas vraiment représentatifs (moins de 0,2 %), nous pouvons les éliminer.
merge_df = merge_df[merge_df['NumberofFloors_2015'].notnull()]
merge_df = merge_df[merge_df['NumberofFloors_2016'].notnull()]

quantitative_df = quantitative_df[quantitative_df['NumberofFloors_2015'].notnull()]
quantitative_df = quantitative_df[quantitative_df['NumberofFloors_2016'].notnull()]

#%%

merge_df.shape

#%% md

**Traitement du cas particulier de DefaultData**

La colonne DefaultData contient des valeurs booléennes (True ou False). Ceci va nous contrarier pour le Feature Engineering.

Nous allons donc remplacer ces valeurs par des "1" pour True et des "0" pour False.

#%%

merge_df.replace({True:1, False:0}, inplace=True)
quantitative_df.replace({True:1, False:0}, inplace=True)

#%% md

**Elimination des colonnes contenant trop de valeurs NaN**

On compte notamment : 2010 Census Tracts & City Council Districts

#%%

merge_df.drop(['2010 Census Tracts', 'City Council Districts'], axis=1, inplace=True)
quantitative_df.drop(['2010 Census Tracts', 'City Council Districts'], axis=1, inplace=True)

#%% md

**SPD Beats**

#%%

len(merge_df['SPD Beats'].value_counts())

#%%

merge_df.drop('SPD Beats', axis=1, inplace=True)

#%% md

**Seattle Police Department Micro Community Policing Plan Areas**

#%%

len(merge_df['Seattle Police Department Micro Community Policing Plan Areas'].value_counts())

#%%

merge_df.drop('Seattle Police Department Micro Community Policing Plan Areas', axis=1, inplace=True)

#%% md

**Colonne DataYear**

Cette colonne contient toujours la même valeur. Nous pouvons la rejeter sans perdre d'information.

#%%
Attention, ça ne va pas du tout

merge_df['DataYear_2015'].value_counts()

#%%

merge_df['DataYear_2016'].value_counts()

#%%

merge_df.drop(['DataYear_2015', 'DataYear_2016'], axis=1, inplace=True)

#%% md

**Colonne NumberofBuildings_2016**

Certaines données indiquent un nombre nul de bâtiment (=0) : ces données sont erronées, remplaçons-les par la valeur *None*.

#%%

merge_df['NumberofBuildings_2016'].replace(0, None, inplace=True)

#%% md

**Colonne OtherFuelUse(kBtu)**

Cette colonne ne semble pas intéressante pour notre étude : 
*   Sa description n'est pas disponible sur le site de Seattle City
*   Elle contient une grande majorité de valeurs nulles.



#%%

merge_df['OtherFuelUse(kBtu)'].value_counts()

#%%

merge_df.drop('OtherFuelUse(kBtu)', axis=1, inplace=True)
quantitative_df.drop('OtherFuelUse(kBtu)', axis=1, inplace=True)

#%% md

**Colonne SteamUse(kBtu)_2015**

#%%

merge_df['SteamUse(kBtu)_2015'].value_counts()

#%%

merge_df.drop('SteamUse(kBtu)_2015', axis=1, inplace=True)
quantitative_df.drop('SteamUse(kBtu)_2015', axis=1, inplace=True)

#%% md

**Colonne SteamUse(kBtu)_2016**

#%%

merge_df['SteamUse(kBtu)_2016'].value_counts()

#%%

merge_df.drop('SteamUse(kBtu)_2016', axis=1, inplace=True)
quantitative_df.drop('SteamUse(kBtu)_2016', axis=1, inplace=True)

#%% md

**Colonne ThirdLargestPropertyUseTypeGFA**

Cette colonne contient très peu de valeurs. Il vaut mieux la rejeter.

#%%

merge_df['ThirdLargestPropertyUseTypeGFA_2015'].value_counts()

#%%

merge_df.drop(['ThirdLargestPropertyUseTypeGFA_2015', 'ThirdLargestPropertyUseTypeGFA_2016'], axis=1, inplace=True)
quantitative_df.drop(['ThirdLargestPropertyUseTypeGFA_2015', 'ThirdLargestPropertyUseTypeGFA_2016'], axis=1, inplace=True)

#%% md

**Colonne SecondLargestPropertyUseTypeGFA**

Cette colonne contient assez peu de valeurs, il vaut mieux la rejeter également.

#%%

merge_df['SecondLargestPropertyUseTypeGFA_2015'].value_counts()

#%%

merge_df.drop(['SecondLargestPropertyUseTypeGFA_2015', 'SecondLargestPropertyUseTypeGFA_2016'], axis=1, inplace=True)
quantitative_df.drop(['SecondLargestPropertyUseTypeGFA_2015', 'SecondLargestPropertyUseTypeGFA_2016'], axis=1, inplace=True)

#%%

merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

### b. Exploration : what do our data contain?



#%%

merge_df.columns

#%%

my_explorator.plot_feature(merge_df, 'ENERGYSTARScore_2015')

#%% md

**Focus sur l'évolution de l'ENERGYSTARScore entre 2015 et 2016**

Sur l'ensemble des bâtiments, l'année 2016 a-t-elle été meilleure que l'année 2015 ?

#%%

def plot_features_differences(df, column_1, column_2):
    years_diff_series = df[column_1] - df[column_2]
    ax = plt.axes()
    ax.yaxis.grid()
    plt.title('Difference between {} and {}'.format(column_1, column_2))
    plt.hist(years_diff_series, bins=100, color='blue', edgecolor='k')

plot_features_differences(merge_df, 'ENERGYSTARScore_2016', 'ENERGYSTARScore_2015')

#%% md

**Focus sur les étiquettes**

Nous souhaitons avoir seulement deux colonnes d'étiquettes. Actuellement, nous avons 4 colonnes, 2 en 2015 et 2 en 2016.

Voyons les différences entre les deux années.

#%%

plot_features_differences(merge_df, 'SiteEnergyUseWN(kBtu)_2016', 'SiteEnergyUseWN(kBtu)_2015')

#%%

merge_df['GHGEmissionsTotal_2015'].describe()

#%%

merge_df['GHGEmissionsTotal_2016'].describe()

#%%

plot_features_differences(merge_df, 'GHGEmissionsTotal_2016', 'GHGEmissionsTotal_2015')

#%% md

Les colonnes étiquettes présentent des caractéristiques statistiques similaires. Nous allons fusionner les deux années en calculant la moyenne par bâtiment. 

#%%

merge_df['SiteEnergyUseWN_average'] = (merge_df['SiteEnergyUseWN(kBtu)_2015'] + merge_df['SiteEnergyUseWN(kBtu)_2016'])/2
merge_df.drop('SiteEnergyUseWN(kBtu)_2015', axis=1, inplace=True)
merge_df.drop('SiteEnergyUseWN(kBtu)_2016', axis=1, inplace=True)

merge_df['GHGEmissionsTotal_average'] = (merge_df['GHGEmissionsTotal_2015'] + merge_df['GHGEmissionsTotal_2016'])/2
merge_df.drop('GHGEmissionsTotal_2015', axis=1, inplace=True)
merge_df.drop('GHGEmissionsTotal_2016', axis=1, inplace=True)

#%%

def log_transformation(df, column):
    new_df = df.copy()
    new_df['log('+column+')'] = math.log(new_df[column])
    new_df.drop(column, axis=1, inplace=True)
    return new_df

merge_df = log_transformation(merge_df, 'GHGEmissionsTotal_average')

#%%

my_explorator.plot_feature(merge_df, 'log(GHGEmissionsTotal_average)', quantile_sup=0.99)

#%%

merge_df = log_transformation(merge_df, 'SiteEnergyUseWN_average')

#%%

merge_df.columns

#%%

merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

### c. Analysis : which of our data are useful?

#%% md

Redéfinition des colonnes quantitatives

#%%

# Données quantitatives
quantitative_columns = [column for column in merge_df.columns if merge_df[column].dtype != object]
quantitative_df = merge_df[quantitative_columns]

#%% md

First objective is to evaluate the interest of ENERGY STAR score for emissions prediction. It leads us to choose the useful columns accordingly.

#%% md

GHGEmissionsIntensity

#%%

get_scatter('ENERGYSTARScore_2016', 'SiteEnergyUseWN_average')

#%%

get_scatter('ENERGYSTARScore_2016', 'GHGEmissionsTotal_average')

#%%

correlation_consumption = {'ZipCode':[0.003656, 0.002976],
                           'YearBuilt_2016':[0.005729, 0.000657],
                           'YearBuilt_2015':[0.005729, 0.000657],
                           'PropertyGFATotal':[0.433141, 0.242183],
                           'PropertyGFABuilding(s)':[0.451066, 0.274090],
                           'NumberofFloors_2016':[0.119185, 0.0350338],
                           'NumberofBuildings_2015':[0.031333, 0.018412],
                           'Longitude':[0.000992, 0.001195],
                           'Latitude':[0.000950, 0.000284],
                           'DefaultData':[0.000772, 0.000772],
                           'DataYear':[0, 0],
                           'Seattle Police Department Micro Community Policing Plan Areas':[4.70361e-05, 0.000634],
                           'SPD Beats':[0.001655, 0.001588],
                           'SiteEnergyUse(kBtu)':[0.996577, 0.794572],
                           'SiteEnergyUseWN(kBtu)':[1.0, 0.797673],
                           'SiteEUIWN(kBtu/sf)':[0.192727, 0.117020],
                           'SiteEUI(kBtu/sf)':[0.198178, 0.119329],
                           'SteamUse(kBtu)_2016':[0.354672, 0.117020],
                           'SourceEUIWN(kBtu/sf)':[0.197180, 0.078598],
                           'SourceEUI(kBtu/sf)':[0.200160, 0.081182],
                           'NumberofBuildings_2015':[0.031334, 0.018412],
                           'NaturalGas(therms)_2016':[0.377854, 0.427167],
                           'GHGEmissionsTotal':[0.797673, 1.0],
                           'GHGEmissionsIntensity':[0.122309, 0.182920],
                           'Electricity(kWh)':[0.870812, 0.476882],
                           'NumberofFloors_2015':[0.119906, 0.035292],
                           'SteamUse(kBtu)_2015':[0.344648, 0.585326],
                           'OtherFuelUse(kBtu)':[0.001977, 0.001611],
                           'NaturalGas(therms)_2015':[0.384946, 0.427094],
                           'LargestPropertyUseTypeGFA':[0.495779, 0.333422],
                           'ENERGYSTARScore':[0.006413, 0.008733]}

#%%

correl_consum_df =  pd.DataFrame(data=correlation_consumption).transpose()
correl_consum_df.sort_values(by=[1], inplace=True)
temp_length = len(correl_consum_df)

plt.figure(figsize=(math.sqrt(temp_length*2), math.sqrt(temp_length*2)))
plt.title('Regression coefficient toward GHGEmissionsTotal')
plt.barh(correl_consum_df.index, correl_consum_df[1], color='blue', edgecolor='k')

#%% md

**Comparaison des corrélations entre colonnes quantitatives**

#%%

my_explorator.quant_heatmap(quantitative_df)

#%% md

Supprimons les colonnes blanches

#%%

blank_columns = ['Comments_2016', 'ENERGYSTAR_2001']
quantitative_df.drop(blank_columns, axis=1, inplace=True)
merge_df.drop(blank_columns, axis=1, inplace=True)

#%%

my_explorator.quant_heatmap(quantitative_df)

#%% md

**Quelques observations**

Certaines colonnes sont trop correlées entre elles (coefficient de corrélation > 0,9).

N.B.: C'est le signe que notre choix de dédoubler certaines colonnes, lors de la fusion des tableaux 2015 et 2016, était inadapté.

De plus, l'enjeu des modélisations à venir est de se passer des relevés annuels, c'est à des colonnes suivantes : 
Electricity, GHGEmissionsIntensity, NaturalGas, SiteEUI, SiteEUIWN, SiteEnergyUse, SourceEUI, SourceEUIWN


#%%

correl_columns = ['CouncilDistrictCode_2015',
                  'ENERGYSTARScore_2015',
                  'ENERGYSTAR_2002',
                  'Electricity(kBtu)_2015',
                  'Electricity(kBtu)_2016',
                  'Electricity(kWh)_2015',
                  'Electricity(kWh)_2016',
                  'GHGEmissionsIntensity_2015',
                  'GHGEmissionsIntensity_2016',
                  'LargestPropertyUseTypeGFA_2015',
                  'NaturalGas(therms)_2015',
                  'NaturalGas(therms)_2016',
                  'NaturalGas(kBtu)_2015',
                  'NaturalGas(kBtu)_2016',
                  'NumberofFloors_2015',
                  'PropertyGFABuilding(s)_2015',
                  'PropertyGFABuilding(s)_2016',
                  'PropertyGFAParking_2015',
                  'PropertyGFATotal_2015',
                  'PropertyGFATotal_2016',
                  'SiteEUI(kBtu/sf)_2015',
                  'SiteEUI(kBtu/sf)_2016',
                  'SiteEUIWN(kBtu/sf)_2015',
                  'SiteEUIWN(kBtu/sf)_2016',
                  'SiteEnergyUse(kBtu)_2015',
                  'SiteEnergyUse(kBtu)_2016',
                  'SourceEUI(kBtu/sf)_2015',
                  'SourceEUI(kBtu/sf)_2016',
                  'SourceEUIWN(kBtu/sf)_2015',
                  'SourceEUIWN(kBtu/sf)_2016',
                  'YearBuilt_2015',
                  'ZipCode_2015',# Identifiant arbitraire
                  'ZipCode_2016'# Identifiant arbitraire
                  ]
quantitative_df.drop(correl_columns, axis=1, inplace=True)
merge_df.drop(correl_columns, axis=1, inplace=True)

#%%

my_explorator.quant_heatmap(quantitative_df)

#%%

merge_df.shape

#%%

NaN_proportion(merge_df)

#%% md

# V.  Modélisations

#%% md

## 0 Introduction

#%% md

Nous allons scinder nos modélisations en deux parties :
- les modélisations visant à prédire la **consommation en énergie**,
- les modélisations visant à prédire les **émissions de CO2**,

#%% md

Emissions de CO2 et consommation d'énergie sont des données **qualitatives**. Il s'agit donc d'une problématique de **régression** (et non de classification).

#%% md

Nous évaluerons les modèles choisis à l'aide de trois critères :
- la **précision**, estimée via des méthodes propres à scikitlearn,
- le **temps de calcul**, estimé via un simple %timeit,
- la **complexité algorithmique**.

#%%

# Déclaration de la matrice X. On ne prend pas en compte les targets.
merge_df = merge_df.dropna()
X = merge_df.drop(['log(SiteEnergyUseWN_average)', 'log(GHGEmissionsTotal_average)'], axis=1)

# Déclaration des vecteurs targets
y_consumptions = merge_df['log(SiteEnergyUseWN_average)']
y_emissions = merge_df['log(GHGEmissionsTotal_average)']
targets = y_consumptions
temp_title = targets.columns[0][4:-9]

#%% md

Fonction d'activation de la caractéristique ENERGYSTARScore

#%%

# On active ou non la ligne suivant que l'on veut rejeter ou garder la caractéristique ENERGYSTARScore
if False:
    X.drop(['ENERGYSTARScore_2016',
            'ENERGYSTAR_2000','ENERGYSTAR_2003','ENERGYSTAR_2004','ENERGYSTAR_2005','ENERGYSTAR_2006',
            'ENERGYSTAR_2007','ENERGYSTAR_2008','ENERGYSTAR_2009','ENERGYSTAR_2010','ENERGYSTAR_2011',
            'ENERGYSTAR_2012','ENERGYSTAR_2013','ENERGYSTAR_2014','ENERGYSTAR_2015','ENERGYSTAR_2016'],
           axis=1, inplace=True)

#%% md

La sélection de modèle montre que les performances de prédiction sont meilleures avec les ENERGYSTARScore. Nous allons les garder.

#%% md

## 1 Feature Engineering

#%% md

### a. One-hot encoding

#%% md

Le OneHotEncoder s'applique aux données catégorielles (qualitatives).

#%%

get_pareto(X, 'BuildingType_2015')

#%%

X_qualitative.drop('OSEBuildingID', axis=1, inplace=True)

#%%

encoder = OneHotEncoder(handle_unknown='ignore')
X_encoded = encoder.fit_transform(X_qualitative)
X_encoded = X_encoded.toarray()
X_encoded.shape

#%% md

Le OneHotEncoder transforme les colonnes catégorielles en 102 colonnes.

Cela semble beaucoup, voyons si nous pouvons obtenir un meilleur résultat en modifiant la modélisation : nous allons rejeter les éléments comptant pour moins de 1% de la colonne (c'est-à-dire, pour une colonne de 1491 éléments, moins de 14 occurences). Ils sont à l'origine de colonnes contenant un "1", puis le reste de "0", et ont une faible utilité pour les modélisations futures.

#%%

# Repérage des valeurs de chaque colonne présentes à plus de n%
df_length = len(X_qualitative)
rate = 0.01
representative_values =  {}
for column in X_qualitative:
    temp_series = X_qualitative[column]
    values_count = temp_series.value_counts()
    representative_values[column] = [element for element in temp_series.unique() if values_count[element] > rate * df_length]

#%%

encoder_v2 = OneHotEncoder(handle_unknown='ignore', categories=list(representative_values.values()))
X_encoded = encoder_v2.fit_transform(X_qualitative)
X_encoded = X_encoded.toarray()
X_encoded.shape

#%%

X['OSEBuildingID'] = X['OSEBuildingID'].astype('int')
qualitative_columns = X.select_dtypes(['object']).columns
onehot_qualitative_columns = encoder_v2.get_feature_names(input_features=None)
X_qualitative = pd.DataFrame(X_transformed, columns=onehot_qualitative_columns)
X_qualitative.shape

#%%

NaN_proportion(X_qualitative)

#%%

X.drop(qualitative_columns, axis=1, inplace=True)
X = pd.merge(X.reset_index(), X_qualitative,
             left_index=True, right_index=True)
# On remplace les espaces vides par des underscores pour éviter les éventuels dysfonctionnements lors de l'utilisation de XGBoost
X.rename(columns = lambda x: x.replace(' ', '_'), inplace=True)
X

#%%

X.shape

#%%

NaN_proportion(X)

#%% md

### b. Standard Scaler

#%%

# On écarte les colonnes créées par le OneHotEncoder, qui commencent par x0_ ..., x1_ ...
X_quantitative = X.filter(regex=('^(?!x\d+_)'))
quantitative_columns = X_quantitative.columns

#%% md

**Distributions des features quantitatives**

#%%

sns.distplot(X_quantitative)

#%% md

**Standardisation des colonnes quantitatives**

Pour chaque colonne de caractéristique quantitative, on ramène la plus grande partie des valeurs dans l'intervalle [-1 ; 1]

#%%

def scale_dataframe(df):
    # Scale quantitative columns
    scaler = StandardScaler()
    df_quant = df.filter(regex=('^(?!x\d+_)'))
    df_quant = scaler.fit_transform(df_quant)
    df_quant = pd.DataFrame(df_quant, columns=df_quant.columns)
    # Modify the original dataframe
    df.drop(df_quant.columns, axis=1, inplace=True)
    for column in df_quant.columns:
        df[column] = df_quant[column]
    return df


X_quantitative = scale_quantitative_dataframe(X_quantitative)

#%% md

**Distribution des features quantitaives après StandardScaler**

#%%

sns.distplot(X_quantitative)

#%%

X.shape

#%%

NaN_proportion(X)

#%% md

## 2 Model Selection

#%% md

### Préparation

#%% md

Chaque modèle sera optimisé et évalué selon les étapes suivantes : 

1.   **Split** 80/20 de l'ensemble de départ (train set / test set)
2.   Réduction en composants principaux (**PCA**), puis split 80/20 de nouveau
1.   Validation croisée (**CV**)
2.   Validation croisée mélangée (**SCV**)
1.   Utilisation d'un sous-ensemble de validation pour trouver les paramètres optimaux (**Validation Set**)
2.   Recherche sur grille (**GridSearchCV**)
2.   Validation croisée imbriquée (**GridSearchCV + CV** & **Gridsearch + SCV**)

#%%

# Performance dataframe on test set
scores_df = pd.DataFrame(columns=['Model', 'Config', 'Métrique', 'Score', 'Rapidité'])
# Performance dataframe on train set
scores_train_df = pd.DataFrame(columns=['Model', 'Config', 'Métrique', 'Score', 'Rapidité'])
# Split
X_train, X_test, y_train, y_test = train_test_split(X, targets, test_size=0.2, random_state=1)

#%%

def best_PCA_number(metrics, model, range_max, range_min, data=X, range_step=1):
    '''Performances de prédiction d'un modèle selon le nombre de composants principaux'''
    pca_mse = {}
    pca_score = {}
    # Récupérer les valeurs
    for n_components in range(range_min, range_max, range_step):
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(data)
        Xtrain_pca, Xtest_pca, ytrain_pca, ytest_pca = train_test_split(X_pca, targets,
                                                                        test_size=0.2, random_state=1)
        reg = model.fit(Xtrain_pca, ytrain_pca)
        if metrics_ == 'mse':
            model_mse = mean_squared_error(y_true=ytest_pca, y_pred=reg.predict(Xtest_pca))
            pca_mse[n_components] = model_mse
        elif metrics_ == 'score':
            model_score = reg.score(Xtest_pca, ytest_pca)
            pca_score[n_components] = model_score
    # Graph
    plt.xlabel('Nombre de composants principaux')
    plt.xticks(range(range_min, range_max))
    ax = plt.axes()
    if metrics_ == 'mse':
        plt.ylim(0, 10)
        plt.title('Erreur quadratique en fonction du nombre de composants principaux')
        plt.ylabel('Erreur quadratique')
    elif metrics_ == 'score':
        fig, ax = plt.subplots()
        ax.xaxis.tick_top()
        plt.ylabel('Score du modèle')
        plt.ylim(-10, 2)
        plt.title('Score moyen en fonction du nombre de composants principaux')
    ax.yaxis.grid()
    plt.bar(list(pca_score.keys()), list(pca_score.values()),
                 color='orange', edgecolor='k')

#%% md

### a LinearRegression

#%% md

L'idée est de déterminer une relation du type : 

y[k] = w[0] \* x[k, 0] + w[1] \* x[k, 1] + ....... + w[p] \* x[k, p]

où y[k] est la k-ième target du vecteur target y, et (x[k, 0], x[k, 1], ..., x[k, p]) sont les éléments de la matrice X, correspondant à la k-ième ligne de la matrice.

#%%

# Enregistrement du nom du modèle
temp_model = 'LinearRegression'

#%% md

**1. Split 80/20**

#%%

# Enregistrement du nom de la configuration (étape d'optimisation)
temp_config = 'Split'

#%%

# Métrique R2
temp_metric = 'R2'
time1 = time.time()
lr = LinearRegression()
reg = lr.fit(X_train, y_train)
temp_value = round(reg.score(X_test, y_test), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

# Métrique MSE
temp_metric = 'RMSE'
time1 = time.time()
lr = LinearRegression()
reg = lr.fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_true=y_test, y_pred=reg.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**2. PCA**

Comment déterminer le nombre de composants principaux ? Lequel donnera la meilleure performance ? Nous allons calculer la performance du modèle pour différents nombres de composants principaux.

#%%

temp_config = 'PCA'

#%%

time0 = time.time()
best_PCA_number(model_=LinearRegression(),
                metrics_='mse',
                data_=X,
                range_min_=2,
                range_max_=30)
pca_delta = time.time() - time0

#%% md

Le meilleur score est obtenu pour 7 composants principaux, et ce pour les deux métriques R2 et RMSE : calculons les performances avec ce nombre.

#%%

temp_metric = 'R2'
time1
pca = PCA(n_components=7)
X_PCA = pca.fit_transform(X)
lr = LinearRegression()
Xtrain_pca, Xtest_pca, ytrain_pca, ytest_pca = train_test_split(X_PCA, targets,
                                                                test_size=0.2, random_state=1)
reg = lr.fit(Xtrain_pca, ytrain_pca)
temp_value = round(reg.score(Xtest_pca, ytest_pca), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
pca = PCA(n_components=7)
X_PCA = pca.fit_transform(X)
lr = LinearRegression()
Xtrain_pca, Xtest_pca, ytrain_pca, ytest_pca = train_test_split(X_PCA,
                                                                targets,
                                                                test_size=0.2,
                                                                random_state=1)
reg = lr.fit(Xtrain_pca, ytrain_pca)
temp_value = round(np.sqrt(mean_squared_error(y_true=ytest_pca, y_pred=reg.predict(Xtest_pca))), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

Les performances sont meilleures avec le PCA. Nous allons l'utiliser.

#%% md

**3. Validation croisée**

#%%

nb_fold = 5

#%%

temp_config = 'CV'

#%%

temp_metric = 'R2'
time1 = time.time()
lr = LinearRegression()
scores = cross_val_score(lr,
                         Xtrain_pca,
                         ytrain_pca,
                         cv=nb_fold,
                         scoring='r2')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
lr = LinearRegression()
scores = cross_val_score(lr,
                         Xtrain_pca,
                         ytrain_pca,
                         cv=nb_fold,
                         scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

kfold = KFold(n_splits=nb_fold)

#%%

temp_config = 'SCV'

#%%

temp_metric = 'R2'
time1 = time.time()
lr = LinearRegression()
scores = cross_val_score(lr,
                         Xtrain_pca,
                         ytrain_pca,
                         cv=kfold,
                         scoring='r2')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
lr = LinearRegression()
scores = cross_val_score(lr,
                         Xtrain_pca,
                         ytrain_pca,
                         cv=kfold,
                         scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### b Ridge

#%% md

Nous allons ajouter une contrainte au modèle de régression linéaire : nous allons faire en sorte de minimiser l'ensemble des coefficients de pondération w[i].

En **complexifiant** ainsi notre modèle, nous le rendons certes plus difficile à gérer, mais plus **capable de coller** aux données et à leur variabilité.

#%%

temp_model = 'Ridge'

#%% md

**1. Split 80/20**

#%%

temp_config = 'Split'

#%%

temp_metric = 'R2'
time1 = time.time()
ridge = Ridge().fit(X_train, y_train)
temp_value = round(ridge.score(X_test, y_test), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
ridge = Ridge().fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_true=y_test, y_pred=ridge.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**2. PCA**

#%% md

Le nombre de 10 composants principaux (PCA) a été déterminé au chapitre précédent pour le modèle LinearRegression. Déterminons-le pour le modèle Ridge.

#%%

temp_config = 'PCA'

#%%

time0 = time.time()
best_PCA_number(model=Ridge(), metrics='score',
                data=X, range_min=2, range_max=30)
pca_delta = time.time() - time0

#%%

temp_metric = 'R2'
time1 = time.time()
pca = PCA(n_components=7)
ridge = Ridge().fit(Xtrain_pca, ytrain_pca)
temp_value = round(ridge.score(Xtest_pca, ytest_pca), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
pca = PCA(n_components=7)
ridge = Ridge().fit(Xtrain_pca, ytrain_pca)
temp_value = round(np.sqrt(mean_squared_error(y_true=ytest_pca, y_pred=ridge.predict(Xtest_pca))), 2)
time_delta = round(time.time() - time1 + pca_delta, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

Le PCA ne semble pas profiter au modèle Ridge, et détériore ses performances. Nous n'allons pas l'utiliser pour les validations croisées.

#%% md

**3. Validation croisée**

#%%

temp_config = 'CV'

#%%

temp_metric = 'R2'
time1 = time.time()
ridge = Ridge()
scores = cross_val_score(ridge, X_train, y_train,
                         cv=nb_fold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
ridge = Ridge()
scores = cross_val_score(ridge, X_train, y_train,
                         cv=nb_fold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'

#%%

temp_metric = 'R2'
time1 = time.time()
ridge = Ridge()
scores = cross_val_score(ridge, X_train, y_train,
                         cv=kfold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
ridge = Ridge()
scores = cross_val_score(ridge, X_train, y_train,
                         cv=kfold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### c Lasso

#%% md

Dans ce modèle, nous reprenons la regression linéaire et, un peu de la même manière qu'avec le Ridge, nous allons ajouter une contrainte aux coefficients.

Mais pour le Lasso, le calcul de la regression linéraire est fait de telle manière que les valeurs absolues des coefficients sont réduites individuellement. A la différence du Ridge, certains coefficients peuvent être réduits à zéro. Cela conduit à une sélection automatique des caractéristiques.

#%%

temp_model = 'Lasso'

#%% md

**1. Split 80/20**

#%%

temp_config = 'Split'
X_train, X_test, y_train, y_test = train_test_split(X, targets, random_state=1)

#%%

temp_metric = 'R2'
time1 = time.time()
lasso = Lasso().fit(X_train, y_train)
temp_value = round(lasso.score(X_test, y_test), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
lasso = Lasso().fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_true=y_test, y_pred=lasso.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**2. PCA**

#%%

temp_config = 'PCA'

#%%

best_PCA_number(model_=Lasso(), metrics_='mse', data_=X,
                range_min_=2, range_max_=15)

#%% md

Le PCA ne semble pas apporter de la précision à notre modèle.

#%%

temp_metric = 'R2'
temp_value = 'Not relevant'
time_delta = 'Not relevant'
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
temp_value = 'Not relevant'
time_delta = 'Not relevant'
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**3. Validation croisée**

#%%

temp_config = 'CV'

#%%

temp_metric = 'R2'
time1 = time.time()
scores = cross_val_score(lasso, X_train, y_train,
                         cv=nb_fold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
scores = cross_val_score(lasso, X_train, y_train,
                         cv=nb_fold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'
kfold = KFold(n_splits=nb_fold)

#%%

temp_metric = 'R2'
time1 = time.time()
scores = cross_val_score(lasso, X_train, y_train,
                         cv=kfold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
scores = cross_val_score(lasso, X_train, y_train,
                         cv=kfold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### d ElasticNet

#%% md

L'ElasticNet est une sorte de combinaison des deux modèles précédents, Ridge et Lasso.

Il prend des hyperparamètres régulant ces deux modèles, ce qui amène à estimer ses performances à l'aide d'une GridSearch.

#%%

temp_model = 'ElasticNet'

#%% md

**1. Split 80/20**

#%%

temp_config = 'Split'
X_train, X_test, y_train, y_test = train_test_split(X, targets, random_state=0)

#%%

temp_metric = 'R2'
time1 = time.time()
elasnet = ElasticNet().fit(X_train, y_train)
temp_value = round(elasnet.score(X_test, y_test), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
elasnet = ElasticNet().fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_test, elasnet.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**2. PCA**

#%%

temp_config = 'PCA'

#%%

best_PCA_number(model_=ElasticNet(), metrics_='score', data_=X,
                range_min_=2, range_max_=30)

#%% md

Le PCA ne semble pas apporter de précision supplémentaire à notre modèle.

#%%
temp_metric = 'R2'
temp_value = 'Not relevant'
time_delta = 'Not relevant'
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
temp_value = 'Not relevant'
time_delta = 'Not relevant'
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**3. Validation croisée**

#%%

temp_config = 'CV'

#%%

temp_metric = 'R2'
time1 = time.time()
elasnet = ElasticNet()
scores = cross_val_score(elasnet, X_train, y_train,
                         cv=nb_fold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
elasnet = ElasticNet()
scores = cross_val_score(elasnet, X_train, y_train,
                         cv=nb_fold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'
kfold = KFold(n_splits=nb_fold)

#%%

temp_metric = 'R2'
time1 = time.time()
Elasnet = ElasticNet()
scores = cross_val_score(elasnet, X_train, y_train,
                         cv=kfold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
Elasnet = ElasticNet()
scores = cross_val_score(elasnet, X_train, y_train,
                         cv=kfold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### e DecisionTreeRegressor

#%% md

L'exemple du DecisionTreeRegressor sur le site internet de scikit-learn est donné pour deux composants.

#%%

temp_model = 'DecisionTreeRegressor'

#%% md

**1. Split 80/20**

#%%

temp_config = 'Split'
X_train, X_test, y_train, y_test = train_test_split(X, targets, random_state=0)

#%%

temp_metric = 'R2'
time1 = time.time()
DecTree = DecisionTreeRegressor(min_samples_split=3)
DecTree.fit(X_train, y_train)
temp_value = round(DecTree.score(X_test, y_test), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
DecTree = DecisionTreeRegressor(min_samples_split=3)
DecTree.fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_test, DecTree.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**3. Validation croisée**

#%%

temp_config = 'CV'

#%%

temp_metric = 'R2'
time1 = time.time()
scores = cross_val_score(DecTree, X_train, y_train,
                         cv=nb_fold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
scores = cross_val_score(DecTree, X_train, y_train,
                         cv=nb_fold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'

#%%

temp_metric = 'R2'
time1 = time.time()
scores = cross_val_score(DecTree, X_train, y_train,
                         cv=kfold, scoring='r2')
temp_score = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
scores = cross_val_score(DecTree, X_train, y_train,
                         cv=kfold, scoring='neg_root_mean_squared_error')
temp_rmse = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### f RandomForest

#%% md

Nous allons construire de nombreux arbres de décision, chacun devant faire un travail de prédiction acceptable, tout en étant différent des autres arbres. L'expression forêt aléatoire vient de cette multiplication des arbres, ainsi que de l'injection d'une part d'aléatoire dans leur construction afin de s'assurer qu'ils sont tous différents.

Source : "Le Machine Learning avec Python", Mueller & Guido

#%%

temp_model = 'RandomForest'

#%% md

**1. Split 80/20**

#%%

temp_config = 'Split'

#%% md

Les RandomForestRegressor n'acceptent que mse et mae comme métriques

#%%

temp_metric = 'RMSE'
time1 = time.time()
RFR = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=50)
RFR.fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_test, RFR.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**2. PCA**

#%% md

Le modèle nécessite trop de temps de calcul pour pouvoir faire trop de simulations.

#%%



#%% md

**3. Validation croisée**


#%%

temp_config = 'CV'

#%%

temp_metric = 'RMSE'
time1 = time.time()
RFR = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=50)
scores = cross_val_score(RFR, X_train, y_train,
                         cv=nb_fold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'

#%%

temp_metric = 'RMSE'
time1 = time.time()
RFR = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=50)
scores = cross_val_score(RFR, X_train, y_train,
                         cv=kfold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### g GradientBoosting

#%% md

A la différence de l'approche utilisée pour les forêts aléatoires, le *gradient boosting* fonctionne en construisant des arbres de manière sérielle, chaque arbre essayant de corriger les erreurs faites par le précédent.

Source : "Le Machine Learning avec Python", Mueller & Guido

#%%

temp_model = 'GradientBoosting'

#%% md

**1. Split 80/20**

#%%

temp_config = 'Split'

#%%

temp_metric = 'R2'
time1 = time.time()
gbrt = GradientBoostingRegressor(random_state=0)
gbrt.fit(X_train, y_train)
temp_value = round(gbrt.score(X_test, y_test), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
gbrt = GradientBoostingRegressor(random_state=0)
gbrt.fit(X_train, y_train)
temp_value = round(np.sqrt(mean_squared_error(y_test, gbrt.predict(X_test))), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

La performance est meilleure sur le jeu d'entraînement que sur le jeu de test : il y a peut-être un surapprentissage.

Voyons si nous pouvons réduire ce surapprentissage, en agissant notamment sur la profondeur maximale des arbres et sur le niveau d'apprentissage d'un arbre à l'autre (learning_rate), dans la partie Gridsearch.

#%% md

**3. Validation croisée**

#%%

temp_config = 'CV'

#%%

temp_metric = 'R2'
time1 = time.time()
gbrt = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(gbrt, X_train, y_train,
                         cv=nb_fold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
gbrt = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(gbrt, X_train, y_train,
                         cv=nb_fold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'

#%%

temp_metric = 'R2'
time1 = time.time()
gbrt = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(gbrt, X_train, y_train,
                         cv=kfold, scoring='r2')
temp_value = round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_metric = 'RMSE'
time1 = time.time()
gbrt = GradientBoostingRegressor(random_state=0)
scores = cross_val_score(gbrt, X_train, y_train,
                         cv=kfold, scoring='neg_root_mean_squared_error')
temp_value = -round(scores.mean(), 2)
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### h XGBoost sur forêt aléatoire

#%% md

"XGBoost is normally used to train gradient-boosted decision trees and other gradient boosted models.  (...) One can use XGBoost to train a standalone random forest."

Source : https://xgboost.readthedocs.io/en/latest/tutorials/rf.html

#%%

temp_model = 'XGBoost'
data_dmatrix = xgb.DMatrix(data=X, label=targets)

#%% md

**1. Split**

#%% md

Source : https://www.datacamp.com/community/tutorials/xgboost-in-python

#%%

temp_config = 'Split'

#%%

temp_metric = 'RMSE'
time1 = time.time()
xg_reg = xgb.XGBRegressor(# General parameters
                          colsample_bytree = 0.3,
                          learning_rate = 0.1,
                          max_depth = 5,
                          n_estimators = 10,
                          objective ='reg:squarederror',
                          # Regularization parameters
                          alpha = 10)
                          # Learning tasks parameters : None
xg_reg.fit(X_train,y_train)
preds = xg_reg.predict(X_test)
temp_value = round(np.sqrt(mean_squared_error(y_test, preds)), 2)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [10, 10]
plt.show()

#%%

xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()

#%% md

**4. Validation croisée mélangée**

#%%

temp_config = 'SCV'

#%%

temp_metric = 'RMSE'
time1 = time.time()
params = {'objective':'reg:squarederror',
          'colsample_bytree':0.3,
          'learning_rate':0.1,
          'max_depth':5,
          'n_estimators':10,
          'alpha':10}

cv_results = xgb.cv(dtrain=data_dmatrix,
                   params=params,
                   nfold=5,
                   num_boost_round=50,
                   early_stopping_rounds=10,
                   metrics='rmse',
                   as_pandas=True,
                   seed=123)

#%%

temp_value = cv_results['test-rmse-mean'][len(cv_results)-1]
time_delta = round(time.time() - time1, 2)
scores_train_df.loc[len(scores_train_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

### i Dummy Regressor

#%%

temp_model = 'DummyRegressor'

#%%

temp_config = 'Mean'
temp_metric = 'R2'
time1 = time.time()
dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
temp_value = dummy_regr.score(X_test, y_test)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%%

temp_config = 'Median'
temp_metric = 'R2'
time1 = time.time()
dummy_regr = DummyRegressor(strategy="median")
dummy_regr.fit(X_train, y_train)
temp_value = dummy_regr.score(X_test, y_test)
time_delta = round(time.time() - time1, 2)
scores_df.loc[len(scores_df)] = [temp_model, temp_config, temp_metric, temp_value, time_delta]

#%% md

## 3 Bilan de performance des modèles utilisés

#%%

models_list = scores_df['Model'].unique()
config_list = scores_df['Config'].unique()

#%%

scores_df.replace(['Not relevant', 'Too long'], np.nan, inplace=True)

#%%

scores_df[scores_df['Métrique']=='R2']

#%%

scores_train_df[scores_train_df['Métrique']=='RMSE']

#%% md

**Comparaisons des modèles**

#%%

best_r2_value = max(scores_df[scores_df['Métrique']=='R2']['Score'])
best_rmse_value = min(abs(scores_df[scores_df['Métrique']=='RMSE']['Score']))
best_r2_model = scores_df[scores_df['Score']==best_r2_value]['Model'].unique()[0]
best_rmse_model = scores_df[scores_df['Score']==best_rmse_value]['Model'].unique()[0]
print('Best r2 score: {}, model {}.'.format(best_r2_value, best_r2_model))
print('Best rmse score: {}, model {}.'.format(best_rmse_value, best_rmse_model))

#%% md

Problème : le modèle le plus précis n'est pas forcément le plus rapide. Avec la complexité algorithmique, précision et rapidité sont les indicateurs les plus importants, nous allons les comparer en même temps.

#%%

# Visualisation des prédictions avec la métrique RMSE
rmse_df = scores_df[scores_df['Métrique']=='RMSE']
rmse_scores = rmse_df.sort_values(by=['Score'], ascending = False)

plt.figure(figsize=(5, 5))

plt.title(temp_title)
plt.xlabel('Score RMSE')
plt.ylabel('Rapidité')
temp_x = list(rmse_scores['Score'])
temp_y = list(rmse_scores['Rapidité'])
plt.xlim(-0.08, 1.08)
plt.ylim(-0.5, 2.5)
plt.grid(True)
plt.scatter(temp_x, temp_y, color='orange', edgecolor='k', s=100)

#%% md

Le bilan des performances montre que le GradientBoosting donne le meilleur compromis précision / rapidité.

Cela dit le XGBoost est réputé donner les meilleures performances la majorité des cas. Nous allons donc optimiser GradientBoosting et XGBoost.

#%% md

## 4 Hyperparameter tuning

#%%

train_n_features = X_train.shape[1]
n_features = X.shape[1]

#%% md

### a.i Gradient Boosting : GridsearchCV

#%%

param_grid = {'n_estimators':[100, 1000],
              'learning_rate':[0.01, 0.2],
              'max_depth':[2, 5]}

#%%

time1 = time.time()
GBing = GradientBoostingRegressor(max_features=int(np.sqrt(train_n_features)))
rdm = GridSearchCV(GBing, param_grid=param_grid,
                   cv=kfold, scoring='neg_root_mean_squared_error')
rdm.fit(X_train, y_train)
rdm_best = rdm.best_estimator_
time_delta = round(time.time() - time1, 2)
train_score = round(mean_squared_error(y_train, rdm_best.predict(X_train)), 2)
test_score = round(mean_squared_error(y_test, rdm_best.predict(X_test)), 2)

#%%

print('Best parameters:', rdm.best_params_,
      '\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

### a.ii GradientBoosting : RandomizedSearchCV

#%% md

**Etape 1**

Fixer certains paramètres pour ne plus les avoir dans la RandomizedSearchCV, et ainsi réduire le temps de calcul.

#%%

distributions = dict(n_estimators=[250, 500, 750],
                     learning_rate=uniform(loc=0.02, scale=1.98),
                     max_depth=[2, 3, 4],
                     alpha=uniform(loc=0.1, scale=0.9))

#%%

time1 = time.time()
GBing = GradientBoostingRegressor(max_features=int(np.sqrt(train_n_features)),
                                  random_state=0)
rdm = RandomizedSearchCV(GBing, distributions, cv=kfold,
                         n_iter=2, # Normalement 60, passé à 2 pour ne pas ralentir le programme
                         scoring='neg_root_mean_squared_error', random_state=0)
rdm.fit(X_train, y_train)
rdm_best = rdm.best_estimator_
time_delta = round(time.time() - time1, 2)
train_score = round(mean_squared_error(y_train, rdm_best.predict(X_train)), 2)
test_score = round(mean_squared_error(y_test, rdm_best.predict(X_test)), 2)

#%%

print('Best parameters:', rdm.best_params_,
      '\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

**Etape 2** : faisons entrer d'autres paramètres en jeu.

#%%

distributions = dict(n_estimators=randint(100, 1000),
                     learning_rate=uniform(loc=0.02, scale=1.98),
                     max_depth=randint(2, 20),
                     alpha=uniform(loc=0.1, scale=0.9),
                     min_samples_leaf=randint(2, 20),
                     min_samples_split=randint(2, 20))

#%%

time1 = time.time()
GBing = GradientBoostingRegressor(max_features=int(np.sqrt(train_n_features)),
                                  #n_estimators=500,
                                  random_state=0)
rdm = RandomizedSearchCV(GBing, distributions, cv=kfold,
                         n_iter=2, # Normalement 60, passé à 2 pour ne pas ralentir le programme
                         scoring='neg_root_mean_squared_error', random_state=0)
rdm.fit(X_train, y_train)
rdm_best = rdm.best_estimator_
time_delta = round(time.time() - time1, 2)
train_score = round(mean_squared_error(y_train, rdm_best.predict(X_train)), 2)
test_score = round(mean_squared_error(y_test, rdm_best.predict(X_test)), 2)

#%%

print('Best parameters:', rdm.best_params_,
      '\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

### a.iii GradientBoosting : recherche manuelle

#%%

time1 = time.time()
GBing = GradientBoostingRegressor(min_samples_split=5,
                                  min_samples_leaf=10,
                                  max_depth=2,
                                  max_features=int(np.sqrt(train_n_features)),
                                  learning_rate=0.08,
                                  n_estimators=1250,
                                  alpha=0.34,
                                  random_state=0)
GBing.fit(X_train, y_train)
time_delta = round(time.time() - time1, 2)
train_score = round(np.sqrt(mean_squared_error(y_train, GBing.predict(X_train))), 2)
test_score = round(np.sqrt(mean_squared_error(y_test, GBing.predict(X_test))), 2)

#%%

print('\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

La recherche manuelle donne parfois de meilleurs résultats que GridSearchCV et RandomizedSearchCV. Mais elle a l'air de dépendre des ensembles train/test, qui varient à chaque fois que l'on démarre le programme depuis le début.

En effet, les ensembles X_train, y_train, X_test et y_test sont définis de manière aléatoire au cours du programme (dans le chapitre V.0), et ainsi font changer les performances d'un simple split.

#%% md

### b.i XGBoost : GridSearchCV


#%%

params = {'eta':[0.05, 0.1, 0.2],
          'colsample_bytree':[0.1, 0.2, 0.5],
          'max_depth':[2, 3, 4],
          'alpha':[1, 2, 5]}
data_dmatrix = xgb.DMatrix(data=X, label=targets)

#%%

time1 = time.time()
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
gscv = GridSearchCV(xg_reg, param_grid=params,
                   cv=kfold, scoring='neg_root_mean_squared_error')
gscv.fit(X_train, y_train)
gscv_best = gscv.best_estimator_
train_score = round(mean_squared_error(y_train, gscv_best.predict(X_train)), 2)
test_score = round(mean_squared_error(y_test, gscv_best.predict(X_test)), 2)
time_delta = round(time.time() - time1, 2)

#%%

print('Best parameters:', gscv.best_params_,
      '\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

### b.ii XGBoost : RandomizedSearchCV

#%%

distributions = dict(eta=uniform(loc=0.01, scale=1.98),
                     #gamma=list(range(1, 8)),
                     max_depth=list(range(2, 11)),
                     subsample=uniform(loc=0.01, scale=0.98),
                     #colsample_bytree=uniform(loc=0.01, scale=0.98),
                     alpha=uniform(loc=0.01, scale=0.98),
                     n_estimators=[100, 200, 350])

#%%

time1 = time.time()
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
gscv = RandomizedSearchCV(xg_reg,
                         distributions,
                         cv=kfold,
                         n_iter=2, # Normalement 60, passé à 2 pour ne pas ralentir le programme
                         scoring='neg_root_mean_squared_error')
gscv.fit(X_train, y_train)
gscv_best = gscv.best_estimator_
train_score = round(mean_squared_error(y_train, gscv_best.predict(X_train)), 2)
test_score = round(mean_squared_error(y_test, gscv_best.predict(X_test)), 2)
time_delta = round(time.time() - time1, 2)

#%%

print('Best parameters:', gscv.best_params_,
      '\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

### b.iii XGBoost : recherche manuelle

#%%

params= {'eta':0.52,
         'max_depth':2,
         #'subsample':0.5,
         'alpha':0.75,
         'n_estimators':200}

#%%

time1 = time.time()

xg_reg = xgb.XGBRegressor(params=params,
                          objective ='reg:squarederror')
xg_reg.fit(X_train,y_train)
train_score = round(mean_squared_error(y_train, xg_reg.predict(X_train)), 2)
test_score = round(mean_squared_error(y_test, xg_reg.predict(X_test)), 2)
time_delta = round(time.time() - time1, 2)

#%%

print('\nTrain score: {} / test score: {}'.format(train_score, test_score),
      '\nTime: {} s'.format(time_delta))

#%% md

### b.iv Cross validation

#%%

time1 = time.time()
#xg_reg = xgb.XGBRegressor(objective ='reg:squarederror')
cv_results = xgb.cv(dtrain=data_dmatrix,
                       params=params,
                       nfold=5,
                       num_boost_round=200,
                       early_stopping_rounds=10,
                       metrics='rmse',
                       as_pandas=True,
                       seed=123)
train_score = np.sqrt(cv_results['train-rmse-mean'][len(cv_results)-1])
train_score = round(train_score, 2)
test_score = np.sqrt(cv_results['test-rmse-mean'][len(cv_results)-1])
test_score = round(test_score, 2)
time_delta = round(time.time() - time1, 2)

#%% md

# VI. Conclusion

#%% md

## 1 Gradient Boosting

#%% md

**Représentation graphique**

#%%

sns.set_theme(color_codes=True)
x = y_test
y = GBing.predict(X_test)
sns.jointplot(x, y, kind="reg")

#%% md

## 2 XGBoost

#%% md

**Représentation graphique**

#%%

sns.set_theme(color_codes=True)
x = y_test
y = xg_reg.predict(X_test)
sns.jointplot(x, y, kind="reg")