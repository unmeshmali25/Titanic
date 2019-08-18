import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

titanic_df = pd.read_csv('train.csv')
titanic_df.head()

# This code has been created and executed on jupyter notebook
%matplotlib inline
sns.catplot('Sex', kind = 'count', data = titanic_df)


sns.catplot('Sex', kind = 'count', data = titanic_df, hue = 'Pclass')


def male_female_child(passenger):
    age, sex = passenger
    if age < 16:
        return 'child'
    else:
        return sex

 titanic_df['person'] = titanic_df[['Age', 'Sex']].apply(male_female_child, axis = 1)

 sns.catplot('person', data = titanic_df, kind = 'count', hue = 'Pclass')

 sns.catplot('Pclass', data = titanic_df, kind = 'count', hue = 'person')

 titanic_df['Age'].hist(bins = 70)

 titanic_df['person'].value_counts()

# Kernel Density Estimation plot of age ~ Sex
fig = sns.FacetGrid(titanic_df, hue = 'Sex', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.set(xlim = (0, titanic_df['Age'].max()))
fig.add_legend()

# Kernel Density Estimation plot of age ~ person
fig = sns.FacetGrid(titanic_df, hue = 'person', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.set(xlim = (0, titanic_df['Age'].max()))
fig.add_legend()

# Dropping all the NA values in the Cabin column
deck = titanic_df['Cabin'].dropna()


# Selecting only the first character (letter) from the Cabin number column
levels = []

for level in deck:
    levels.append(level[0])


cabin_df = DataFrame(levels)
cabin_df.columns = ['Cabin']

sns.catplot('Cabin', data = cabin_df, kind = 'count', palette = 'winter_d')

# Removing the T deck since it is very low (an outlier)
cabin_df = cabin_df[cabin_df.Cabin != 'T']

sns.catplot('Cabin', data = cabin_df, kind = 'count', palette = 'summer_d')


# Checking number of passengers across the cities
sns.catplot('Embarked', data = titanic_df, kind = 'count')

sns.catplot('Embarked', data = titanic_df, kind = 'count', hue = 'Pclass')

sns.catplot('Pclass', data = titanic_df, kind = 'count', hue = 'Embarked')


# Checking who was alone and who came with family
titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch

titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'
titanic_df['Alone'].loc[titanic_df['Alone'] > 0] = 'With Family'


