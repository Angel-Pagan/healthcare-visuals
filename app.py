""" Import packages """
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


sns.set_theme(style="whitegrid") # Assigning sns theme for future use 


""" Loading dataset """

georgia_cvd_19_case_data = pd.read_csv('https://raw.githubusercontent.com/hantswilliams/AHI_Microcourse_Visualization/main/Data/Georgia_COVID/Georgia_COVID-19_Case_Data.csv') # Importing data from Github 

georgia_cvd_19_case_data.shape # Size of file

list(georgia_cvd_19_case_data) # List of Columns

georgia_cvd_19_case_data.dtypes # Column Datatypes

georgia_cvd_19_case_data.head(10) # Preview 10 Rows of the dataset

""" Cleaning Data Set before assigning variables """

#Cleaning Columns 

georgia_cvd_19_case_data.columns = georgia_cvd_19_case_data.columns.str.lower() # change all column names to lowercase

georgia_cvd_19_case_data.columns = georgia_cvd_19_case_data.columns.str.replace(' ', '_') # replace all whitespace in column names with an underscore

georgia_cvd_19_case_data.columns = georgia_cvd_19_case_data.columns.str.replace('[^A-Za-z0-9]+', '_') # remove all special characters and whitespace ' ' from a specific column 

list(georgia_cvd_19_case_data.columns) # Print list of columns

#Cleaning Object County 

georgia_cvd_19_case_data['county'] = georgia_cvd_19_case_data['county'].str.lower()

georgia_cvd_19_case_data['county'] = georgia_cvd_19_case_data['county'].str.replace(' ', '_')
                                                          
georgia_cvd_19_case_data['county'] = georgia_cvd_19_case_data['county'].str.replace('[^A-Za-z0-9]+', '_')

georgia_cvd_19_case_data['county'].sample(10) # Print 10 random counties rows 


df = georgia_cvd_19_case_data # Changing variable from georgia_cvd_19_case_data to df (df) for coding easiblity 

""" creating and assigning variables"""

df['county'].value_counts()

df_counties = df['county'].value_counts().sort_index() #created a new variable counties

df_counties.head(20)

# Created a variable called datestamp_mod
df['datestamp_mod'] = df['datestamp'] #duplicating datestamp column to datestamp_mod column 
df['datestamp_mod'].head(5)


# Changing column ['datestamp_mod'] datatype from object to datetime64
df['datestamp_mod'] = pd.to_datetime(df['datestamp_mod'])
df['datestamp_mod'].head(5)
df['datestamp_mod'].dtypes

# Creating day, month, year, quarter columns from datestamp_mod column

df['datestamp_mod_day'] = df['datestamp_mod'].dt.date
df['datestamp_mod_day']

df['datestamp_mod_month_year'] = df['datestamp_mod'].dt.to_period('M')
df[['datestamp_mod_month_year']]


df['datestamp_mod_month'] = df['datestamp_mod'].dt.month_name()
df[['datestamp_mod_month','datestamp_mod_day', 'datestamp_mod', 'datestamp']]

df['datestamp_mod_week'] = df['datestamp_mod'].dt.week
df[['datestamp_mod_month','datestamp_mod_day', 'datestamp_mod', 'datestamp', 'datestamp_mod_week']]

df['datestamp_mod_quarter'] = df['datestamp_mod'].dt.to_period('Q')
df[['datestamp_mod_month','datestamp_mod_day', 'datestamp_mod', 'datestamp', 'datestamp_mod_week', 'datestamp_mod_quarter']]

df['datestamp_mod_day_string'] = df['datestamp_mod_day'].astype(str)
df['datestamp_mod_week_string'] = df['datestamp_mod_week'].astype(str)
df['datestamp_string'] = df['datestamp_mod_month_year'].astype(str)

df

""" Selecting county and timeframe for analysis """

# Selecting counties to analyize 

countlist = ['cobb', 'delk', 'fulton', 'gwinnett', 'hall']
countlist

selectCounties = df[df['county'].isin(countlist)]
len(selectCounties)

# Selecting timeframe

selectCountyTime = selectCounties 

selectCountyTime

selectCountTime_AprilMay2020 = selectCountyTime[(selectCountyTime['datestamp_mod_month_year'] == '2020-04') | (selectCountyTime['datestamp_mod_month_year'] == '2020-05')]
len(selectCountTime_AprilMay2020)

selectCountTime_AprilMay2020.sample(10)

#creating a dataframe containing the information needed for analysys 

finaldf = selectCountTime_AprilMay2020[[
    'county',
    'datestamp_mod',
    'datestamp_mod_day',
    'datestamp_mod_day_string',
    'datestamp_string',
    'datestamp_mod_month_year',
    'c_new', # New Cases
    'c_cum', # Total Cases
    'h_new', # New Hospitalizations
    'h_cum', # Total Hospitalizations
    'd_new', # New Deaths
    'd_cum'  # Total Deaths
    ]]



 
""" Looking at Total Covid-19 Cases by Month  """

finaldf_dropdups = finaldf.drop_duplicates(subset=['county', 'datestamp_string'], keep='last')
finaldf_dropdups

pd.pivot_table(finaldf_dropdups, values='c_cum', index=['county'],
                    columns=['datestamp_mod_month_year'], aggfunc=np.sum)

vis1 = sns.barplot(x='datestamp_mod_month_year', 
                    y = 'c_cum', 
                    data=finaldf_dropdups)

vis2 = sns.barplot(x='datestamp_mod_month_year', 
                   y = 'c_cum', hue='county', 
                   data=finaldf_dropdups)

plotly1 = px.bar(finaldf_dropdups, 
                 x='datestamp_string', 
                 y = 'c_cum', color='county', 
                 barmode='group')

plotly1.show()

plotly2 = px.bar(finaldf_dropdups, x='datestamp_string', 
                 y = 'c_cum', 
                 color='county', 
                 barmode='stack')
plotly2.show()

""" Looking at COVID-19 Daily Cases  """

daily = finaldf
len(daily)

pd.pivot_table( daily, 
                values='c_cum', 
                index=['county'],
                columns=['datestamp_mod_day'], 
                aggfunc=np.sum)

pd.pivot_table(daily,
               values='c_cum',
               index=['datestamp_mod_day'],
               columns=['county'],
               aggfunc=np.sum)

startDate = pd.to_datetime('2020-04-26').date()
endDate = pd.to_datetime('2020-05-09').date()

maskFilter = (daily['datestamp_mod_day'] >= startDate) & (daily['datestamp_mod_day'] <= endDate)
dailySpecific = daily.loc[maskFilter]
dailySpecific

dailySpecific[dailySpecific['county'] == 'fulton']

vis3 = sns.lineplot(data=dailySpecific,
                    x='datestamp_mod_day',
                    y='c_cum')

vis4 = sns.lineplot(data=dailySpecific,
                    x='datestamp_mod_day',
                    y='c_cum',
                    hue='county')

plotly3 = px.bar(dailySpecific,
                 x='datestamp_mod_day',
                 y = 'c_cum',
                 color='county')
plotly3.show()

plotly4 = px.bar(dailySpecific, 
                 x='datestamp_mod_day', 
                 y = 'h_new', 
                 color='county',
                 barmode= 'group')
plotly4.show()

plotly5 = px.bar(dailySpecific, 
                 x='datestamp_mod_day', 
                 y = 'h_cum', 
                 color='county',
                 barmode= 'group')
plotly5.show()

plotly6 = px.bar(dailySpecific, 
                 x='datestamp_mod_day', 
                 y = 'd_new', 
                 color='county',
                 barmode= 'group')
plotly6.show()

plotly7 = px.bar(dailySpecific, 
                 x='datestamp_mod_day', 
                 y = 'd_cum', 
                 color='county',
                 barmode= 'group')
plotly7.show()

dailySpecific['newHospAndDeathsAndCovid'] = dailySpecific['h_new'].astype(int) + dailySpecific['d_new'].astype(int) + dailySpecific['c_new'].astype(int)
dailySpecific['newHospAndDeathsAndCovid']

dailySpecific

plotly8 = px.bar(dailySpecific, 
                 x='datestamp_mod_day', 
                 y = 'newHospAndDeathsAndCovid', 
                 color='county',
                 barmode= 'group',
                 title='Georgia 2020 Covid Data: New Hospitalizations, Deaths, and COVID-19 Cases',
                 labels={'datestamp_mod_day':'Time (Month, Day, Year)', 
                         'newHospAndDeathsAndCovid':'New Hospitalizations, Deaths, and COVID-19 Cases', 
                         'county':'county'},
                 )
plotly8.update_layout(
    xaxis=dict(
        tickmode = 'linear',
        type = 'category',
    )
)
plotly8.show()




