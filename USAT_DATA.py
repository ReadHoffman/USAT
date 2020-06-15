# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 19:55:54 2020

@author: Read
"""

import gc
from datetime import date
from datetime import datetime, timedelta 
import pandas as pd
import numpy as np
from scipy import stats
import requests
from bs4 import BeautifulSoup

gc.collect()

# import HTMLSession from requests_html
from requests_html import HTMLSession, AsyncHTMLSession
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys

#modeling
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from rfpimp import permutation_importances
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

#other
import statistics
from yellowbrick.regressor import ResidualsPlot


"""
data notes
-some records have 00:00:00
-dnf records have out of pattern low times and times that make no sense, messy
-some records have out of pattern high times eg with 01:20:05 bike time
-some records have  35:09:00 format

post-ETL notes
-some Finish Times are FUBAR.  as a general rule, I wouldn't recommend trusting finsh times
-if you really need a finish time, make an aggregate one one valid race leg times


"""

# build formulas
def tot_r2(rfm,X_train, y_train):
    return r2_score(y_train, rfm.predict(X_train))


#pull saved data
RACES_FILENAME = 'C:/Users/Read/Desktop/Code/Python/USAT/RACE_DATA.csv'
df_races2 = pd.read_csv(RACES_FILENAME)
df_races2 = df_races2.drop(['Unnamed: 0'], axis = 1) 

#create column lists for subsequent melting (aka pivoting from wide to long df)
time_cols = ['Swim','T1','Bike','T2','Run','Finish']
all_cols = df_races2.columns.tolist()

# define race legs and aggregate legs
dict_legs = { 
    'Swim' : { 'Legs':['Swim'] , 'PCT_FR_MED_HI': 2 , 'PCT_FR_MED_LO': -.3,'Time_col':True } 
    ,'T1': { 'Legs':['T1'] , 'PCT_FR_MED_HI': 4 , 'PCT_FR_MED_LO': -.3, 'Time_col':True} 
    ,'Bike': { 'Legs':['Bike'] , 'PCT_FR_MED_HI': 2, 'PCT_FR_MED_LO': -.3,'Time_col':True } 
    ,'T2': { 'Legs':['T2'] , 'PCT_FR_MED_HI': 4, 'PCT_FR_MED_LO': -.3, 'Time_col':True } 
    ,'Run': { 'Legs':['Run'] , 'PCT_FR_MED_HI': 2, 'PCT_FR_MED_LO': -.3, 'Time_col':True } 
    ,'Finish': { 'Legs':['Swim','T1','Bike','T2','Run']  , 'PCT_FR_MED_HI': 2,'PCT_FR_MED_LO': -.3,'Time_col':True } 
    ,'Finish_Calc': { 'Legs':['Swim','T1','Bike','T2','Run']  , 'PCT_FR_MED_HI': 2,'PCT_FR_MED_LO': -.3,'Time_col':False } 
    ,'Bike_Out': { 'Legs':['Swim','T1'] , 'PCT_FR_MED_HI': 2,'PCT_FR_MED_LO': -.3, 'Time_col':False } 
    , 'Bike_In': { 'Legs':['Swim','T1','Bike'] , 'PCT_FR_MED_HI': 2 ,'PCT_FR_MED_LO': -.3,'Time_col':False} 
    , 'Run_Out': { 'Legs':['Swim','T1','Bike','T2']  , 'PCT_FR_MED_HI': 2 ,'PCT_FR_MED_LO': -.3,'Time_col':False} 
    }

#melt data frame to have all values in one column, making calcs easier
# this makes the grain Date, Race, Person, Sport level
# Overall place field is being kept as an attribute across all Race Legs (not just finish) intentionally for later analysis
df = df_races2.melt(id_vars=[x for x in all_cols if x not in time_cols],value_vars=time_cols,value_name='Time')
df = df.rename(columns={'variable':'Race_Leg'})


#fix erroneous values
df.loc[:,'Time'] = df.loc[:,'Time'].replace('00:00',np.NaN)

# fix DNFs (at least partially)
dnf_list = ( (df.Place=='0') | (df.Place=='DNF') )
df.loc[dnf_list,'Place'] = np.NaN
df['Place'] = pd.to_numeric(df['Place'])

#clean up times, convert to seconds
bike_clean_rows = (df.Race_Leg=='Bike') & (df.Time.str.count(':')==2) & (df.Time.str.slice(stop=2) != '01')
df.loc[ bike_clean_rows==True,'Time'] = df.loc[ bike_clean_rows==True,'Time'].str.slice(stop=5)
T2_clean_rows = (df.Race_Leg=='T2') & (df.Time.str.count(':')==2) & (df.Time.str.slice(stop=2) != '01')
df.loc[ T2_clean_rows==True,'Time'] = df.loc[ T2_clean_rows==True,'Time'].str.slice(stop=5)

#add '00:' onto length 5
df.loc[df.Time.str.len()==5,'Time'] = '00:'+df.loc[df.Time.str.len()==5,'Time']

def get_sec(time_str):
    if pd.isnull(time_str) == True:
        result = time_str
    else:
        """Get Seconds from time."""
        h, m, s = time_str.split(':')
        result =  int(h) * 3600 + int(m) * 60 + int(s)
    return result

df.loc[:,'Time'] = df.Time.apply(get_sec)

#create flag to indicate the "clean" race legs so that filtering discrete race legs easier after we union aggregate race legs
df.loc[:,'Race_Leg_Flag'] = 1

# list for grouping down to the person level
list_gb_person = ['Name', 'Country', 'Racing Age', 'Place', 'Team', 'Race_Name', 'Division', 'Date', 'Location'] #everything but Time

# list for grouping down to the race leg level
list_gb_race_leg = ['Race_Name','Division','Date','Race_Leg']


df['Mean'] = df.groupby(list_gb_race_leg)['Time'].transform('mean')
df['Median'] = df.groupby(list_gb_race_leg)['Time'].transform('median')
df['Quantile_20'] = df.groupby(list_gb_race_leg)['Time'].transform('quantile',.2)
df['Quantile_80'] = df.groupby(list_gb_race_leg)['Time'].transform('quantile',.8)
df['Min'] = df.groupby(list_gb_race_leg)['Time'].transform('min')
df['Max'] = df.groupby(list_gb_race_leg)['Time'].transform('max')
df['Std'] = df.groupby(list_gb_race_leg)['Time'].transform('std')
df['Var_From_Normal'] = abs(df.Time - df.Median)
df['Var_From_Median_Pct'] = df.Time/df.Median-1
df['Var_From_20_PCTLE_Pct'] = df.Time/df.Quantile_20-1
df['Var_From_80_PCTLE_Pct'] = df.Time/df.Quantile_80-1
df['PCT_FR_MED_HI'] = df.Race_Leg.apply(lambda x: dict_legs.get(x)['PCT_FR_MED_HI'])
df['PCT_FR_MED_LO'] = df.Race_Leg.apply(lambda x: dict_legs.get(x)['PCT_FR_MED_LO'])
df['Valid'] = (df.Var_From_80_PCTLE_Pct<df['PCT_FR_MED_HI']) & (df.Var_From_20_PCTLE_Pct>df['PCT_FR_MED_LO'] ) & (df.Time<12000)



df_aggs = df[0:0] #empty data frame shell for concat union
for key in dict_legs.keys():
    #check if the key is novel to the df eg if 'Swim' then skip
    if key in df.Race_Leg.unique():
        continue
    list_legs = dict_legs.get(key).get('Legs')  
    #how many legs we are summing
    list_legs_len = len(list_legs)
    
    #sum both the legs and the valid indicator
    df_loop = df.loc[df.Race_Leg.isin(list_legs),:].groupby(['Name', 'Race_Name', 'Division', 'Date']).agg( { 'Time': 'sum', 'Valid' : 'sum'} ).reset_index()
    df_loop['Race_Leg'] =key
    df_loop['Race_Legs_Summed'] = df_loop.Valid
    df_loop['Valid_needed'] = list_legs_len
    #valid indicator sum must match number of legs expected to be summed or the data is invalid
    df_loop['Valid'] = (df_loop.Valid_needed==df_loop['Race_Legs_Summed'])
    #add back in columns with NAs that had to be removed from groupby statement because pandas doesn't like NaNs in group by
    df_loop=df_loop.merge(df[['Name', 'Race_Name', 'Division', 'Date']+['Country', 'Racing Age', 'Place', 'Team','Location']].drop_duplicates(), on =['Name', 'Race_Name', 'Division', 'Date'],how='left')
    #join loop df to holding df
    df_aggs = pd.concat([df_aggs,df_loop], ignore_index=True, axis=0,sort=False)

#clean up columns needed for loop
df_aggs.drop(columns=['Race_Legs_Summed','Valid_needed'],inplace=True)

#union aggregate data fields to non-aggregate fields
df = pd.concat([df,df_aggs], ignore_index=True, axis=0,sort=False).sort_values(by=['Race_Name','Division','Date','Name','Race_Leg'])


#calc top 3 in each event and append on at the individual level
df['Time_Rank'] = df[df.Valid==True].groupby(list_gb_race_leg, as_index=False)['Time'].rank(method='min')
gb_rnk = df[df.Time_Rank<=3].groupby(list_gb_race_leg, as_index=False)['Time'].mean()
df = df.merge(gb_rnk,how='left',left_on = list_gb_race_leg, right_on=list_gb_race_leg,suffixes=('', '_Top3'))
df.loc[(df.Valid==True),'Time_Top3Pct'] = (df.loc[(df.Valid==True),'Time'] /df.loc[(df.Valid==True),'Time_Top3'] )-1


#calc top 3 podium overall times for each race leg
place_list=[1,2,3]
for place in place_list:
    pl_rnk = df[df.Place==place].groupby(list_gb_race_leg, as_index=False)['Time'].mean()
    suffix2 = '_PL'+str(place)
    df = df.merge(pl_rnk,how='left',on=list_gb_race_leg,suffixes=('',suffix2 ))
    df['Time'+suffix2+'Diff']=df.Time-df['Time'+suffix2]
    df['Time'+suffix2+'Pct']=df.Time/df['Time'+suffix2]-1

#create gender group
df['Gender'] = df.Division.apply(lambda x: 'Female' if ((x == 'Junior Elite Female') | (x == 'Youth Elite Female' )) else 'Male')

#best place at age 18+
df = df.merge(df[df['Racing Age']>=18].groupby(['Name'], as_index=False)['Place'].max(),how='left', left_on='Name',right_on='Name',suffixes=('', '_MAX_18+') )

#--nth race in data
df['Date'] = df.Date.astype('datetime64[ns]')
gb_nrace = df.loc[( (df.Valid==True) & (df.Race_Leg=='Swim') ),:].groupby(['Name'])['Date'].rank(method='min').to_frame(name='N_Race_In_Data')
gb_nrace = gb_nrace.merge(df[['Name','Date','Race_Name','Division']],left_index=True,right_index=True,how='left')
df = df.merge(gb_nrace,on = ['Name','Date','Race_Name','Division'],how='left' )


#--min race in data
#df_races2 = df_races2.drop(['MIN_RACE_AGE_IN_DATA','MIN_DATE_IN_DATA'],axis=1)
grp_min_date_age = df.groupby('Name')[['Date','Racing Age']].min()
grp_min_date_age = grp_min_date_age.rename(columns={"Date": "MIN_DATE_IN_DATA", "Racing Age": "MIN_RACING_AGE_IN_DATA"})
df = df.merge(grp_min_date_age,how='left',left_on = 'Name', right_index=True)

df = df.merge(df[df['Racing Age']==16].groupby(['Name'], as_index=False)['Place'].max(),how='left', left_on='Name',right_on='Name',suffixes=('', '_MAX_16') )


list_USA_races = [x for x in df.Location.unique() if x.find('USA')>=0]
df['USA_Race'] = df.Location.isin(list_USA_races)


df = df.merge(df[df.USA_Race==False].groupby('Name')['Date'].min(),how='left',on = 'Name',suffixes=['','_First_NonUSA_Race'])

df['International_Racer_Flag'] = (df.Date_First_NonUSA_Race<=df.Date)


## more fields for modeling ## 2020-06-07

df.loc[(df.Valid==True),'Time_PctPri1'] = df.loc[(df.Valid==True),:].sort_values(by=['Date'], ascending=True).groupby(['Name','Race_Leg'])['Time_Top3Pct'].shift(1)
df.loc[(df.Valid==True),'Time_PctPri2'] = df.loc[(df.Valid==True),:].sort_values(by=['Date'], ascending=True).groupby(['Name','Race_Leg'])['Time_Top3Pct'].shift(2)
df.loc[(df.Valid==True),'Time_PctPri3'] = df.loc[(df.Valid==True),:].sort_values(by=['Date'], ascending=True).groupby(['Name','Race_Leg'])['Time_Top3Pct'].shift(3)
df.loc[(df.Valid==True),'Time_PctPri4'] = df.loc[(df.Valid==True),:].sort_values(by=['Date'], ascending=True).groupby(['Name','Race_Leg'])['Time_Top3Pct'].shift(4)
df.loc[(df.Valid==True),'Time_PctPri5'] = df.loc[(df.Valid==True),:].sort_values(by=['Date'], ascending=True).groupby(['Name','Race_Leg'])['Time_Top3Pct'].shift(5)
df['Time_PctPriAvg2'] = (df['Time_PctPri1'].fillna(0) + df['Time_PctPri2'].fillna(0) ) /  (df['Time_PctPri1'].notna()*1+df['Time_PctPri2'].notna()*1) 
df['Time_PctPriAvg3'] = (df['Time_PctPri1'].fillna(0) + df['Time_PctPri2'].fillna(0) + df['Time_PctPri3'].fillna(0)) /  (df['Time_PctPri1'].notna()*1+df['Time_PctPri2'].notna()*1+df['Time_PctPri3'].notna()*1) 
df['Time_PctPriAvg5'] = (df['Time_PctPri1'].fillna(0) + df['Time_PctPri2'].fillna(0) + df['Time_PctPri3'].fillna(0) + df['Time_PctPri4'].fillna(0) + df['Time_PctPri5'].fillna(0)) /  (df['Time_PctPri1'].notna()*1+df['Time_PctPri2'].notna()*1+df['Time_PctPri3'].notna()*1+df['Time_PctPri4'].notna()*1+df['Time_PctPri5'].notna()*1) 
df['Time_PctPriAMax5'] = df.loc[:,['Time_PctPri1','Time_PctPri2','Time_PctPri3','Time_PctPri4','Time_PctPri5']].max(axis=1, skipna=False)
df['Time_PctPriAMax3'] = df.loc[:,['Time_PctPri1','Time_PctPri2','Time_PctPri3']].max(axis=1, skipna=False)
df['Time_PctPriAMin3'] = df.loc[:,['Time_PctPri1','Time_PctPri2','Time_PctPri3']].min(axis=1, skipna=False)

#pack
#for each race for each leg
#sort by time
#shift 1 to prior record , select time
#calcualte gap to person ahead
df.loc[(df.Valid==True),'Time_Gap'] = df.loc[(df.Valid==True),:].Time - df.loc[(df.Valid==True),:].sort_values(by=['Time'],ascending=True).groupby(list_gb_race_leg)['Time'].shift(1)


#for each race for each leg
#sort by time
#if gap exceeds X seconds then 1 else 0
#cumulative sum along x axis
#NOT a priority



# team is not null
df['HAS_TEAM'] = df.Team.notnull()


# best 3 last-3-race-avgs, avg together as a means for estimating how "good" top 3 will be this race
df.loc[(df.Valid==True),'Time_PctPriAvg3_Rank'] = df.loc[(df.Valid==True),:].groupby(list_gb_race_leg)['Time_PctPriAvg3'].rank(method='min')

gb_top3priavg = df.loc[(df.Time_PctPriAvg3_Rank<=3),:].groupby(list_gb_race_leg)['Time_PctPriAvg3'].mean().reset_index()
df = df.merge(gb_top3priavg,how='left',on = list_gb_race_leg,suffixes=['','_Top3'])


# best 3 max 5 of 5 recent races, avg together as a means for estimating how "good" top 3 will be this race
df.loc[(df.Valid==True),'Time_PctPriAMax5_Rank'] = df.loc[(df.Valid==True),:].groupby(list_gb_race_leg)['Time_PctPriAMax5'].rank(method='min')

gb_top3priavg = df.loc[(df.Time_PctPriAMax5_Rank<=3),:].groupby(list_gb_race_leg)['Time_PctPriAMax5'].mean().reset_index()
df = df.merge(gb_top3priavg,how='left',on = list_gb_race_leg,suffixes=['','_Top3'])

#write table
RACES_FILENAME_final2 = 'C:/Users/Read/Desktop/Code/Python/USAT/RACE_DATA_final2.csv'
df.to_csv(RACES_FILENAME_final2, encoding='utf-8', index=True)
#pull saved data
#df_races2 = pd.read_csv(RACES_FILENAME_final)
#df_races2 = df_races2.drop(['Unnamed: 0'], axis = 1) 










###################
#### Modeling #####
###################

# pull cached file
dfm = pd.read_csv(RACES_FILENAME_final2)
dfm = dfm.drop(['Unnamed: 0'], axis = 1) 

#modify/fix columns, consider moving to prev section
dfm['Date'] = pd.to_datetime(dfm.Date, format='%Y-%m-%d') 

#column metadata
#dfm_meta = pd.DataFrame(dfm.columns,columns=['Field'])  # df meta not needed right now

list_features = [ 
                 'Country',
                  'Racing Age'
                 , 'Division'
                   , 'Race_Leg'
                   , 'N_Race_In_Data'
                   , 'USA_Race'
                   , 'International_Racer_Flag'
                   , 'Time_PctPriAvg2'
                   , 'Time_PctPriAvg3'
                   , 'Time_PctPriAvg5'
                   , 'Time_PctPri1'
                   , 'Time_PctPriAMax3'
                   , 'Time_PctPriAMax5'
                   , 'Time_PctPriAMin3'
                   , 'HAS_TEAM'
                   , 'Time_PctPriAvg3_Top3'
                   , 'Time_PctPriAMax5_Top3'
                   ]
target = 'Time_Top3Pct'

#dfm_meta['Feature'] = dfm_meta.Field.isin(list_features)
#dfm_meta['Target'] = dfm_meta.Field==target

for time_col in time_cols:
        dfm2 = dfm.loc[(dfm.Valid==True) & (dfm.Race_Leg==time_col),:].copy()
        dfm2 = dfm2.loc[:,list_features+[target]]
#        len(dfm2)
                
        #actually lets try replacing with -1 instead with the hopes that the model will understand this
        dfm2=dfm2.fillna(-1)      
        
        #figure out which columns we need to encode
        obj_cols = dfm2.select_dtypes(include=['object','bool']).columns.values
        # creating instance of labelencoder
        dfm2[obj_cols]=dfm2[obj_cols].apply(lambda x: LabelEncoder().fit_transform(x.astype(str)) )
        
        
        X = dfm2.loc[:,list_features]
        Y = dfm2.loc[:,[target]]
        
        
        X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = .2)
        
        
        ##############################################################
        
        #tried many models, RF was best
        #The RF Train is 0.6203183331774287. Test is 0.5494430680915874. 
        #The GBR Train is 0.5136991863658964. Test is 0.5093991369850603. 
        #The LINEAR REGRESSION Train is 0.33104664775927123. Test is 0.3312855631191846. 
        #The RIDGE REGRESSION Train is 0.33104664775927123. Test is 0.3312855631191846. 
        #The SVM Train is 0.19502078187297178. Test is 0.19523202707195542. 
        #The Lasso Train is 0.3310463993531328. Test is 0.331284370399007. 
        
        
        #### Random Forest Regressor
        
        parameters = {'bootstrap': True
                          ,'min_samples_leaf': 6
                          ,'n_estimators': 50
                          ,'min_samples_split': 15
                          ,'max_features': 'auto'
                          ,'max_depth': 50
                          ,'oob_score': True
                          ,'n_jobs': -1
                          ,'criterion': 'mse'
        #                  ,'min_impurity_decrease': .001 #not helping
                          , 'random_state': 1
                          } 
        rf = RandomForestRegressor(**parameters)
        rfm = rf.fit(X_train, y_train.values.ravel())
        #    model_predict = rf.predict(X_test).astype(int)
        #    actual = y_test.astype(int)
        
        #### rf model evaluation
        rfm_train_score = rfm.score(X_train,y_train)
        rfm_test_score = rfm.score(X_test,y_test)
        
        print("The RF Train is {}. Test is {}. ".format(rfm_train_score,rfm_test_score) )
        
        #moved to beginning of script
        #def tot_r2(rfm,X_train, y_train):
        #    return r2_score(y_train, rfm.predict(X_train))
        #perm_imp_rfpimp = permutation_importances(rfm, X_train, y_train, tot_r2)
        
        #plt.plot(rf.predict(X_train),y_train,'o' )
        
        
        def prep_model_data(df_var,feature_var,target_var):
            df_var = df_var.loc[:,feature_var+[target_var]].fillna(-1)    
            
            obj_cols2 = df_var.select_dtypes(include=['object','bool']).columns.values
            if len(obj_cols2) != 0:
                df_var[obj_cols2]=df_var[obj_cols2].apply(lambda x: LabelEncoder().fit_transform(x.astype(str)) )    
            else:
                pass
            
            return df_var.loc[:,list_features]    
        
        dfm3 = prep_model_data(df_var=dfm.loc[(dfm.Valid==True) & (dfm.Race_Leg==time_col),:], feature_var=list_features, target_var=target)
        
        dfm.loc[(dfm.Valid) & (dfm.Race_Leg==time_col),'Time_Top3Pct_Pred'] = rfm.predict(dfm3)

#write table
RACES_FILENAME_final2 = 'C:/Users/Read/Desktop/Code/Python/USAT/RACE_DATA_final2.csv'
dfm.to_csv(RACES_FILENAME_final2, encoding='utf-8', index=True)
 

##############################################################


##### Gradient boost regressor
#
##Create the model
#gbr = GradientBoostingRegressor()
#
## Fit the model on the training data
#gbr.fit(X_train, y_train.values.ravel())
#
## Make predictions on the test data
#gbr_predictions = gbr.predict(X_test)
#
## Evaluate the model
#gbr_train_score = gbr.score(X_train,y_train)
#gbr_test_score = gbr.score(X_test,y_test)
#
#print("The GBR Train is {}. Test is {}. ".format(gbr_train_score,gbr_test_score) )
#
#
#
###############################################################
#
#### Linear Regression
#
##Create the model
#regr = linear_model.LinearRegression()
#
## Fit the model on the training data
#regr.fit(X_train, y_train.values.ravel())
#
## Make predictions on the test data
#regr_predictions = regr.predict(X_test)
#
## Evaluate the model
#regr_train_score = regr.score(X_train,y_train)
#regr_test_score = regr.score(X_test,y_test)
#
#print("The LINEAR REGRESSION Train is {}. Test is {}. ".format(regr_train_score,regr_test_score) )
#
#
###############################################################
#
#### Ridge Regression
#
##Create the model
#ridg = linear_model.LinearRegression()
#
## Fit the model on the training data
#ridg.fit(X_train, y_train.values.ravel())
#
## Make predictions on the test data
#ridg_predictions = ridg.predict(X_test)
#
## Evaluate the model
#ridg_train_score = ridg.score(X_train,y_train)
#ridg_test_score = ridg.score(X_test,y_test)
#
#print("The RIDGE REGRESSION Train is {}. Test is {}. ".format(ridg_train_score,ridg_test_score) )
#
#
###############################################################
#
#### SVM Regression
###huge and The SVM REGRESSION Train is 0.19502078187297178. Test is 0.19523202707195542. 
#
#### Lasso Regression
#
##Create the model
#lasso = Lasso()
#
## Fit the model on the training data
#lasso.fit(X_train, y_train.values.ravel())
#
## Make predictions on the test data
#lasso_predictions = lasso.predict(X_test)
#
## Evaluate the model
#lasso_train_score = lasso.score(X_train,y_train)
#lasso_test_score = lasso.score(X_test,y_test)
#
#print("The SVM REGRESSION Train is {}. Test is {}. ".format(lasso_train_score,lasso_test_score) )

# this is not working, takes forever
#logreg = LogisticRegression()
#logreg.fit(X_train,y_train.values.ravel())
#logreg_predictions = logreg.predict(X_train)
#print('Train LOGREG accuracy score:',accuracy_score(y_train,logreg_predictions))
#print('Test LOGREG accuracy score:', accuracy_score(y_test,logreg.predict(X_test)))
## Evaluate the model
#logreg_train_score = logreg.score(X_train,y_train)
#logreg_test_score = logreg.score(X_test,y_test)
#
#print("The GBR Train is {}. Test is {}. ".format(logreg_train_score,logreg_test_score) )


#from xgboost import XGBClassifier
#from sklearn.model_selection import train_test_split
#
#
## fit model no training data
#xgb = XGBClassifier()
#xgb.fit(X_train, y_train.values.ravel())
#
## make predictions for test data
#xgb_predictions = xgb.predict(X_test)
#
## Evaluate the model
#xgb_train_score = xgb.score(X_train,y_train)
#xgb_test_score = xgb.score(X_test,y_test)
#
#print("The GBR Train is {}. Test is {}. ".format(xgb_train_score,xgb_test_score) )

#
#
#tot_y = Y
#tot_X = X
#tot_X_train, tot_X_test, tot_y_train, tot_y_test = train_test_split(tot_X, tot_y,test_size = .4)
#tot_X_val, tot_X_test, tot_y_val, tot_y_test = train_test_split(tot_X_test, tot_y_test,test_size = .5)
#
## define model function
#def tot_do_rf_model(bootstrap, min_samples_leaf, n_estimators,min_samples_split, 
#          max_features, max_depth, oob_score, n_jobs, criterion, 
#          X_train1, X_test1, y_train1, y_test1 ):
#    parameters = {'bootstrap': bootstrap, #True
#                      'min_samples_leaf': min_samples_leaf, #10
#                      'n_estimators': n_estimators, #300
#                      'min_samples_split': min_samples_split, #2
#                      'max_features': max_features, #log2
#                      'max_depth': max_depth, #none
#                      'oob_score': oob_score, #True
#                      'n_jobs': n_jobs, #2
#                      'criterion': criterion
#                      } #mae
#    tot_rf = RandomForestRegressor(**parameters)
#    tot_rf.fit(X_train1, y_train1)
##    model_predict = rf.predict(X_test).astype(int)
##    actual = y_test.astype(int)
#
#    #### rf model evaluation
#    tot_train_score = tot_rf.score(X_train1,y_train1)
#    tot_test_score = tot_rf.score(X_test1,y_test1)
#    tot_tr_minus_tst = tot_train_score-tot_test_score
#    tot_tr_over_tst = tot_train_score/tot_test_score
#
#    return([tot_train_score,tot_test_score,tot_tr_minus_tst,tot_rf,tot_tr_over_tst])
#
#
#
#tot_bootstrap_opts = ['True','False']
#tot_min_samples_leaf_opts = [20,40]
#tot_n_estimators_opts = [100]
#tot_min_samples_split_opts = [100]
#tot_max_features_opts = ['auto']#'log2','sqrt',]
#tot_max_depth_opts = [10,40] #none
#tot_oob_score_opts = [True]#,False]
#tot_n_jobs_opts = [2]
#tot_criterion_opts = ['mse','mae']
#tot_min_impurity_dec_opts = [.2] #A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
#
#tot_param_objects = [tot_bootstrap_opts, tot_min_samples_leaf_opts,tot_n_estimators_opts,tot_min_samples_split_opts,tot_max_features_opts,tot_max_depth_opts
#               ,tot_oob_score_opts,tot_n_jobs_opts,tot_criterion_opts]
#tot_param_names = ["tot_bootstrap_opts", "tot_min_samples_leaf_opts","tot_n_estimators_opts","tot_min_samples_split_opts","tot_max_features_opts","tot_max_depth_opts"
#               ,"tot_oob_score_opts","tot_n_jobs_opts","tot_criterion_opts"]
#
##build cartesian product of parameters
#tot_index1 = pd.MultiIndex.from_product(tot_param_objects, names = tot_param_names)
#tot_eval_df = pd.DataFrame(index = tot_index1).reset_index()
#
##RandomForestRegressor()
#
#tot_result_df = pd.DataFrame(columns = ['tot_iteration','tot_train_score','tot_test_score','tot_tr_minus_tst','tot_tr_over_tst'])
#for i in range(len(tot_eval_df)): 
#    print(i)
#    x,y,z,model,aa = tot_do_rf_model(
#          bootstrap = tot_eval_df.loc[i,"tot_bootstrap_opts"], #True
#          min_samples_leaf= tot_eval_df.loc[i,"tot_min_samples_leaf_opts"], #10
#          n_estimators= tot_eval_df.loc[i,"tot_n_estimators_opts"], #300
#          min_samples_split= tot_eval_df.loc[i,"tot_min_samples_split_opts"], #2
#          max_features= tot_eval_df.loc[i,"tot_max_features_opts"], #log2
#          max_depth= None, #tot_eval_df.loc[i,"tot_max_depth_opts"], #none
#          oob_score= tot_eval_df.loc[i,"tot_oob_score_opts"], #True
#          n_jobs= tot_eval_df.loc[i,"tot_n_jobs_opts"], #2
#          criterion= tot_eval_df.loc[i,"tot_criterion_opts"],
#          X_train1=tot_X_train, X_test1=tot_X_test, y_train1=tot_y_train, y_test1=tot_y_test
#            )
#    tot_result_df.loc[i,:] = [i,x,y,z,aa]
#
#
#tot_eval_df=tot_eval_df.merge(tot_result_df,left_on=tot_eval_df.index,right_on='tot_iteration',how='left')
#plt.plot(tot_eval_df['tot_train_score'],tot_eval_df['tot_test_score'],'o')
#
#
#tot_eval_df = tot_eval_df.sort_values(by='tot_tr_over_tst',ascending=False).reset_index(drop=True)
#tot_best_params = tot_eval_df[tot_eval_df.index==0]
#
## run model on best params
#tot_rf=tot_do_rf_model(bootstrap = tot_best_params.loc[0,"tot_bootstrap_opts"], #True
#          min_samples_leaf= tot_best_params.loc[0,"tot_min_samples_leaf_opts"], #10
#          n_estimators= tot_best_params.loc[0,"tot_n_estimators_opts"], #300
#          min_samples_split= tot_best_params.loc[0,"tot_min_samples_split_opts"], #2
#          max_features= tot_best_params.loc[0,"tot_max_features_opts"], #log2
#          max_depth= tot_best_params.loc[0,"tot_max_depth_opts"], #none
#          oob_score= tot_best_params.loc[0,"tot_oob_score_opts"], #True
#          n_jobs= tot_best_params.loc[0,"tot_n_jobs_opts"], #2
#          criterion= tot_best_params.loc[0,"tot_criterion_opts"],
#          X_train1=tot_X_train, X_test1=tot_X_val, y_train1=tot_y_train, y_test1=tot_y_val
#          )
#tot_rfm = tot_rf[3]
#tot_model_predict = tot_rfm.predict(tot_X_val).astype(int)
#tot_actual = tot_y_val.astype(int)
#
#print("The Train is {}. Test is {}. Validation is {}. ".format(tot_rf[0],tot_best_params.loc[0,'tot_test_score'], tot_rf[1]) )
#print("R2 of Validation is: {}".format(r2_score(tot_actual, tot_model_predict)))
#
###feature importance
#def tot_r2(tot_rfm, tot_X_train, tot_y_train):
#    return r2_score(tot_y_train, tot_rfm.predict(tot_X_train))
#tot_perm_imp_rfpimp = permutation_importances(tot_rfm, tot_X_train, tot_y_train, tot_r2)
#tot_perm_imp_rfpimp.reset_index(drop = False, inplace = True)
#
#
## plots
#tot_fig, (tot_ax1, tot_ax2) = plt.subplots(1, 2)
#tot_fig.suptitle('MODEL_EVALUATIONs')
#tot_ax1.plot(tot_eval_df['tot_train_score'],tot_eval_df['tot_test_score'],'o')
#
#tot_coef = np.polyfit(tot_model_predict,tot_actual,1)
#tot_poly1d_fn = np.poly1d(tot_coef) 
#
#tot_ax2.plot(tot_model_predict,tot_actual, 'o')
#
## standard deviation
#abs_variance_val = abs(tot_model_predict-tot_actual)
#
#print("The Point Standard Deviation is: {}".format(  statistics.stdev(abs_variance_val)  ) )
#
#
#visualizer = ResidualsPlot(tot_rfm)
#visualizer.fit(tot_X_train, tot_y_train)  # Fit the training data to the visualizer
#visualizer.score(tot_X_test, tot_y_test)  # Evaluate the model on the test data
#visualizer.show()                 # Finalize and render the figure
#
#
#
#
#
#
#
#
#
#
#####more models
#from sklearn.ensemble import MinMaxScaler
#
## Create the scaler object with a range of 0-1
#scaler = MinMaxScaler(feature_range=(0, 1))
## Fit on the training data
#scaler.fit(X)
## Transform both the training and testing data
#X = scaler.transform(X)
#X_test = scaler.transform(X_test)
#
#from sklearn.ensemble import GradientBoostingRegressor
#
## Create the model
#gradient_boosted = GradientBoostingRegressor()
#
## Fit the model on the training data
#gradient_boosted.fit(X, y)
#
## Make predictions on the test data
#predictions = gradient_boosted.predict(X_test)
#
## Evaluate the model
#mae = np.mean(abs(predictions - y_test))
#
#print('Gradient Boosted Performance on the test set: MAE = %0.4f' % mae)