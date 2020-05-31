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
from sklearn.model_selection import train_test_split
from rfpimp import permutation_importances
from sklearn.metrics import r2_score
import statistics
from yellowbrick.regressor import ResidualsPlot

URL_races = 'http://usat.whitegoosetech.com/races'
URL_races_2 = requests.get(URL_races).content
soup_races = BeautifulSoup(URL_races_2)
soup_races_tables = soup_races.find_all('table')
race_rows = []
for table in soup_races_tables:
    for row in table.find_all('a'):
        race_rows.append(row)

race_links = ['http://usat.whitegoosetech.com'+i['href'] for i in race_rows]

df_races = pd.DataFrame()


#chromium driver
driver = webdriver.Chrome()

for URL_race in race_links:
    #URL_race = 'http://usat.whitegoosetech.com/races/360'
    
    
    driver.get(URL_race)
    soup = BeautifulSoup(driver.page_source,"lxml")
    
    abbrevs = {'Junior Elite Male':'jem'
                ,'Junior Elite Female':'jef'
                ,'Youth Elite Male':'yem'
                ,'Youth Elite Female':'yef' }
    
    
    
    
    race = soup.h2.text
    race_date_loc = soup.find("div", {"class": "center"}).text.split('\n')
    race_date_loc = [i for i in race_date_loc if i != '']
    race_date, race_loc = race_date_loc
    
    tab_names = soup.find_all("a", {"data-toggle": "tab"})
    tab_names_clean = [s.text for s in tab_names]
    
    
    for tab_name in tab_names_clean:
        print(tab_name)
        abbrev = abbrevs[tab_name]
        
        driver.find_element_by_link_text(tab_name).click()
        results_num_list = driver.find_elements_by_tag_name('option')
        results_num_list2 = [i.text for i in results_num_list]
        results_num_list2 = list(map(int, [i for i in results_num_list2 if i != ''])) 
        result_selection = str(max(results_num_list2))
        
        results_num_loc = driver.find_element_by_tag_name('select')
    #    results_num_loc.get_attribute('outerHTML') del
    
    #trying another way
        
        select_test = driver.find_element_by_name('resultsTable-'+abbrev+'_length')
        select = Select(select_test)  
        select.select_by_value(result_selection)
        
    #    select = Select(results_num_loc) del
    
    #    select.select_by_value(result_selection) del    
        
        data = []
        table = driver.find_element_by_id('resultsTable-'+abbrev)
    #    table.get_attribute('outerHTML')
        
        table_head = table.find_element_by_tag_name('thead')
        rows_head = table_head.find_elements_by_tag_name('th')
        rows_head_clean = [s.text for s in rows_head]
        
        
        table_body = table.find_element_by_tag_name('tbody')
        
        rows = table_body.find_elements_by_tag_name('tr')
        for row in rows:
            cols = row.find_elements_by_tag_name('td')
            cols = [ele.text.strip() for ele in cols]
            data.append(cols)
        
        df_data = pd.DataFrame(data, columns=rows_head_clean)
        df_data['Race_Name'] = race
        df_data['Division'] = tab_name
        race_date_f = datetime.strptime(race_date , '%A, %B %d, %Y')
        df_data['Date'] = race_date_f.strftime('%Y-%m-%d')
        df_data['Location'] = race_loc
    
        print(df_data)
    #    df_races.append(df_data)
        df_races = pd.concat([df_races,df_data], axis=0, join='outer')
        print(df_races)
        #results = soup.find(id='ResultsContainer')
        #job_elems = results.find_all('section', class_='card-content')

df_races= df_races.reset_index(drop='True')

driver.quit()

#cached data run
RACES_FILENAME = 'C:/Users/Read/Desktop/Code/Python/USAT/RACE_DATA.csv'
df_races.to_csv(RACES_FILENAME, encoding='utf-8', index=True)


"""
data notes
-some records have 00:00:00
-dnf records have out of pattern low times and times that make no sense, messy
-some records have out of pattern high times eg with 01:20:05 bike time
-some records have  35:09:00 format
-

idea
-does the time make sense based on the race, is it within some reasonable std dev of mean time
-
"""

#pull saved data
df_races2 = pd.read_csv(RACES_FILENAME)
df_races2 = df_races2.drop(['Unnamed: 0'], axis = 1) 

#clean up data
time_cols = ['Swim','T1','Bike','T2','Run','Finish']
df_races2.loc[:,time_cols] = df_races2.loc[:,time_cols].replace('00:00',np.NaN)
dnf_list = (df_races2.Place=='0') | (df_races2.Place=='DNF')
df_races2.loc[dnf_list,'Place'] = np.NaN
df_races2['Place'] = pd.to_numeric(df_races2['Place'])

#clean up times, convert to seconds
bike_clean_rows = (df_races2.loc[:,'Bike'].str.count(':')==2) & (df_races2.loc[:,'Bike'].str.slice(stop=2) != '01')
df_races2.loc[ bike_clean_rows==True,'Bike'] = df_races2.loc[ bike_clean_rows==True,'Bike'].str.slice(stop=5)
T2_clean_rows = (df_races2.loc[:,'T2'].str.count(':')==2) & (df_races2.loc[:,'T2'].str.slice(stop=2) != '01')
df_races2.loc[ T2_clean_rows==True,'T2'] = df_races2.loc[ T2_clean_rows==True,'T2'].str.slice(stop=5)

#add '00:' onto length 5
for time_col in time_cols:
    df_races2.loc[df_races2.loc[:,time_col].str.len()==5,time_col] = '00:'+df_races2.loc[df_races2.loc[:,time_col].str.len()==5,time_col]

def get_sec(time_str):
    if pd.isnull(time_str) == True:
        result = time_str
    else:
        """Get Seconds from time."""
        h, m, s = time_str.split(':')
        result =  int(h) * 3600 + int(m) * 60 + int(s)
    return result

for time_col in time_cols:
    df_races2.loc[:,time_col] = df_races2.loc[:,time_col].apply(get_sec)

#calc bike out gap
df_races2['Bike_Out'] = df_races2.Swim+df_races2.T1
df_races2['Bike_Out_lag'] = (df_races2.sort_values(by=['Bike_Out'], ascending=True).groupby(['Race_Name','Division','Date'])['Bike_Out'].shift(1))
df_races2['Bike_Out_gap'] = df_races2.Bike_Out-df_races2.Bike_Out_lag

#calc bike in for Bobby mcgee template
df_races2['Bike_In'] = df_races2.Swim+df_races2.T1+df_races2.Bike
#df_races2['Bike_In_lag'] = (df_races2.sort_values(by=['Bike_Out'], ascending=True).groupby(['Race_Name','Division','Date'])['Bike_In'].shift(1))
#df_races2['Bike_In_gap'] = df_races2.Bike_Out-df_races2.Bike_In_lag

#calc run gap
df_races2['Run_Out'] = df_races2.Swim+df_races2.T1+df_races2.Bike+df_races2.T2
df_races2['Run_Out_lag'] = (df_races2.sort_values(by=['Run_Out'], ascending=True).groupby(['Race_Name','Division','Date'])['Run_Out'].shift(1))
df_races2['Run_Out_gap'] = df_races2.Run_Out-df_races2.Run_Out_lag

#calc number of athletes within plus minus 5 seconds of Bikeout runout
sec_window = 5 
def comp_time_ct(row=df_races2.iloc[764],measure='Bike_Out',df=df_races2,sec_window1=sec_window):
    row=pd.DataFrame(row.values.reshape(1,len(row.values)),columns=row.index)
    if row.loc[0,measure]==np.NaN:
        result1 = np.NaN
    else:
        ind_time = row.loc[0,measure]
        row_filtered = row[['Race_Name','Division','Date']]
        grp_time = df.merge(row_filtered, on=['Race_Name','Division','Date'], how='inner')[measure]
#        abs_diff = grp_time.subtract(ind_time)
        abs_diff = grp_time.map(lambda x: abs(x-ind_time) )
        result1 = sum(  (abs_diff<=sec_window1)*1 ,-1)
    return result1



grp_Swim_T1 = ['Race_Name','Division','Date','Name']

df_races2['Bike_Out_w/in_'+str(sec_window)] = df_races2.apply(lambda row: comp_time_ct(row=row,measure='Bike_Out',df=df_races2,sec_window1=sec_window) ,axis=1) 
df_races2['Run_Out_w/in_'+str(sec_window)] = df_races2.apply(lambda row: comp_time_ct(row=row,measure='Run_Out',df=df_races2,sec_window1=sec_window) ,axis=1) 



#zzz_test_swim_t1_ct= df_races2[df_races.Swim_T1>].groupby(['Race_Name','Division','Date','Name'], as_index=False)['Swim_T1']
#df_races2['Swim_T1_w/in_'+sec_window]

dist_dict = {'Junior Elite Male': {'Swim':750,'Bike':19.3121,'Run':5,'Gender': 'Male'}
                ,'Junior Elite Female':{'Swim':750,'Bike':19.3121,'Run':5,'Gender': 'Female'}
                ,'Youth Elite Male':{'Swim':375,'Bike':9.65606,'Run':2.41402,'Gender': 'Male'}
                ,'Youth Elite Female':{'Swim':376,'Bike':9.65606,'Run':2.41402,'Gender': 'Female'} }


#used to create z scores, ended up just picking valid times manually in a dict
#for time_col in time_cols:
#    grp = 'Division'
#    df_grp_division = df_races2.groupby(grp, as_index=False)[grp,time_col].agg(['mean','std'])
#    df_grp_division.columns = [time_col+'_avg',time_col+'_std']
#    df_races2 = df_races2.merge(df_grp_division, left_on = 'Division', right_index=True)
#
#for time_col in time_cols:
#    df_races2.loc[:,time_col+'_z'] = abs( (df_races2[time_col] - df_races2[time_col+'_avg'] ) / df_races2[time_col+'_std'] )

for sport in ['Swim','Bike','Run']:
    dict_map = df_races2['Division'].map(dist_dict)
    df_races2.loc[:,sport+'_d'] = [i.get(sport) for i in dict_map]

df_races2['Swim_sp100'] = df_races2.Swim/df_races2.Swim_d*100
df_races2['Bike_kph'] = df_races2.Bike_d/df_races2.Bike*3600
df_races2['Run_spk'] = df_races2.Run/df_races2.Run_d

## build _val columns that 1 or 0 based on valid result for each split of race
dict_valid = { 'Swim': {'Meas' : 'Swim','Max': 1200,'Min': 100},
                'T1': {'Meas' : 'T1','Max': 250,'Min': 0},
                'Bike': {'Meas' : 'Bike','Max': 3000,'Min': 600},
                'T2': {'Meas' : 'T2','Max': 150,'Min': 0},
                'Run': {'Meas' : 'Run_spk','Max': 2000,'Min': 200}
                }

list_valid = time_cols.copy()
list_valid.remove('Finish')
for time_col in list_valid :
    meas,min1,max1 = [dict_valid.get(time_col).get(key) for key in ['Meas','Min','Max'] ]
    bounds = (df_races2[meas] > min1) & (df_races2[meas] < max1)
    df_races2[time_col+'_val'] = bounds*1

df_races2['Finish_val'] = df_races2.loc[:, [i+'_val' for i in time_cols if i != 'Finish']].sum(axis=1)
df_races2['Finish_val'] = df_races2['Finish_val'].apply(lambda x: 1 if x==5 else 0) 
df_races2['Finish'] = df_races2.loc[:, time_cols].sum(axis=1)#df_races2['Finish_val']==1

#
#for time_col in list_valid :
#    df_races2[time_col+'_rnk'] = df_races2[df_races2[time_col+'_val']==1][time_col].rank(method='min')

#calc top 3 in each event and append on at the individual level
for time_col in time_cols:
    group_cols = ['Race_Name','Division','Date']#,'Racing Age'] # chose not to group by racing age
    df_races2[time_col+'_rnk'] = df_races2[df_races2[time_col+'_val']==1].groupby(group_cols, as_index=False)[time_col].rank(method='min')
    gb_rnk = df_races2[df_races2[time_col+'_rnk']<=3].groupby(group_cols, as_index=False)[time_col].mean()
    df_races2 = df_races2.merge(gb_rnk,how='left',left_on = group_cols, right_on=group_cols,suffixes=('', '_Top3'))

#calc top 3 overall times
place_list=[1,2,3]
time_cols2 = time_cols+['Bike_In']
for time_col in time_cols2:
    print(time_col)
    group_cols = ['Race_Name','Division','Date']
    for place in place_list:
        print(place)
        group_cols2 = group_cols + [time_col]
        print(group_cols2)
        pl_rnk = df_races2[df_races2['Place']==place].groupby(group_cols, as_index=False)[time_col].mean()
        print(pl_rnk.shape)
        suffix2 = '_PL'+str(place)
        df_races2 = df_races2.merge(pl_rnk,how='left',on=group_cols,suffixes=('',suffix2 ))
        df_races2[time_col+suffix2+'Diff']=df_races2[time_col+suffix2]-df_races2[time_col]

#    for col_from in df_Top3.columns:
#        if col_from in time_cols: df_Top3.rename(columns={col_from:col_from+'_Top3'}, inplace=True)
    
#    df_races2 = df_races2.merge(df_Top3, left_on = group_cols, right_on=group_cols)


for time_col in time_cols:
    df_races2.loc[:,time_col+'_pct'] = df_races2[time_col+'_Top3']/df_races2[time_col]

df_races2['Gender'] = df_races2['Division'].apply(lambda x: 'Female' if ((x == 'Junior Elite Female') | (x == 'Youth Elite Female' )) else 'Male')

df_races2 = df_races2.merge(df_races2[df_races2['Racing Age']>=18].groupby(['Name'], as_index=False)['Place'].max(),how='left', left_on='Name',right_on='Name',suffixes=('', '_MAX_18+') )
#df_races2 = df_races2.drop(['Max_Place_19','Place_MAX_19'],axis=1)

#--nth race
df_races2['Nth_Race'] = df_races2.apply(lambda row: sum(df_races2.loc[ (row.Name==df_races2.Name.values)&(row.Date>=df_races2.Date.values),'Swim_val']  ) ,axis=1)


#df_races2 = df_races2.drop(['MIN_RACE_AGE_IN_DATA','MIN_DATE_IN_DATA'],axis=1)
grp_min_date_age = df_races2.groupby('Name')['Date','Racing Age'].min()
grp_min_date_age = grp_min_date_age.rename(columns={"Date": "MIN_DATE_IN_DATA", "Racing Age": "MIN_RACING_AGE_IN_DATA"})
df_races2 = df_races2.merge(grp_min_date_age,how='left',left_on = 'Name', right_index=True)

df_races2 = df_races2.merge(df_races2[df_races2['Racing Age']==16].groupby(['Name'], as_index=False)['Place'].max(),how='left', left_on='Name',right_on='Name',suffixes=('', '_MAX_16') )



RACES_FILENAME_final = 'C:/Users/Read/Desktop/Code/Python/USAT/RACE_DATA_final.csv'
df_races2.to_csv(RACES_FILENAME_final, encoding='utf-8', index=True)
#pull saved data
df_races2 = pd.read_csv(RACES_FILENAME_final)
df_races2 = df_races2.drop(['Unnamed: 0'], axis = 1) 










###################
#### Modeling #####
###################

df_races3 = pd.read_csv(RACES_FILENAME_final)
df_races3 = df_races3.drop(['Unnamed: 0'], axis = 1) 

#modify/fix columns, consider moving to prev section
df_races3['Date'] = pd.to_datetime(df_races3['Date'], format='%Y-%m-%d') 
df_races3.loc[(df_races3.Finish_pct>1.3)|(df_races3.Finish_pct<.60),'Finish_val'] = 0  #this is a bandaid for illogical finish times

#column metadata
df_cols3 = pd.DataFrame(df_races3.columns,columns=['Field'])

zzz_time_col = 'Swim'
# y = target variable pct of top 3 time in race for each leg so 5 models, one for each leg
for time_col in time_cols:
    df_races3[time_col+'_valid'] = (df_races3[time_col]*df_races3[time_col+'_val']).replace(0,np.NaN)

#
# x =
# racing age current
#DONE!

# n races
#DONE!
    
# teamsize = n races in last 365 days, need a df by team by date


def tm_races_ct(row):
    bool1 = (row.Team==df_races3.Team.values)
    bool2 = (row.Date>=df_races3.Date ).to_numpy()
    bool3 = (df_races3.Date>=( row.Date - pd.DateOffset(years=1) ) ).to_numpy()
    tm_race_bool =  bool1 & bool2 & bool3
    return len(df_races3.loc[tm_race_bool,'Team'])

df_races3['TM_RACES_1YR'] = df_races3.apply(lambda row: tm_races_ct(row) ,axis=1) 

#--prior 1 race pct swim, lag shift 1 by athlete
for time_col in time_cols:
    df_races3[time_col+'_pct_pri'] = df_races3.sort_values(by=['Date'], ascending=True).groupby(['Name'])[time_col+'_pct'].shift(1)

#--prior all race pct swim avg
#--max pct swim in all prior races
#--min pct swim in all prior races
#--median pct swim in all prior races
def all_pri_race_stat(row,stat,time_col):
    bool1 = (row.Name==df_races3.Name).to_numpy()
    bool2 = (row.Date>df_races3.Date ).to_numpy()
    bool3 = (df_races3[time_col+'_val']==1).to_numpy()
    tm_race_bool =  bool1 & bool2 & bool3
    return df_races3.loc[tm_race_bool,time_col+'_pct'].agg(stat)

pri_race_stats = ['mean','max','min','median']

#this takes a while
for time_col in time_cols:
    for stat in pri_race_stats:
        df_races3[time_col+'_pct_'+stat] = df_races3[['Name','Date']].apply( lambda row: all_pri_race_stat(row,stat,time_col) ,axis=1)


#advanced
# of others racing in this race, whats your pctile of prior race swim pct vs that cohort
def competitor_pctile(row,time_col):
    bool1 = (row.Race_Name==df_races3.Race_Name).to_numpy()
    bool2 = (row.Division==df_races3.Division).to_numpy()
    bool3 = (row.Date==df_races3.Date ).to_numpy()
    bool4 = (df_races3[time_col+'_val']==1).to_numpy()
    bool5 = (row.Name!=df_races3.Name).to_numpy()
    tm_race_bool =  bool1 & bool2 & bool3 & bool4 & bool5
    pctile_array = df_races3.loc[tm_race_bool,time_col+'_pct']
    pri_percentile = stats.percentileofscore(pctile_array, row[time_col+'_pct_pri'])
    return pri_percentile

#this takes a while
for time_col in time_cols:
    for stat in pri_race_stats:
        df_races3[time_col+'_pri_pct_pctile'] = df_races3.apply( lambda row: competitor_pctile(row,time_col) ,axis=1) 

model_results = []
time_col1 = 'Finish'    
Target = [time_col1+'_pct']
M1_Feature_List = ['Racing Age','Nth_Race','TM_RACES_1YR',time_col1+'_pct_pri',time_col1+'_pct_mean',time_col1+'_pct_max',time_col1+'_pct_min',time_col1+'_pct_median',time_col+'_race_pctile']

df_model = df_races3.copy().loc[(df_races3[time_col1+'_val']==1),M1_Feature_List+Target]
df_model['Racing Age'] = df_model['Racing Age'].replace(np.NaN,13)
df_model['Nth_Race'] = df_model['Nth_Race'].replace(np.NaN,1)
df_model['TM_RACES_1YR'] = df_model['TM_RACES_1YR'].replace(np.NaN,0)
df_model[time_col+'_race_pctile'] = df_model[time_col+'_race_pctile'].replace(np.NaN,.5)
df_model = df_model.replace(np.NaN,.85)
df_model = df_model.dropna()

X = df_model.loc[:,M1_Feature_List]
Y = df_model.loc[:,Target]


X_train, X_test, y_train, y_test = train_test_split(X, Y,test_size = .2)


parameters = {'bootstrap': 'True'
                  ,'min_samples_leaf': 20
                  ,'n_estimators': 300
                  ,'min_samples_split': 5
                  ,'max_features': 'auto'
                  ,'max_depth': 30
                  ,'oob_score': True
                  ,'n_jobs': 2
                  ,'criterion': 'mse'
#                  ,'min_impurity_decrease': .01
                  } 
rf = RandomForestRegressor(**parameters)
rfm = rf.fit(X_train, y_train)
#    model_predict = rf.predict(X_test).astype(int)
#    actual = y_test.astype(int)

#### rf model evaluation
train_score = rf.score(X_train,y_train)
test_score = rf.score(X_test,y_test)

print("The Train is {}. Test is {}. ".format(train_score,test_score) )

def tot_r2(rf,X_train, y_train):
    return r2_score(y_train, rf.predict(X_train))
perm_imp_rfpimp = permutation_importances(rf, X_train, y_train, tot_r2)

plt.plot(rf.predict(X_train),y_train,'o' )










tot_y = Y
tot_X = X
tot_X_train, tot_X_test, tot_y_train, tot_y_test = train_test_split(tot_X, tot_y,test_size = .4)
tot_X_val, tot_X_test, tot_y_val, tot_y_test = train_test_split(tot_X_test, tot_y_test,test_size = .5)

# define model function
def tot_do_rf_model(bootstrap, min_samples_leaf, n_estimators,min_samples_split, 
          max_features, max_depth, oob_score, n_jobs, criterion, 
          X_train1, X_test1, y_train1, y_test1 ):
    parameters = {'bootstrap': bootstrap, #True
                      'min_samples_leaf': min_samples_leaf, #10
                      'n_estimators': n_estimators, #300
                      'min_samples_split': min_samples_split, #2
                      'max_features': max_features, #log2
                      'max_depth': max_depth, #none
                      'oob_score': oob_score, #True
                      'n_jobs': n_jobs, #2
                      'criterion': criterion
                      } #mae
    tot_rf = RandomForestRegressor(**parameters)
    tot_rf.fit(X_train1, y_train1)
#    model_predict = rf.predict(X_test).astype(int)
#    actual = y_test.astype(int)

    #### rf model evaluation
    tot_train_score = tot_rf.score(X_train1,y_train1)
    tot_test_score = tot_rf.score(X_test1,y_test1)
    tot_tr_minus_tst = tot_train_score-tot_test_score
    tot_tr_over_tst = tot_train_score/tot_test_score

    return([tot_train_score,tot_test_score,tot_tr_minus_tst,tot_rf,tot_tr_over_tst])



tot_bootstrap_opts = ['True','False']
tot_min_samples_leaf_opts = [20,40]
tot_n_estimators_opts = [100]
tot_min_samples_split_opts = [100]
tot_max_features_opts = ['auto']#'log2','sqrt',]
tot_max_depth_opts = [10,40] #none
tot_oob_score_opts = [True]#,False]
tot_n_jobs_opts = [2]
tot_criterion_opts = ['mse','mae']
tot_min_impurity_dec_opts = [.2] #A node will be split if this split induces a decrease of the impurity greater than or equal to this value.

tot_param_objects = [tot_bootstrap_opts, tot_min_samples_leaf_opts,tot_n_estimators_opts,tot_min_samples_split_opts,tot_max_features_opts,tot_max_depth_opts
               ,tot_oob_score_opts,tot_n_jobs_opts,tot_criterion_opts]
tot_param_names = ["tot_bootstrap_opts", "tot_min_samples_leaf_opts","tot_n_estimators_opts","tot_min_samples_split_opts","tot_max_features_opts","tot_max_depth_opts"
               ,"tot_oob_score_opts","tot_n_jobs_opts","tot_criterion_opts"]

#build cartesian product of parameters
tot_index1 = pd.MultiIndex.from_product(tot_param_objects, names = tot_param_names)
tot_eval_df = pd.DataFrame(index = tot_index1).reset_index()

#RandomForestRegressor()

tot_result_df = pd.DataFrame(columns = ['tot_iteration','tot_train_score','tot_test_score','tot_tr_minus_tst','tot_tr_over_tst'])
for i in range(len(tot_eval_df)): 
    print(i)
    x,y,z,model,aa = tot_do_rf_model(
          bootstrap = tot_eval_df.loc[i,"tot_bootstrap_opts"], #True
          min_samples_leaf= tot_eval_df.loc[i,"tot_min_samples_leaf_opts"], #10
          n_estimators= tot_eval_df.loc[i,"tot_n_estimators_opts"], #300
          min_samples_split= tot_eval_df.loc[i,"tot_min_samples_split_opts"], #2
          max_features= tot_eval_df.loc[i,"tot_max_features_opts"], #log2
          max_depth= None, #tot_eval_df.loc[i,"tot_max_depth_opts"], #none
          oob_score= tot_eval_df.loc[i,"tot_oob_score_opts"], #True
          n_jobs= tot_eval_df.loc[i,"tot_n_jobs_opts"], #2
          criterion= tot_eval_df.loc[i,"tot_criterion_opts"],
          X_train1=tot_X_train, X_test1=tot_X_test, y_train1=tot_y_train, y_test1=tot_y_test
            )
    tot_result_df.loc[i,:] = [i,x,y,z,aa]


tot_eval_df=tot_eval_df.merge(tot_result_df,left_on=tot_eval_df.index,right_on='tot_iteration',how='left')
plt.plot(tot_eval_df['tot_train_score'],tot_eval_df['tot_test_score'],'o')


tot_eval_df = tot_eval_df.sort_values(by='tot_tr_over_tst',ascending=False).reset_index(drop=True)
tot_best_params = tot_eval_df[tot_eval_df.index==0]

# run model on best params
tot_rf=tot_do_rf_model(bootstrap = tot_best_params.loc[0,"tot_bootstrap_opts"], #True
          min_samples_leaf= tot_best_params.loc[0,"tot_min_samples_leaf_opts"], #10
          n_estimators= tot_best_params.loc[0,"tot_n_estimators_opts"], #300
          min_samples_split= tot_best_params.loc[0,"tot_min_samples_split_opts"], #2
          max_features= tot_best_params.loc[0,"tot_max_features_opts"], #log2
          max_depth= tot_best_params.loc[0,"tot_max_depth_opts"], #none
          oob_score= tot_best_params.loc[0,"tot_oob_score_opts"], #True
          n_jobs= tot_best_params.loc[0,"tot_n_jobs_opts"], #2
          criterion= tot_best_params.loc[0,"tot_criterion_opts"],
          X_train1=tot_X_train, X_test1=tot_X_val, y_train1=tot_y_train, y_test1=tot_y_val
          )
tot_rfm = tot_rf[3]
tot_model_predict = tot_rfm.predict(tot_X_val).astype(int)
tot_actual = tot_y_val.astype(int)

print("The Train is {}. Test is {}. Validation is {}. ".format(tot_rf[0],tot_best_params.loc[0,'tot_test_score'], tot_rf[1]) )
print("R2 of Validation is: {}".format(r2_score(tot_actual, tot_model_predict)))

##feature importance
def tot_r2(tot_rfm, tot_X_train, tot_y_train):
    return r2_score(tot_y_train, tot_rfm.predict(tot_X_train))
tot_perm_imp_rfpimp = permutation_importances(tot_rfm, tot_X_train, tot_y_train, tot_r2)
tot_perm_imp_rfpimp.reset_index(drop = False, inplace = True)


# plots
tot_fig, (tot_ax1, tot_ax2) = plt.subplots(1, 2)
tot_fig.suptitle('MODEL_EVALUATIONs')
tot_ax1.plot(tot_eval_df['tot_train_score'],tot_eval_df['tot_test_score'],'o')

tot_coef = np.polyfit(tot_model_predict,tot_actual,1)
tot_poly1d_fn = np.poly1d(tot_coef) 

tot_ax2.plot(tot_model_predict,tot_actual, 'o')

# standard deviation
abs_variance_val = abs(tot_model_predict-tot_actual)

print("The Point Standard Deviation is: {}".format(  statistics.stdev(abs_variance_val)  ) )


visualizer = ResidualsPlot(tot_rfm)
visualizer.fit(tot_X_train, tot_y_train)  # Fit the training data to the visualizer
visualizer.score(tot_X_test, tot_y_test)  # Evaluate the model on the test data
visualizer.show()                 # Finalize and render the figure










####more models
from sklearn.ensemble import MinMaxScaler

# Create the scaler object with a range of 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
# Fit on the training data
scaler.fit(X)
# Transform both the training and testing data
X = scaler.transform(X)
X_test = scaler.transform(X_test)

from sklearn.ensemble import GradientBoostingRegressor

# Create the model
gradient_boosted = GradientBoostingRegressor()

# Fit the model on the training data
gradient_boosted.fit(X, y)

# Make predictions on the test data
predictions = gradient_boosted.predict(X_test)

# Evaluate the model
mae = np.mean(abs(predictions - y_test))

print('Gradient Boosted Performance on the test set: MAE = %0.4f' % mae)