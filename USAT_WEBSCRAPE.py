# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 20:20:05 2020

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