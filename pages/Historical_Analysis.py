# ANCHOR Parameters - Import Packages
from pyrsistent import v
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
from datetime import date
import plotly.graph_objects as go
import datetime
import json
import requests

############################################################################################################
############################################################################################################
### ANCHOR Parameters - Instantiate Jira

from atlassian import Jira

jira_instance = Jira(
    url = st.secrets["db_url"],
    username = st.secrets["db_username"],
    password = st.secrets["db_password"]
)

############################################################################################################
############################################################################################################
### ANCHOR - Initialize ChatGPT Service

def initialize_service(key):
    # Load the service key
    with open(key, "r") as key_file:
        svc_key = json.load(key_file)

    svc_url = svc_key["url"]
    client_id = svc_key["uaa"]["clientid"]
    client_secret = svc_key["uaa"]["clientsecret"]
    uaa_url = svc_key["uaa"]["url"]

    params = {"grant_type": "client_credentials"}
    resp = requests.post(f"{uaa_url}/oauth/token",
                        auth=(client_id, client_secret),
                        params=params)

    token = resp.json()["access_token"]

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    return headers, svc_url

############################################################################################################
############################################################################################################
### ANCHOR Parameters - Set the width of the page

st.set_page_config(layout="wide", page_title="Our Historical Metrics", page_icon="🌱", initial_sidebar_state='auto')

############################################################################################################
############################################################################################################
# ANCHOR Parameters - Global Layout Configuration

# Hide 'Made with Streamlit'

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
            viewerBadge_container__1QSob {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

############################################################################################################
############################################################################################################
# ANCHOR Sidebar configuration

    
submit_button = st.button(label='Run Historical Analysis')
        
if submit_button:
    
    if 'resolved_date' not in st.session_state:
        st.session_state['resolved_date'] = '2024-09-09'
        
    if 'non_working_brazil' not in st.session_state:
        st.session_state['non_working_brazil'] = ['2024-01-01', '2024-02-12', '2024-02-13', 
                                                '2024-02-14', '2024-03-29', '2024-04-21', '2024-05-01', '2024-07-25', 
                                                '2024-09-07', '2024-09-20', '2024-10-12', '2024-11-02', '2024-11-15', 
                                                '2024-11-20', '2024-12-08', '2024-12-25',]

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Home Page

    st.markdown('#')
    st.write("""

    ### Historical Analysis Metrics Dashboard of Intelligent Deal Entry! 🤖🌱🛢️

    --------------------

    """)

    ############################################################################################################
    ############################################################################################################
    ### ANCHOR Parameters - Session State (before main function)

    if 'resolved_date' not in st.session_state:
        # Input string
        date_string = '2024-09-09'
        
        # Convert string to timestamp
        timestamp = pd.to_datetime(date_string)
        date_only = timestamp.date()
        
        st.session_state['resolved_date'] = date_only
        
    if 'non_working_brazil' not in st.session_state:
        st.session_state['non_working_brazil'] = ['2024-01-01', '2024-02-12', '2024-02-13', 
                                                '2024-02-14', '2024-03-29', '2024-04-21', '2024-05-01', '2024-07-25', 
                                                '2024-09-07', '2024-09-20', '2024-10-12', '2024-11-02', '2024-11-15', 
                                                '2024-11-20', '2024-12-08', '2024-12-25',]

    completed_status = ['Completed']
    
    ############################################################################################################
    ############################################################################################################
    # ANCHOR Get All Sprints from Boards

    board_id = [49077]

    df_sprints = pd.DataFrame()
        
    def getSprints(board_id):
        start = 0
        limit = 50
        all_sprints = []

        while True:
            df_sprints_prep = jira_instance.get_all_sprint(board_id, start=start, limit=limit)
            sprints = df_sprints_prep['values']
            
            if not sprints:
                break

            all_sprints.extend(sprints)
            start += limit

        df_sprints_prep = pd.json_normalize(all_sprints)

        df_sprints_prep['startDate'] = pd.to_datetime(df_sprints_prep['startDate'],
                                                    dayfirst=True, format='mixed')
        df_sprints_prep['startDate'] = df_sprints_prep['startDate'].dt.strftime('%Y-%m-%d')
        df_sprints_prep['startDate'] = pd.to_datetime(df_sprints_prep['startDate'],
                                                    dayfirst=True, format='mixed')

        df_sprints_prep['endDate'] = pd.to_datetime(df_sprints_prep['endDate'], dayfirst=True)
        df_sprints_prep['endDate'] = df_sprints_prep['endDate'].dt.strftime('%Y-%m-%d')
        df_sprints_prep['endDate'] = pd.to_datetime(df_sprints_prep['endDate'], dayfirst=True)

        df_sprints_prep = df_sprints_prep.drop(columns=['id', 'self', 'activatedDate', 'originBoardId'], axis=1)
        
        df_sprints_prep = df_sprints_prep.sort_values(by='startDate', ascending=True)
        
        return df_sprints_prep


    for i in range(0, len(board_id)):
        df_sprints_prep = getSprints(board_id[i])
        
        df_sprints = pd.concat([df_sprints, df_sprints_prep], ignore_index=True)
        
    df_sprints = df_sprints.drop_duplicates(keep='first')
    df_sprints = df_sprints.sort_values(by='startDate', ascending= True).reset_index(drop=True)            

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Import Backlog Items (Current Sprint and Completed Items)
    
    def prepareData(df, non_working_days):
        ######## Get the first log for 'Resolved Date' and first log for 'Started Date'
        ### Creates a Data Frame to store the Issue Keys

        df_change_log = pd.DataFrame(columns=['Issue key'])
        # df_resolved_change_log['Issue key'] = df[df['Resolved'].notnull()]['Issue key']
        df_change_log['Issue key'] = df['Issue key']
        df_change_log = df_change_log.reset_index(drop=True)

        df_resolved_transition_log = pd.DataFrame(columns=[
            'field', 'fieldtype', 'from', 'fromString', 'to', 'toString', 'created'
        ])

        df_transition_log = pd.DataFrame(columns=[
            'field', 'fieldtype', 'from', 'fromString', 'to', 'toString', 'created'
        ])

        df.drop(columns=['Resolved', 'Updated'], inplace=True)

        for i in range(0, len(df_change_log), 1):
            get_change_log = jira_instance.get_issue_changelog(df_change_log['Issue key'][i])
            df_get_change_log = pd.json_normalize(get_change_log['histories'], 'items', 'created')
            df_get_change_log['issue_key'] = df_change_log['Issue key'][i]

            if len(df_get_change_log) > 0:
                if len(df_get_change_log.loc[df_get_change_log['toString'] == 'Completed']) > 0:
                    df_resolved_transition_log = pd.concat([df_resolved_transition_log, df_get_change_log.loc[df_get_change_log['toString'] == 'Completed']],ignore_index=True)
                
                if len(df_get_change_log.loc[df_get_change_log['toString'] == 'Obsolete']) > 0:
                    df_resolved_transition_log = pd.concat([df_resolved_transition_log, df_get_change_log.loc[df_get_change_log['toString'] == 'Obsolete']],ignore_index=True)

                if len(df_get_change_log.loc[df_get_change_log['toString'] == 'In Progress']) > 0:
                    df_transition_log = pd.concat([df_transition_log, df_get_change_log.loc[df_get_change_log['toString'] == 'In Progress']],ignore_index=True)

        df_resolved_transition_log['created'] = df_resolved_transition_log['created'].str.split("T").str[0]

        try:
            df_resolved_transition_log = df[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='first').reset_index(drop=True)
        except:
            pass
        
        # df_resolved_transition_log = df_resolved_transition_log[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='first').reset_index(drop=True)

        df_resolved_transition_log['created'] = pd.to_datetime(df_resolved_transition_log['created'], dayfirst=True, format="mixed")

        df_resolved_transition_log = df_resolved_transition_log.rename(columns={'issue_key': 'Issue key', 'created': 'Resolved'})

        df = df.merge(df_resolved_transition_log, how='left', on='Issue key')



        df_transition_log['created'] = df_transition_log['created'].str.split("T").str[0]

        df_transition_log = df_transition_log[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='first').reset_index(drop=True)

        df_transition_log['created'] = pd.to_datetime(df_transition_log['created'], dayfirst=True, format='mixed')

        df_transition_log = df_transition_log.rename(columns={'issue_key': 'Issue key', 'created': 'started'})

        df = df.merge(df_transition_log, how='left', on='Issue key')


        #### Dates preparation
        df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format='mixed')
        df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")
        df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")


        df['Created'] = df['Created'].dt.strftime('%Y-%m-%d')
        df['Resolved'] = df['Resolved'].dt.strftime('%Y-%m-%d')
        df['started'] = df['started'].dt.strftime('%Y-%m-%d')

        df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format="mixed")
        df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")
        df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")


        #### Cria coluna para identificar o número da semana que a entrega aconteceu
        df['week_number'] = df['Resolved'].dt.isocalendar().week

        # #### Converte Lead Time para Dias
        # df['lead_time'] = df['lead_time'].dt.days

        #### Unifica colunas com Sprints
        cols = []
        for col in df.columns:
            if 'Sprint' in col:
                cols.append(col)

        merged_sprints = df[cols]
        merged_sprints = merged_sprints[merged_sprints.columns[0:]].apply(
        lambda x: ','.join(x.dropna()), axis=1)

        df['sprints'] = merged_sprints

        columns_to_remove = [col for col in df if 'Sprint' in col]
        df.drop(columns=columns_to_remove, inplace=True)

        #### Transforma sprints em lista para fazer contagem depois criando coluna 'num_sprints':
        for i in range(len(df['sprints'])):
            if ',' in df['sprints'][i]:
                df['sprints'][i] = df['sprints'][i].split(',')
            else:
                df['sprints'][i] = [df['sprints'][i]]
                
        df['num_sprints'] = df['sprints'].str.len()

        #### Unifica colunas com Parent
        cols = []
        for col in df.columns:
            if 'parent' in col:
                cols.append(col)

        merged_parents = df[cols]
        merged_parents = merged_parents[merged_parents.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype('str')), axis=1)
        merged_parents.tolist()

        df['Parent'] = merged_parents

        # df = df[df['Resolved'] >= resolved_date]
        df = df.reset_index(drop=True)
        df = df.rename(columns={'Custom field (Planned Effort)': 'planned_effort', 
                                'Time Spent': 'time_spent',
                                'Original Estimate': 'original_estimate'})

        #### Time convertion
        df['time_spent'] = df['time_spent'].replace(np.nan, 0)
        df['planned_effort'] = df['planned_effort'].replace(np.nan, 0)
        df['original_estimate'] = df['original_estimate'].replace(np.nan, 0)

        df['hours_spent'] = df['time_spent'].astype(int) / 3600
        df['days_spent'] = df['time_spent'].astype(int)/86400 * 24 / 8

        df['planned_hours'] = df['planned_effort'].astype(int) * 8 * 0.9
        df['planned_effort'] = df['planned_effort'].astype(int)
        
        df['original_estimate'] = df['original_estimate'].astype(int)/86400 * 24 / 8
        
        df['planned_effort'] = df['planned_effort'] + df['original_estimate']

        #### Unifica colunas com Components:
        cols = []
        for col in df.columns:
            if 'Component' in col:
                cols.append(col)

        merged_components = df[cols]
        merged_components = merged_components[merged_components.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype('str')), axis=1)
        merged_components.tolist()

        df['Components'] = merged_components

        columns_to_remove = [col for col in df if 'Component/s' in col]
        df.drop(columns=columns_to_remove, inplace=True)

        #### Creates a new column to state the week number the delivery happened

        df['week_num'] = df['Resolved'].dt.isocalendar().week


        ### Format the dates to get the Business Days

        df['Created'] = df['Created'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '')
        df['started'] = df['started'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '')
        df['Resolved'] = df['Resolved'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '')

        ### Calculate the Business Days for Cycle Time

        df['cycle_time'] = ''

        for i in range(0,len(df), 1):
            if len(df['started'][i]) > 0:
                if len(df['Resolved'][i]) > 0:
                    
                    range_cycle_time = pd.bdate_range(start=df['started'][i], end=df['Resolved'][i], freq="B")
                    
                    df['cycle_time'][i] = len(range_cycle_time)
                    
                    for j in range(0, len(non_working_days), 1):
                        if non_working_days[j] in range_cycle_time:
                            df['cycle_time'][i] -= 1
                    
                else:
                    df['cycle_time'][i] = ''

        df["cycle_time"] = pd.to_numeric(df["cycle_time"], downcast="float")

        #### Calculate the Business Days for Lead Time

        df['lead_time'] = ''

        for i in range(0,len(df), 1):
            if len(df['Created'][i]) > 0:
                if len(df['Resolved'][i]) > 0:
                    
                    range_lead_time = pd.bdate_range(start=df['Created'][i], end=df['Resolved'][i], freq="B")
                    
                    df['lead_time'][i] = len(range_lead_time)
                    
                    for j in range(0, len(non_working_days), 1):
                        if non_working_days[j] in range_lead_time:
                            df['lead_time'][i] -= 1
                    
                else:
                    df['lead_time'][i] = ''
                        
        df["lead_time"] = pd.to_numeric(df["lead_time"], downcast="float")
        
        #### Unifica colunas com Inward Dependencies:
        cols = []
        for col in df.columns:
            if 'Inward' in col:
                cols.append(col)

        merged_inward_dep = df[cols]
        merged_inward_dep = merged_inward_dep[merged_inward_dep.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype('str')), axis=1)
        merged_inward_dep.tolist()

        df['inward_dep'] = merged_inward_dep

        #### Unifica colunas com Outward Dependencies:
        cols = []
        for col in df.columns:
            if 'Outward' in col:
                cols.append(col)

        merged_outward_dep = df[cols]
        merged_outward_dep = merged_outward_dep[merged_outward_dep.columns[0:]].apply(
            lambda x: ', '.join(x.dropna().astype('str')), axis=1)
        merged_outward_dep.tolist()

        df['outward_dep'] = merged_outward_dep
        
        df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format="mixed")
        df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")
        df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")

        return df

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Import data

    
    # Get last imported date from csv
    csv_filename = "./db/db_ide/last_imported_date.csv"
    last_imported_date_df = pd.read_csv(csv_filename)
    last_imported_date = last_imported_date_df.iloc[0, 0]
    
    
    # Get Historical Data
    csv_filename = "./db/db_ide/df_historical_data_ide.csv"
    csv_filename_bugs = "./db/db_ide/df_historical_bugs_ide.csv"
    
    historical_data_df_ide = pd.read_csv(csv_filename)
    historical_bug_data_df_ide = pd.read_csv(csv_filename_bugs)
    
    # Convert Date Columns to Timestamp
    historical_data_df_ide['Resolved'] = pd.to_datetime(historical_data_df_ide['Resolved'], dayfirst=True, format='mixed')
    historical_data_df_ide['Resolved'] = historical_data_df_ide['Resolved'].dt.strftime('%Y-%m-%d')
    
    historical_data_df_ide['Created'] = pd.to_datetime(historical_data_df_ide['Created'], dayfirst=True, format='mixed')
    historical_data_df_ide['Created'] = historical_data_df_ide['Created'].dt.strftime('%Y-%m-%d')
    
    historical_data_df_ide['Resolved'] = pd.to_datetime(historical_data_df_ide['Resolved'], dayfirst=True, format='mixed')
    historical_data_df_ide['Resolved'] = historical_data_df_ide['Resolved'].dt.strftime('%Y-%m-%d')
    
    historical_data_df_ide['Created'] = pd.to_datetime(historical_data_df_ide['Created'], dayfirst=True, format='mixed')
    historical_data_df_ide['Created'] = historical_data_df_ide['Created'].dt.strftime('%Y-%m-%d')  

    # # Convert column 'sprints' to list again
    historical_data_df_ide['sprints'] = historical_data_df_ide['sprints'].str.replace('[', '').str.replace(']', '')


    # Function to convert the string to a list
    def string_to_list(s):
        if isinstance(s, str):
            return s.split(', ')
        return []
    
    historical_data_df_ide['sprints'] = historical_data_df_ide['sprints'].apply(string_to_list)
    
    # Function to clean up strings in a list
    def clean_list_strings(lst):
        cleaned_strings = []
        for s in lst:
            # Remove unnecessary quotation marks
            cleaned_string = s.replace("'", "").replace('"', "")
            # Remove unnecessary backslashes
            cleaned_string = cleaned_string.replace("\\", "")
            cleaned_strings.append(cleaned_string)
        return cleaned_strings

    historical_data_df_ide['sprints'] = historical_data_df_ide['sprints'].apply(clean_list_strings)
    
    def update_historical_data(historical_data_df_ide, jira_instance):
        # Query just the delta from Jira
        query_ide = f"project = IDEGENAI AND type in ('User Story', Activity, 'Backlog Item', 'Test Execution', 'Test Task', Bug) AND status in (Completed, Obsolete) and updated >= {last_imported_date}"

        df_delta = pd.read_csv(BytesIO(jira_instance.csv(query_ide, all_fields=True, delimiter=',')))
        
        df_delta = prepareData(df_delta, st.session_state['non_working_brazil'])

        df_delta = df_delta[[
        'Priority', 'Issue Type', 'Issue key', 'Issue id', 'Status', 'Created',
        'Resolved', 'planned_effort','time_spent','original_estimate', 
        'sprints', 'started', 'num_sprints', 'hours_spent',
        'days_spent', 'planned_hours', 'Components', 'week_number', 'cycle_time', 
        'lead_time', 'inward_dep', 'outward_dep', 'Assignee', 'Summary',
        ]]
        
        # Append delta to the main dataframe
        df = pd.concat([historical_data_df_ide, df_delta], ignore_index=True)
        
        # Drop rows with duplicated Jira IDs and keep the last occurrence
        df = df[~df['Issue key'].duplicated(keep='last')]
        df = df.reset_index(drop=True)
                    
        # Export updated df to csv - Team Brazil
        df.to_csv('./db/db_ide/df_historical_data_ide.csv', index=False)
        
        return df
    
    def update_bug_historical_data(historical_bug_data_df_ide, jira_instance):
        # Query just the delta from Jira
        query_ide_bug = f"project in (IDEGENAI) AND issuetype in (Bug) and updated >= {last_imported_date}"

        df_delta = pd.read_csv(BytesIO(jira_instance.csv(query_ide_bug, all_fields=True, delimiter=',')))
        
        df_delta = prepareData(df_delta, st.session_state['non_working_brazil'])

        df_delta = df_delta[[
        'Priority', 'Issue Type', 'Issue key', 'Issue id', 'Status', 'Created',
        'Resolved', 
        'Custom field (Parent Link)', 'planned_effort','time_spent','original_estimate', 
        'sprints', 'started', 'Parent', 'num_sprints', 'hours_spent',
        'days_spent', 'planned_hours', 'Components', 'week_number', 'cycle_time', 
        'lead_time', 'inward_dep', 'outward_dep', 'Assignee', 'Summary',
        ]]
        
        # Append delta to the main dataframe
        df = pd.concat([historical_bug_data_df_ide, df_delta], ignore_index=True)
        
        # Drop rows with duplicated Jira IDs and keep the last occurrence
        df = df[~df['Issue key'].duplicated(keep='last')]
        df = df.reset_index(drop=True)
                    
        # Export updated df to csv - Team Brazil
        df.to_csv('./db/db_ide/df_historical_bugs_ide.csv', index=False)
        
        return df 
            
    if date.today().strftime('%Y-%m-%d') >= last_imported_date:
        try:
            df = update_historical_data(historical_data_df_ide, jira_instance)
            df_bugs = update_bug_historical_data(historical_bug_data_df_ide, jira_instance)
            
        except pd.errors.EmptyDataError:
            df = historical_data_df_ide
            df_bugs = historical_bug_data_df_ide
            
    # Update Last Imported Date and save as csv again
    last_imported_date_df.at[0, 'Last Imported Date'] = date.today().strftime('%Y-%m-%d')
    last_imported_date_df.to_csv('./db/db_ide/last_imported_date.csv', index=False)
        
    ############################################################################################################
    ############################################################################################################
    # ANCHOR Subheader layout

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Count of Items Delivered by Sprint

    # Stores in which sprint the item was delivered

    df['sprint_delivered'] = ''

    for i in range(0, len(df), 1):
        if df['Status'][i] in completed_status:
            if df['sprints'][i] != '':
                if df['sprints'][i]:
                    df.at[i, 'sprint_delivered'] = df['sprints'][i][-1]
                else:
                    df.at[i, 'sprint_delivered'] = None

    df_delivered_sprints = pd.DataFrame(df[(df['Status'].isin(completed_status))]['sprint_delivered'].value_counts()).reset_index()

    # Get number of bugs delivered by sprint
    df_bugs_by_sprint = pd.DataFrame(df[((df['Issue Type'] == 'Bug')) & (df['Status'].isin(completed_status))]['sprint_delivered'].value_counts()).reset_index()
    df_bugs_by_sprint.columns = ['Sprint', 'Num_bugs']

    # Get number of items that are not bugs delivered by sprint
    df_bli_by_sprint = pd.DataFrame(df[((df['Issue Type'] != 'Bug')) & (df['Status'].isin(completed_status))]['sprint_delivered'].value_counts()).reset_index()
    df_bli_by_sprint.columns = ['Sprint', 'Num_BLI']
    
    df_delivered_sprints = df_delivered_sprints.rename(columns={'sprint_delivered': 'Sprint'})

    # Merge total deliveries with types of items delivered by sprint
    df_delivered_sprints = df_delivered_sprints.merge(df_bugs_by_sprint, on='Sprint', how='left')
    df_delivered_sprints = df_delivered_sprints.merge(df_bli_by_sprint, on='Sprint', how='left')
    df_delivered_sprints['Num_bugs'] = df_delivered_sprints['Num_bugs'].fillna(0)

    df_delivered_sprints['Bugs_Over_BLI'] = (df_delivered_sprints['Num_bugs'] / df_delivered_sprints['Num_BLI']).round(2) * 100


    ## Sort sprints based on the order they were started
    delivered_by_sprint = df_sprints.loc[df_sprints['name'].isin(df_delivered_sprints['Sprint'])].reset_index(drop=True)
    delivered_by_sprint = delivered_by_sprint.merge(df_delivered_sprints, left_on='name', right_on='Sprint', how='left')
    delivered_by_sprint = delivered_by_sprint.loc[delivered_by_sprint['state'] != 'future']

    # delivered_by_sprint['Sprint_order'] = pd.Categorical(delivered_by_sprint['Sprint'], categories=list_sprints_filter, ordered=True)

    # Plot the chart
    fig_countofitems = px.bar(delivered_by_sprint,
                                x='Sprint',
                                y='count',
                                color='Sprint',
                                title="Velocity by Sprint (Number of Items Delivered)",
                                template='plotly_white', 
                                text_auto=True
                                )

    fig_countofitems.update_layout(showlegend=False)
    
    past_delivery_data = delivered_by_sprint['count'].to_list()

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Bugs X BLI per Sprint

    fig_bugsXitems = px.bar(delivered_by_sprint,
                            x='Sprint',
                            y=['Num_BLI', 'Num_bugs'],
                            template='plotly_white',
                            title='Bugs X Other Items per Sprint',
                            text_auto=True
                            )

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Sum of PDs by Sprint

    delivered_pds = df[(df['Status'].isin(completed_status))].groupby(by='sprint_delivered')['planned_effort'].sum()
    delivered_pds = delivered_pds.reset_index()
    delivered_pds = delivered_pds.rename(columns={'sprint_delivered': 'Sprint'})

    ## Sort sprints based on the order they were started
    delivered_pds_by_sprint = df_sprints.loc[df_sprints['name'].isin(delivered_pds['Sprint'])].reset_index(drop=True)
    delivered_pds_by_sprint = delivered_pds_by_sprint.merge(delivered_pds, left_on='name', right_on='Sprint', how='left')
    delivered_pds_by_sprint = delivered_pds_by_sprint.loc[delivered_pds_by_sprint['state'] != 'future']

    fig_sumpds = px.bar(delivered_pds_by_sprint,
                        x='Sprint',
                        y='planned_effort',
                        color='Sprint',
                        title="Velocity by Sprint (Number of PDs Delivered)",
                        template='plotly_white', 
                        text_auto=True
                        )


    fig_sumpds.update_layout(showlegend=False) 

    past_delivery_data_pds = delivered_pds_by_sprint['planned_effort'].to_list()


    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Median Cycle Time by Sprint
    df_cycle = df[['Issue key', 'Issue Type', 'Status', 'Components', 'cycle_time', 'started',  'Resolved', 'sprint_delivered', 'Assignee', 'planned_effort']]
    
    df_cycle['Resolved'] = pd.to_datetime(df_cycle['Resolved'], dayfirst=True, format='mixed')
    df_cycle['Resolved'] = df_cycle['Resolved'].dt.strftime('%Y-%m-%d')
    
    df_cycle = df_cycle[(df_cycle['Resolved'] >= st.session_state['resolved_date']) & (df['Status'].isin(completed_status))]

    df_cycle = df_cycle.reset_index(drop=True)

    cycle_time_sprint = round(df_cycle.groupby(by='sprint_delivered')['cycle_time'].median(),2).to_frame().reset_index()


    ## Sort sprints based on the order they were started
    cycle_by_sprint = df_sprints.loc[df_sprints['name'].isin(cycle_time_sprint['sprint_delivered'])].reset_index(drop=True)
    cycle_by_sprint = cycle_by_sprint.merge(cycle_time_sprint, left_on='name', right_on='sprint_delivered', how='left')
    cycle_by_sprint = cycle_by_sprint.loc[cycle_by_sprint['state'] != 'future']
        
    fig_mediancycle = px.bar(cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)],
                            x=cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)]['sprint_delivered'],
                            y=cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)]['cycle_time'],
                            color=cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)]['sprint_delivered'],
                            text_auto=True,
                            title="Cycle Time by Sprint (in days) using Median"
                            )

    fig_mediancycle.update_layout(showlegend=False) 


    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Cycle Time Standard Deviation by Sprint

    df_cycle = df_cycle[(df_cycle['Resolved'] >= st.session_state['resolved_date']) & (df_cycle['Status'].isin(completed_status))]

    df_cycle = df_cycle.reset_index(drop=True)

    cycle_time_sprint = round(df_cycle.groupby(by='sprint_delivered')['cycle_time'].std(),2).to_frame().reset_index()

    ## Sort sprints based on the order they were started
    cycle_by_sprint = df_sprints.loc[df_sprints['name'].isin(cycle_time_sprint['sprint_delivered'])].reset_index(drop=True)
    cycle_by_sprint = cycle_by_sprint.merge(cycle_time_sprint, left_on='name', right_on='sprint_delivered', how='left')
    cycle_by_sprint = cycle_by_sprint.loc[cycle_by_sprint['state'] != 'future']
        
    fig_meancyclestd = px.bar(cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)],
                                x=cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)]['sprint_delivered'],
                                y=cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)]['cycle_time'],
                                color=cycle_by_sprint[~cycle_by_sprint.apply(lambda row: 'Wave_8_Sprint_24_Integration' in row.values, axis=1)]['sprint_delivered'],
                                text_auto=True,
                                title="Cycle Time by Sprint (in days) using Standard Deviation"
                                )

    fig_meancyclestd.update_layout(showlegend=False) 

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Cycle Time by Sprint using a Bloxplot

    df_cycle_sprint = df_cycle[(df_cycle['Status'].isin(completed_status))].sort_values(by='sprint_delivered')
    df_cycle_sprint = df_cycle_sprint[df_cycle_sprint['sprint_delivered'] != ''].reset_index(drop=True)

    df_cycle_sprint['sprint_order'] = ''
    
    ## Sort sprints based on the order they were started
    sprint_order = df_sprints.loc[df_sprints['name'].isin(cycle_time_sprint['sprint_delivered'])].reset_index(drop=True)
    sprint_order = sprint_order['name'].to_list()

    for i in range(0, len(df_cycle_sprint)):
        sprint_position = sprint_order.index(df_cycle_sprint['sprint_delivered'][i])
        df_cycle_sprint.at[i, 'sprint_order'] = sprint_position

    df_cycle_sprint = df_cycle_sprint.sort_values(by='sprint_order', ascending=True).reset_index(drop=True)

    fig_box = px.box(df_cycle_sprint, 
                    x='sprint_delivered', 
                    y='cycle_time', 
                    color='sprint_delivered', 
                    title='Cycle Time by sprint using Boxplot', 
                    points='suspectedoutliers'
                    )

    fig_box.update_layout(showlegend=False) 

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Bugs Created by Week (Grouped)

        
    # Assuming df_bugs is your DataFrame
    df_bugs['Created'] = pd.to_datetime(df_bugs['Created'])

    # Extract year and week number from the creation_date
    df_bugs['year'] = df_bugs['Created'].dt.isocalendar().year
    df_bugs['week_number'] = df_bugs['Created'].dt.isocalendar().week

    # Group by both year and week number, and count the number of items
    weekly_counts = df_bugs.groupby(['year', 'week_number']).size().reset_index(name='item_count')

    # Sort by year and week number
    weekly_counts = weekly_counts.sort_values(by=['year', 'week_number'])

    # Create a bar chart
    fig_bar = px.bar(
        weekly_counts, 
        x='week_number', 
        y='item_count', 
        color='year', 
        title='Bugs Created by Week and Year',
        labels={'week_number': 'Week Number', 'item_count': 'Number of Bugs', 'year': 'Year'},
        text_auto=True
    )

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Historical Sprint Analysis - Organizing Plots in Grid

    st.write('## Historical Sprint Analysis')

    col1, col2 = st.columns(2)

    with col1:
        
        st.plotly_chart(fig_countofitems)
        
        st.plotly_chart(fig_mediancycle)
        
        st.plotly_chart(fig_box)
        
        st.plotly_chart(fig_bar)

    with col2:
        
        st.plotly_chart(fig_sumpds)
        
        st.plotly_chart(fig_meancyclestd)
        
        st.plotly_chart(fig_bugsXitems)

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Cycle Time by Issue Type over time
    
    st.write("""--------------------""")
    st.header("Cycle Time by issue type over time")
    
    df_cycle_issue_type = pd.DataFrame()
    df_cycle_issue_type = df[['Issue key', 'Issue Type', 'started', 'Resolved']]
    
    resolved_date = st.session_state['resolved_date']
    resolved_date = pd.to_datetime(resolved_date, dayfirst=True, format='mixed')
    
    df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format='mixed')
    
    mean = df[(df['Resolved'] >= st.session_state['resolved_date'])]['cycle_time'].mean()
    median = df[(df['Resolved'] >= st.session_state['resolved_date'])]['cycle_time'].median()
    q80 = df[(df['Resolved'] >= st.session_state['resolved_date'])]['cycle_time'].quantile(0.80)

    st.write('### Project Average Cycle Time = {} days | 50% of deliveries happened in {} days or less | 80% of deliveries happened in {} days or less'.format(round(mean,0), median, q80))
    
    today = datetime.date.today()
    one_month_ago = today - datetime.timedelta(days=1*30)
    three_months_ago = today - datetime.timedelta(days=3*30)
    six_months_ago = today - datetime.timedelta(days=6*30)
    twelve_months_ago = today - datetime.timedelta(days=12*30)

    beginning_of_project = '2022-01-01'
    today = today.strftime("%Y-%m-%d")
    one_month_ago = one_month_ago.strftime("%Y-%m-%d")
    three_months_ago = three_months_ago.strftime("%Y-%m-%d")
    six_months_ago = six_months_ago.strftime("%Y-%m-%d")
    twelve_months_ago = twelve_months_ago.strftime("%Y-%m-%d")
    
    dates = [beginning_of_project, twelve_months_ago, six_months_ago, three_months_ago, one_month_ago]
    
    issue_types = ['Backlog Item', 'Bug', 'Activity', 'User Story', 'Test Execution']
    
    issue_type_dfs = {
        'User Story': pd.DataFrame(columns=dates, index=['Issue Type','Mean Planned Effort', 'Mean Cycle Time', 'Std Dev', 'Median (Q50)', 'Quartile 60', 'Quartile 70', 'Quartile 80', 'Quartile 90', 'Maximum']),
        'Test Execution': pd.DataFrame(columns=dates, index=['Issue Type','Mean Planned Effort', 'Mean Cycle Time', 'Std Dev', 'Median (Q50)', 'Quartile 60', 'Quartile 70', 'Quartile 80', 'Quartile 90', 'Maximum']),
        'Backlog Item': pd.DataFrame(columns=dates, index=['Issue Type','Mean Planned Effort', 'Mean Cycle Time', 'Std Dev', 'Median (Q50)', 'Quartile 60', 'Quartile 70', 'Quartile 80', 'Quartile 90', 'Maximum']),
        'Bug': pd.DataFrame(columns=dates, index=['Issue Type', 'Mean Planned Effort', 'Mean Cycle Time', 'Std Dev', 'Median (Q50)', 'Quartile 60', 'Quartile 70', 'Quartile 80', 'Quartile 90', 'Maximum']),
        'Activity': pd.DataFrame(columns=dates, index=['Issue Type', 'Mean Planned Effort', 'Mean Cycle Time', 'Std Dev', 'Median (Q50)', 'Quartile 60', 'Quartile 70', 'Quartile 80', 'Quartile 90', 'Maximum'])
    }

    for issue_type in issue_types:
        for date in dates:
            df_base_cycle_time = df[
                (df['Resolved'] >= date) & 
                (df['Issue Type'].str.lower() == issue_type.lower()) &  
                (df['cycle_time'].notna()) & 
                (df['Status'] != 'Obsolete')
            ][['cycle_time', 'Issue Type', 'planned_effort']]
            
            planned_effort_mean = round(df_base_cycle_time['planned_effort'].mean(),4)
            cycle_time_mean = round(df_base_cycle_time['cycle_time'].mean(),4)
            cycle_time_std = round(df_base_cycle_time['cycle_time'].std(),4)
            cycle_time_median = round(df_base_cycle_time['cycle_time'].median(),4)
            cycle_time_q60 = round(df_base_cycle_time['cycle_time'].quantile(0.60),4)
            cycle_time_q70 = round(df_base_cycle_time['cycle_time'].quantile(0.70),4)
            cycle_time_q80 = round(df_base_cycle_time['cycle_time'].quantile(0.80),4)
            cycle_time_q90 = round(df_base_cycle_time['cycle_time'].quantile(0.90),4)
            cycle_time_max = round(df_base_cycle_time['cycle_time'].max(),4)
            
            df_results = issue_type_dfs[issue_type]
            df_results.at['Issue Type', date] = issue_type                
            df_results.at['Mean Planned Effort', date] = planned_effort_mean
            df_results.at['Mean Cycle Time', date] = cycle_time_mean
            df_results.at['Std Dev', date] = cycle_time_std
            df_results.at['Median (Q50)', date] = cycle_time_median
            df_results.at['Quartile 60', date] = cycle_time_q60
            df_results.at['Quartile 70', date] = cycle_time_q70
            df_results.at['Quartile 80', date] = cycle_time_q80
            df_results.at['Quartile 90', date] = cycle_time_q90
            df_results.at['Maximum', date] = cycle_time_max
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_story = issue_type_dfs['User Story']
        st.dataframe(user_story)
        
        test_execution = issue_type_dfs['Test Execution']
        st.dataframe(test_execution)
    
    with col2:
        backlog_item = issue_type_dfs['Backlog Item']
        st.dataframe(backlog_item)
        
        activities = issue_type_dfs['Activity']
        st.dataframe(activities)

    with col3:
        
        bugs = issue_type_dfs['Bug']
        st.dataframe(bugs)