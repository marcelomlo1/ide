# ANCHOR Parameters - Import Packages
from pyrsistent import v
import streamlit as st
import warnings
warnings.filterwarnings('ignore')
from datetime import date, datetime
import pandas as pd
import numpy as np
import plotly.express as px
from io import BytesIO
import plotly.graph_objects as go
from atlassian import Jira
import json
import requests

############################################################################################################
############################################################################################################
### ANCHOR Parameters - Instantiate Jira


jira_instance = Jira(
    url = st.secrets["db_url"],
    username = st.secrets["db_username"],
    password = st.secrets["db_password"]
)

############################################################################################################
############################################################################################################
### ANCHOR Main Function

def main():


    ############################################################################################################
    ############################################################################################################
    ### ANCHOR Parameters - Set the width of the page

    st.set_page_config(layout="wide", page_title="Our Current Metrics", page_icon="🌱", initial_sidebar_state='auto')


    ############################################################################################################
    ############################################################################################################
    # ANCHOR Parameters - Global Layout Configuration

    # Hide 'Made with Streamlit'

    hide_streamlit_style = """
                <style>
                # MainMenu {visibility: hidden;}
                # header {visibility: hidden;}
                footer {visibility: hidden;}
                viewerBadge_container__1QSob {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    # What day is today?
    today = datetime.now()
    today = today.strftime("%d-%m-%Y")


    if st.button("Clear cache and rerun"):

        # Clear all memo cache:
        st.experimental_memo.clear()
        st.cache_resource.clear()
        st.experimental_rerun()

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Home Page

    st.markdown('#')
    st.write("""

    ### Current Sprint Metrics Dashboard of Intelligent Deal Entry! 🤖🌱🛢️

    """)
 
    ############################################################################################################
    ############################################################################################################
    ### ANCHOR Parameters - Session State (before main function)

    if 'resolved_date' not in st.session_state:
        st.session_state['resolved_date'] = '2024-09-09'
        
    if 'non_working_brazil' not in st.session_state:
        st.session_state['non_working_brazil'] = ['2024-01-01', '2024-02-12', '2024-02-13', 
                                                '2024-02-14', '2024-03-29', '2024-04-21', '2024-05-01', '2024-07-25', 
                                                '2024-09-07', '2024-09-20', '2024-10-12', '2024-11-02', '2024-11-15', 
                                                '2024-11-20', '2024-12-08', '2024-12-25',]
        
    ############################################################################################################
    ############################################################################################################
    # ANCHOR Get All Sprints from Boards

    with st.spinner("Getting all sprints..."):
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
    # ANCHOR Get Active Sprints
    with st.spinner("Getting Active Sprints..."):
        if len(df_sprints[(df_sprints['state'] == 'active')]) == 0:
            active_sprints = df_sprints[(df_sprints['state'] == 'closed')]
            active_sprints = active_sprints.iloc[[-1]]
            active_sprints = active_sprints.reset_index(drop=True)
            
            update_query = f"({active_sprints['name'].values[0]})"
            
            
        else: 
            active_sprints = df_sprints[(df_sprints['state'] == 'active')]
            active_sprints = active_sprints.reset_index(drop=True)
            
            update_query = 'openSprints()'

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Import Backlog Items (Current Sprint and Completed Items)
    
    def prepareData(df, non_working_days):
        ######## Get the first log for 'Resolved Date' and first log for 'Started Date'
        ### Creates a Data Frame to store the Issue Keys

        df_change_log = pd.DataFrame(columns=['Issue key'])
        df_change_log['Issue key'] = df['Issue key']
        df_change_log = df_change_log.reset_index(drop=True)
        df_resolved_transition_log = pd.DataFrame(columns=[
            'field', 'fieldtype', 'from', 'fromString', 'to', 'toString', 'created'
        ])

        df_transition_log = pd.DataFrame(columns=[
            'field', 'fieldtype', 'from', 'fromString', 'to', 'toString', 'created'
        ])
        
        # New DataFrame for tracking blockage times
        df_blocked_log = pd.DataFrame(columns=['issue_key', 'blocked_at', 'unblocked_at'])


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

                # Identify and sort blocked and unblocked entries
                blocked_entries = df_get_change_log.loc[(df_get_change_log['toString'] == 'Impediment')].sort_values(by='created')
                unblocked_entries = df_get_change_log.loc[(df_get_change_log['fromString'] == 'Impediment')].sort_values(by='created')

                for _, block_entry in blocked_entries.iterrows():
                    issue_key = block_entry['issue_key']
                    blocked_at = block_entry['created']

                    # Find the nearest following unblock event
                    next_unblock = unblocked_entries[(unblocked_entries['issue_key'] == issue_key) & (unblocked_entries['created'] > blocked_at)]

                    if not next_unblock.empty:
                        unblocked_at = next_unblock.iloc[0]['created']
                        # Remove the matched unblock event to prevent reuse
                        unblocked_entries = unblocked_entries.drop(next_unblock.index[0])
                    else:
                        unblocked_at = None

                    # Create a new DataFrame row
                    new_row = pd.DataFrame({'issue_key': [issue_key], 'blocked_at': [blocked_at], 'unblocked_at': [unblocked_at]})
                    df_blocked_log = pd.concat([df_blocked_log, new_row], ignore_index=True)

        if not df_resolved_transition_log.empty:
            df_resolved_transition_log['created'] = df_resolved_transition_log['created'].str.split("T").str[0]
            df_resolved_transition_log = df_resolved_transition_log[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='first').reset_index(drop=True)
            df_resolved_transition_log['created'] = pd.to_datetime(df_resolved_transition_log['created'], dayfirst=True, format="mixed")
            df_resolved_transition_log = df_resolved_transition_log.rename(columns={'issue_key': 'Issue key', 'created': 'Resolved'})
            df = df.merge(df_resolved_transition_log, how='left', on='Issue key')

        if not df_transition_log.empty:
            df_transition_log['created'] = df_transition_log['created'].str.split("T").str[0]
            df_transition_log = df_transition_log[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='first').reset_index(drop=True)
            df_transition_log['created'] = pd.to_datetime(df_transition_log['created'], dayfirst=True, format='mixed')
            df_transition_log = df_transition_log.rename(columns={'issue_key': 'Issue key', 'created': 'started'})
            df = df.merge(df_transition_log, how='left', on='Issue key')
        
        df_blocked_log['blocked_at'] = pd.to_datetime(df_blocked_log['blocked_at'], dayfirst=True, format='mixed')
        df_blocked_log['unblocked_at'] = pd.to_datetime(df_blocked_log['unblocked_at'], dayfirst=True, format='mixed')
        df_blocked_log['blocked_duration'] = 0
        
        for i in range(0,len(df_blocked_log), 1):
            if pd.notna(df_blocked_log['unblocked_at'][i]):
                range_blocked_time = pd.bdate_range(start=df_blocked_log['blocked_at'][i], end=df_blocked_log['unblocked_at'][i], freq="B")
                df_blocked_log.at[i, 'blocked_duration'] = len(range_blocked_time)
        
        df_blocked_log = df_blocked_log.rename(columns={'issue_key': 'Issue key'})
        grouped_df = df_blocked_log.groupby('Issue key')['blocked_duration'].sum().reset_index()
        df = df.merge(grouped_df, how='left', on='Issue key')


        #### Dates preparation
        df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format='mixed')

        if 'Resolved' in df:
            df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")

        if 'started' in df:
            df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")

        df['Created'] = df['Created'].dt.strftime('%Y-%m-%d')

        if 'Resolved' in df:
            df['Resolved'] = df['Resolved'].dt.strftime('%Y-%m-%d')

        if 'started' in df:
            df['started'] = df['started'].dt.strftime('%Y-%m-%d')

        df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format="mixed")

        if 'Resolved' in df:
            df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")

        if 'started' in df:
            df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")


        #### Cria coluna para identificar o número da semana que a entrega aconteceu
        df['week_number'] = df['Resolved'].dt.isocalendar().week

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
                            # Check if cycle_time is greater than 1 before decrementing otherwise, it will be 0 and will show wrong values in the charts later
                            if df['cycle_time'][i] > 1:
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
    # ANCHOR Import data from Current Sprint
    with st.spinner("Importing Backlog Items..."):
        query_ide = f"project = IDEGENAI AND type in ('User Story', Activity, 'Backlog Item', 'Test Execution', 'Test Task', Bug) AND sprint in {update_query} ORDER BY resolved ASC"

        try:
            df = pd.read_csv(BytesIO(jira_instance.csv(query_ide, all_fields=True, delimiter=',')))
            df = prepareData(df, st.session_state['non_working_brazil'])
            try:
                df = df[[
                'Priority', 'Issue Type', 'Issue key', 'Issue id', 'Status', 'Created',
                'Resolved', 'planned_effort','time_spent', 
                'sprints', 'started', 'num_sprints', 'hours_spent',
                'days_spent', 'planned_hours', 'Components', 'week_number', 'cycle_time', 
                'lead_time', 'inward_dep', 'outward_dep', 'Custom field (Flagged)', 'Summary', 
                'blocked_duration', 'Assignee'
                ]]
                
            except KeyError:
                
                df = df[[
                'Priority', 'Issue Type', 'Issue key', 'Issue id', 'Status', 'Created',
                'Resolved', 'planned_effort','time_spent', 
                'sprints', 'started', 'num_sprints', 'hours_spent',
                'days_spent', 'planned_hours', 'Components', 'week_number', 'cycle_time', 
                'lead_time', 'inward_dep', 'outward_dep', 'Custom field (Flagged)', 'Summary', 
                'blocked_duration', 'Assignee'
                ]]
            
                
            df_current_sprint = df
            df_time_blocked = df[['Issue key', 'Issue Type', 'blocked_duration']]
            
            # Create a Data Frame to analyze Planned Effort X Cycle Time later
            df_perf_analy = df_current_sprint
            
            
            df_current_sprint['planned'] = ''

            # Exclude items closed before the sprint was started (during refinement we can cancel or complete something that was already done). This prevents errors in the Burndown
            df_current_sprint = df_current_sprint.loc[(df_current_sprint['Resolved'].isna()) | (df_current_sprint['Resolved'] >= active_sprints['startDate'][0])].reset_index(drop=True)
        
        except pd.errors.EmptyDataError:
            st.warning("There is no active sprint at this moment, but it is possible to analyze our **historical data** by selecting this option on the menu.", icon="⚠")

        ############################################################################################################
        ############################################################################################################
        # ANCHOR Remove Non-Working Days from the analysis

        if len(active_sprints) > 0:
            original_range_sprint = pd.bdate_range(
                start= active_sprints['startDate'][0],
                end= active_sprints['endDate'][0],
                freq="B")
            
            cont_nonworking = 0
            for i in range(0, len(st.session_state['non_working_brazil']), 1):
                if st.session_state['non_working_brazil'][i] in original_range_sprint:
                    cont_nonworking += 1

                
            # Remove non_working days from the dataframes
            non_working_brazil_datetime = [pd.to_datetime(date, format="%Y-%m-%d") for date in st.session_state['non_working_brazil']]
            index_to_delete = []
            for i in range(0, len(original_range_sprint), 1):
                    if original_range_sprint[i] in non_working_brazil_datetime:
                            index_to_delete.append(i)

            range_sprint = original_range_sprint.delete(index_to_delete)

        else:
            pass


    ############################################################################################################
    ############################################################################################################
    # ANCHOR Subheader Constructor

    my_date = date.today()
    year, week_number, day_of_week = my_date.isocalendar()

    # Deliveries of current week
    this_week_done = df[(df['week_number'] == week_number) & 
        (df['Resolved'] >= st.session_state['resolved_date']) & 
        (df['Status'] == 'Completed')].groupby(by='week_number')['Resolved'].value_counts().sum()

    ############################################################################################################
    ############################################################################################################
    # ANCHOR last sprint disclaimer
    
    if active_sprints['state'][0] == 'closed':
        st.write(f"## You are seeing data from our last sprint {active_sprints['name'][0]} since there is no active sprint at the moment.")
        
    ############################################################################################################
    ############################################################################################################
    # ANCHOR Subheader layout

    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

    st.write("""
            **Flow Metrics.**
            """)
    
    
    col1, col2, col3, col4, col5 = st.columns(5)

    completed_status = ['Completed', 'Obsolete']

    with col1:
        this_sprint_done_pds = df_current_sprint[df_current_sprint['Status'].isin(completed_status)]['planned_effort'].agg('sum')
        
        st.write("#### PDs completed/obsoleted")
        st.metric(label= "", value = "{} PDs".format(int(this_sprint_done_pds)))

    with col2:
        total_sprint_pds = df_current_sprint['planned_effort'].sum()
        
        st.write("#### Total PDs in Sprint")
        st.metric(label= "", value = "{} PDs".format(int(total_sprint_pds)))               

            
    with col3:
        
        this_sprint_done = df_current_sprint[df_current_sprint['Status'].isin(completed_status)]['Status'].agg('count')
        
        st.write("#### Items completed/obsoleted")
        st.metric(label= "", value = "{} items".format(int(this_sprint_done)))

    with col4:
        
        st.write("#### Num. of Items in Sprint")
        st.metric(label= "", value = "{} items".format(int(len(df_current_sprint))))

    with col5:
        if len(df_current_sprint[df_current_sprint['Status'].isin(completed_status)]) > 0:
            current_sprint_cycle_time = df_current_sprint[(df_current_sprint['Resolved'] >= st.session_state['resolved_date']) &
                                                                (df_current_sprint['Status'].isin(completed_status))]['cycle_time'].mean()
            
            st.write("#### Cycle time in current sprint")
            st.metric(label= "",
                value = f"{current_sprint_cycle_time:.2f} days"
                )
        else:
            current_sprint_cycle_time = 'No items delivered'
            st.write("#### Cycle time in current sprint")
            st.metric(label= "",
                value = current_sprint_cycle_time
                )    

    st.write("""--------------------""")

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Burndown Chart - Items
    with st.spinner("Creating Burndown Chart for Items..."):
        def burndownChartItems(df_current_sprint):
            # Get from Jira when the item of the current sprint was added to the sprint.

            df_added_after = pd.DataFrame()
            considered_items = []
            for i in range(0, len(df_current_sprint), 1):
                get_sprint_change_log = jira_instance.get_issue_changelog(df_current_sprint['Issue key'][i])
                df_get_sprint_change_log = pd.json_normalize(get_sprint_change_log['histories'], 'items','created')

                df_get_sprint_change_log['Issue key'] = ''

                # Check the row where it states when the item was added and then split the date to match project standards
                if len(df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)) > 0:
                    
                    validate_date_added_sprint = df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)
                    validate_date_added_sprint = validate_date_added_sprint.reset_index(drop = True)
                    validate_date_added_sprint['created'][0] = validate_date_added_sprint['created'][0].split("T")[0]
                    validate_date_added_sprint['created'] = pd.to_datetime(validate_date_added_sprint['created'])

                    
                    if validate_date_added_sprint['created'][0] > active_sprints['startDate'][0]:
                        date_added_after_to_sprint = df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)
                        df_added_after = pd.concat([df_added_after, date_added_after_to_sprint], ignore_index=True)
                        df_added_after['Issue key'][len(df_added_after)-1] = df_current_sprint['Issue key'][i]
                        df_current_sprint.at[i, 'planned'] = '0'

                    else:
                        
                        date_added_to_sprint = df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)
                        date_added_to_sprint = date_added_to_sprint.reset_index(drop = True)
                        
                        if len(date_added_to_sprint['created']) > 0:
                    
                            date_added_to_sprint['created'][0] = date_added_to_sprint['created'][0].split("T")[0]
                            date_added_to_sprint['created'] = pd.to_datetime(date_added_to_sprint['created'])
                            date_added_to_sprint['Issue key'][len(date_added_to_sprint)-1] = df_current_sprint['Issue key'][i]
                            
                            # Check if the date is less or equal to the current sprint start date. If yes, populates df_current_sprint['planned'] with 1
                            if date_added_to_sprint['created'][0] <= active_sprints['startDate'][0]:
                                
                                df_current_sprint.at[i, 'planned'] = '1'

                            else:
                                df_current_sprint.at[i, 'planned'] = '0'
                        
                else:
                    date_added_to_sprint = df_get_sprint_change_log.head(1)
                    date_added_to_sprint = date_added_to_sprint.reset_index(drop = True)

                    if len(date_added_to_sprint['created']) > 0:
                        
                        date_added_to_sprint['created'][0] = date_added_to_sprint['created'][0].split("T")[0]
                        date_added_to_sprint['created'] = pd.to_datetime(date_added_to_sprint['created'])
                        date_added_to_sprint['Issue key'][len(date_added_to_sprint)-1] = df_current_sprint['Issue key'][i]
                    
                        # Check if the date is less or equal to the current sprint start date. If yes, populates df_current_sprint['planned'] with 1
                        if date_added_to_sprint['created'][0] <= active_sprints['startDate'][0]:
                            
                            df_current_sprint.at[i, 'planned'] = '1'

                        else:
                            df_current_sprint.at[i, 'planned'] = '0'
                            df_added_after = pd.concat([df_added_after, date_added_to_sprint], ignore_index=True)
                            df_added_after['Issue key'][len(df_added_after)-1] = df_current_sprint['Issue key'][i]

            # Create data frame for the ideal burndown line

            df_ideal_burndown = pd.DataFrame(columns=['dates', 'ideal_trend'])
            df_ideal_burndown['dates'] = range_sprint


            #### Dates preparation
            df_ideal_burndown['dates'] = pd.to_datetime(df_ideal_burndown['dates'], dayfirst=True, format="mixed")
            df_ideal_burndown['dates'] = df_ideal_burndown['dates'].dt.strftime('%Y-%m-%d')
            df_ideal_burndown['dates'] = pd.to_datetime(df_ideal_burndown['dates'], dayfirst=True, format="mixed")

            # Define the sprint lenght

            days_sprint = len(range_sprint)

            # Get how many items are in the current sprint
            commited = len(df_current_sprint[df_current_sprint['planned'] == '1'])

            # Define the ideal number of items should be delivered by day
            ideal_burn = round(commited/days_sprint,2)

            # Create a list of remaining items to be delivered by day
            burndown = [commited - ideal_burn]


            # Day of the sprint -> starts with 2, since the first day is already in the list above
            sprint_day = 2

            # Iterate to create the ideal trend line in numbers
            for i in range(1, len(df_ideal_burndown), 1):

                burndown.append(round((commited - (ideal_burn * sprint_day)),0))
                
                sprint_day += 1
                
                
            # Add the ideal burndown to the column
            df_ideal_burndown['ideal_trend'] = burndown


            # Adjust the data frame to get the number of items added after the Sprint Planning and add to the Delivered Effort Data Frame
            if len(df_added_after) > 0:
                # df_added_after['created'] = df_added_after['created'].str.split("T").str[0]
                df_added_after['created'] = pd.to_datetime(df_added_after['created'], dayfirst=True, format='mixed')
                df_added_after['created'] = df_added_after['created'].dt.strftime('%Y-%m-%d')


                df_added_after_final = df_added_after.groupby(by='created')['field'].count().to_frame()
                df_added_after_final.index.names = ['dates']
                df_added_after_final = df_added_after_final.reset_index()
                df_added_after_final.columns = ['dates', 'items_added']

                # Dates preparation
                df_added_after_final['dates'] = pd.to_datetime(df_added_after_final['dates'], dayfirst=True, format='mixed')
                df_added_after_final['dates'] = df_added_after_final['dates'].dt.strftime('%Y-%m-%d')
                df_added_after_final['dates'] = pd.to_datetime(df_added_after_final['dates'], dayfirst=True, format='mixed')


                df_added_after_final = df_added_after_final.loc[df_added_after_final['dates'] > df_sprints['startDate'][0]]

            else:
                build_index = []
                added_items = []
                
                for i in range(0, len(range_sprint)):
                    # build_index.append(i)

                    number_to_add = 0
                    added_items.append(number_to_add)
                    
                
                # df_added_after.index = build_index
                df_added_after['dates'] = pd.Series(range_sprint)
                df_added_after['items_added'] = pd.Series(added_items)
                
                df_added_after_final = df_added_after

            # Create a second data frame to get the number of items delivered each day
            df_delivered_effort = pd.DataFrame(columns=['dates'])
            df_delivered_effort['dates'] = original_range_sprint


            #### Dates preparation
            df_delivered_effort['dates'] = pd.to_datetime(df_delivered_effort['dates'], dayfirst=True, format="mixed")
            df_delivered_effort['dates'] = df_delivered_effort['dates'].dt.strftime('%Y-%m-%d')
            df_delivered_effort['dates'] = pd.to_datetime(df_delivered_effort['dates'], dayfirst=True, format="mixed")

            # ### Getting information about delivered items in the current sprint

            delivered_effort = df_current_sprint.loc[df_current_sprint['Resolved'] >= active_sprints['startDate'][0]].groupby(by='Resolved')['Issue key'].count()

            delivered_effort = delivered_effort.reset_index()

            delivered_effort.columns = ['dates', 'delivered_items']

            #### Dates preparation
            delivered_effort['dates'] = pd.to_datetime(delivered_effort['dates'], dayfirst=True, format="mixed")
            delivered_effort['dates'] = delivered_effort['dates'].dt.strftime('%Y-%m-%d')
            delivered_effort['dates'] = pd.to_datetime(delivered_effort['dates'], dayfirst=True, format="mixed")


            # # ### Merge with the main data frame
            df_delivered_effort = df_delivered_effort.merge(delivered_effort, how='left', on='dates')
            df_delivered_effort = df_delivered_effort.merge(df_added_after_final, how='left', on='dates')


            # ### Create a column to store how many items still need to be delivered
            df_delivered_effort['remaining_items'] = np.nan


            ### Fill NaN with 0 to allow calculation based on dates until today to have a nicer Burndownchart in the final process
            for i in range(0, len(df_delivered_effort), 1):
                if i == 0:
                    df_delivered_effort['delivered_items'][i] = np.nan_to_num(df_delivered_effort['delivered_items'][i])
                    df_delivered_effort['items_added'][i] = np.nan_to_num(df_delivered_effort['items_added'][i])
                    df_delivered_effort['remaining_items'][i] = commited - df_delivered_effort['delivered_items'][i]
                    
                else:
                    if df_delivered_effort['dates'][i] < pd.to_datetime(today, dayfirst=True):
                        df_delivered_effort['delivered_items'][i] = np.nan_to_num(df_delivered_effort['delivered_items'][i])
                        df_delivered_effort['items_added'][i] = np.nan_to_num(df_delivered_effort['items_added'][i])
                        
                        df_delivered_effort.at[i,'remaining_items'] = df_delivered_effort['remaining_items'][i-1] - df_delivered_effort['delivered_items'][i] + df_delivered_effort['items_added'][i]

                    elif df_delivered_effort['dates'][i] == pd.to_datetime(today, dayfirst=True):
                        cur_sprint_open = len(df_current_sprint[df_current_sprint['Status'].isin(['Completed', 'Obsolete']) == False])
                        df_delivered_effort['items_added'][i] = np.nan_to_num(df_delivered_effort['items_added'][i])
                        df_delivered_effort.at[i, 'remaining_items'] = cur_sprint_open

            df_burndown = df_delivered_effort.merge(df_ideal_burndown, how='left', on='dates')
            df_burndown = df_burndown[df_burndown.dates.isin(range_sprint)]
            
            return df_burndown

        df_burndown = burndownChartItems(df_current_sprint)

        # Plot the Burndown Chart        
        st.write("## Burndown Chart - Items")
        
        current_remaining = len(df_current_sprint.loc[~df_current_sprint['Status'].isin(['Completed', 'Obsolete'])])
        idx_today = int(df_burndown['remaining_items'].count())
        days_left = len(df_burndown['dates'][idx_today:])
        
        if days_left == 0:
            days_left = 1
        
        required_pace = round(current_remaining / days_left, 2)
        st.write(f"To accomplish the plan, we need to deliver **{required_pace}** items per day.")
        
        if len(df_burndown) > 0:

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_burndown['dates'], 
                    y=df_burndown['ideal_trend'], 
                    name='Ideal Trend',
                    line=dict(color="Black")
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_burndown['dates'], 
                    y=df_burndown['remaining_items'], 
                    name = 'Remaining Items',
                    fill='tozeroy',
                    line=dict(color="MediumSeaGreen")

                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_burndown['dates'], 
                    y=df_burndown['items_added'], 
                    name = 'Added Items',
                    fill='tozeroy',
                    line=dict(color="Brown")
                )
            )
            
            fig.update_layout(
                        width=1000, 
                        height=500
                        )
            
            fig.update_layout(legend=dict(orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                        legend_title_text='')


            st.plotly_chart(fig) 

        else:
            st.write("### There is no Active Sprint at this moment.")
                        
    ############################################################################################################
    ############################################################################################################
    # ANCHOR Burndown Chart - PDs
    with st.spinner("Creating Burndown Chart for PDs..."):
        def burndownChartPDS(df_current_sprint):
            
            # Get from Jira when the item of the current sprint was added to the sprint.

            df_added_after = pd.DataFrame()

            for i in range(0, len(df_current_sprint), 1):
                get_sprint_change_log = jira_instance.get_issue_changelog(df_current_sprint['Issue key'][i])

                df_get_sprint_change_log = pd.json_normalize(get_sprint_change_log['histories'], 'items','created')

                df_get_sprint_change_log['Issue key'] = ''

                # Check the row where it states when the item was added and then split the date to match project standards
                
                if len(df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)) > 0:
                    
                    validate_date_added_sprint = df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)
                    validate_date_added_sprint = validate_date_added_sprint.reset_index(drop = True)
                    validate_date_added_sprint['created'][0] = validate_date_added_sprint['created'][0].split("T")[0]
                    validate_date_added_sprint['created'] = pd.to_datetime(validate_date_added_sprint['created'])

                    
                    if validate_date_added_sprint['created'][0] > active_sprints['startDate'][0]:
                        
                        date_added_after_to_sprint = df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)
                        df_added_after = pd.concat([df_added_after, date_added_after_to_sprint], ignore_index=True)
                        df_added_after['Issue key'][len(df_added_after)-1] = df_current_sprint['Issue key'][i]
                        df_current_sprint.at[i, 'planned'] = '0'
                        
                        continue
                            
                    else:
                        
                        date_added_to_sprint = df_get_sprint_change_log.loc[df_get_sprint_change_log['toString'].astype(str).str.contains(active_sprints['name'][0])].tail(1)
                        date_added_to_sprint = date_added_to_sprint.reset_index(drop = True)
                        
                        if len(date_added_to_sprint['created']) > 0:
                    
                            date_added_to_sprint['created'][0] = date_added_to_sprint['created'][0].split("T")[0]
                            date_added_to_sprint['created'] = pd.to_datetime(date_added_to_sprint['created'])
                            date_added_to_sprint['Issue key'][len(date_added_to_sprint)-1] = df_current_sprint['Issue key'][i]
                            
                            # Check if the date is less or equal to the current sprint start date. If yes, populates df_current_sprint['planned'] with 1
                            if date_added_to_sprint['created'][0] <= active_sprints['startDate'][0]:
                                
                                df_current_sprint.at[i, 'planned'] = '1'

                            else:
                                df_current_sprint.at[i, 'planned'] = '0'
                        
                else:
                    date_added_to_sprint = df_get_sprint_change_log.head(1)
                    date_added_to_sprint = date_added_to_sprint.reset_index(drop = True)

                    if len(date_added_to_sprint['created']) > 0:
                        
                        date_added_to_sprint['created'][0] = date_added_to_sprint['created'][0].split("T")[0]
                        date_added_to_sprint['created'] = pd.to_datetime(date_added_to_sprint['created'])
                        date_added_to_sprint['Issue key'][len(date_added_to_sprint)-1] = df_current_sprint['Issue key'][i]
                    
                        # Check if the date is less or equal to the current sprint start date. If yes, populates df_current_sprint['planned'] with 1
                        if date_added_to_sprint['created'][0] <= active_sprints['startDate'][0]:
                            
                            df_current_sprint.at[i, 'planned'] = '1'

                        else:
                            df_current_sprint.at[i, 'planned'] = '0'
                            df_added_after = pd.concat([df_added_after, date_added_to_sprint], ignore_index=True)
                            df_added_after['Issue key'][len(df_added_after)-1] = df_current_sprint['Issue key'][i]

            
            # Create data frame for the ideal burndown line
            df_ideal_burndown = pd.DataFrame(columns=['dates', 'ideal_trend'])
            df_ideal_burndown['dates'] = range_sprint


            #### Dates preparation
            df_ideal_burndown['dates'] = pd.to_datetime(df_ideal_burndown['dates'], dayfirst=True, format='mixed')
            df_ideal_burndown['dates'] = df_ideal_burndown['dates'].dt.strftime('%Y-%m-%d')
            df_ideal_burndown['dates'] = pd.to_datetime(df_ideal_burndown['dates'], dayfirst=True, format='mixed')

            # Define the sprint lenght

            days_sprint = len(range_sprint)

            # Get how many items are in the current sprint
            commited = df_current_sprint[df_current_sprint['planned'] == '1']['planned_effort'].sum()

            # Define the ideal number of items should be delivered by day
            ideal_burn = round(commited/days_sprint,2)

            # Create a list of remaining items to be delivered by day
            burndown = [commited - ideal_burn]


            # Day of the sprint -> starts with 2, since the first day is already in the list above
            sprint_day = 2

            # Iterate to create the ideal trend line in numbers
            for i in range(1, len(df_ideal_burndown), 1):

                burndown.append(round((commited - (ideal_burn * sprint_day)),0))
                
                sprint_day += 1
                
                
            # Add the ideal burndown to the column
            df_ideal_burndown['ideal_trend'] = burndown
            
            
            # Adjust the data frame to get the number of items added after the Sprint Planning and add to the Delivered Effort Data Frame
            if len(df_added_after) > 0:
                items_added_after = df_added_after['Issue key'].to_list()
                df_added_after['pds_added_after'] = df_current_sprint.loc[df_current_sprint['Issue key'].isin(items_added_after)]['planned_effort'].to_list()
                
                # df_added_after['created'] = df_added_after['created'].str.split("T").str[0]
                df_added_after['created'] = pd.to_datetime(df_added_after['created'], dayfirst=True, format='mixed')
                df_added_after['created'] = df_added_after['created'].dt.strftime('%Y-%m-%d')

                
                df_added_after_final = df_added_after.groupby(by='created')['pds_added_after'].sum().to_frame()
                df_added_after_final.index.names = ['dates']
                df_added_after_final = df_added_after_final.reset_index()
                df_added_after_final.columns = ['dates', 'pds_added']

                # Dates preparation
                df_added_after_final['dates'] = pd.to_datetime(df_added_after_final['dates'], dayfirst=True, format='mixed')
                df_added_after_final['dates'] = df_added_after_final['dates'].dt.strftime('%Y-%m-%d')
                df_added_after_final['dates'] = pd.to_datetime(df_added_after_final['dates'], dayfirst=True, format='mixed')


                df_added_after_final = df_added_after_final.loc[df_added_after_final['dates'] > df_sprints['startDate'][0]]

            else:
                build_index = []
                added_pds = []
                
                for i in range(0, len(range_sprint)):
                    # build_index.append(i)

                    number_to_add = 0
                    added_pds.append(number_to_add)
                    
                
                # df_added_after.index = build_index
                df_added_after['dates'] = pd.Series(range_sprint)
                df_added_after['pds_added'] = pd.Series(added_pds)
                
                df_added_after_final = df_added_after
            
                
            # Create a second data frame to get the number of items delivered each day
            df_delivered_effort = pd.DataFrame(columns=['dates'])
            df_delivered_effort['dates'] = original_range_sprint


            #### Dates preparation
            df_delivered_effort['dates'] = pd.to_datetime(df_delivered_effort['dates'], dayfirst=True, format='mixed')
            df_delivered_effort['dates'] = df_delivered_effort['dates'].dt.strftime('%Y-%m-%d')
            df_delivered_effort['dates'] = pd.to_datetime(df_delivered_effort['dates'], dayfirst=True, format='mixed')

            delivered_effort = df_current_sprint.loc[df_current_sprint['Resolved'] >= active_sprints['startDate'][0]].groupby(by='Resolved')['planned_effort'].sum()

            delivered_effort = delivered_effort.reset_index()

            delivered_effort.columns = ['dates', 'delivered_pds']

            #### Dates preparation
            delivered_effort['dates'] = pd.to_datetime(delivered_effort['dates'], dayfirst=True, format='mixed')
            delivered_effort['dates'] = delivered_effort['dates'].dt.strftime('%Y-%m-%d')
            delivered_effort['dates'] = pd.to_datetime(delivered_effort['dates'], dayfirst=True, format='mixed')


            # # ### Merge with the main data frame
            df_delivered_effort = df_delivered_effort.merge(delivered_effort, how='left', on='dates')
            df_delivered_effort = df_delivered_effort.merge(df_added_after_final, how='left', on='dates')


            # ### Create a column to store how many PDs still need to be delivered
            df_delivered_effort['remaining_pds'] = np.nan


            ### Fill NaN with 0 to allow calculation based on dates until today to have a nicer Burndownchart in the final process

            for i in range(0, len(df_delivered_effort), 1):
                if i == 0:
                    df_delivered_effort['delivered_pds'][i] = np.nan_to_num(df_delivered_effort['delivered_pds'][i])
                    df_delivered_effort['pds_added'][i] = np.nan_to_num(df_delivered_effort['pds_added'][i])
                    df_delivered_effort['remaining_pds'][i] = commited - df_delivered_effort['delivered_pds'][i]
                    
                else:
                    if df_delivered_effort['dates'][i] < pd.to_datetime(today, dayfirst=True):
                        df_delivered_effort['delivered_pds'][i] = np.nan_to_num(df_delivered_effort['delivered_pds'][i])
                        df_delivered_effort['pds_added'][i] = np.nan_to_num(df_delivered_effort['pds_added'][i])
                        
                        df_delivered_effort['remaining_pds'][i] = df_delivered_effort['remaining_pds'][i-1] - df_delivered_effort['delivered_pds'][i] + df_delivered_effort['pds_added'][i]
                        
                    elif df_delivered_effort['dates'][i] == pd.to_datetime(today, dayfirst=True):
                        cur_sprint_pds_open = df_current_sprint[df_current_sprint['Status'].isin(['Completed', 'Obsolete']) == False]['planned_effort'].sum()
                        df_delivered_effort['pds_added'][i] = np.nan_to_num(df_delivered_effort['pds_added'][i])
                        df_delivered_effort['remaining_pds'][i] = cur_sprint_pds_open


            df_burndown_pds = df_delivered_effort.merge(df_ideal_burndown, how='left', on='dates')
            df_burndown_pds = df_burndown_pds[df_burndown_pds.dates.isin(range_sprint)]
            
            return df_burndown_pds
        
        
        
        df_burndown_pds = burndownChartPDS(df_current_sprint)
        
            # Plot the Burndown Chart
        st.write("## Burndown Chart - PDs")  
        
        current_remaining = df_current_sprint.loc[~df_current_sprint['Status'].isin(['Completed', 'Obsolete'])]['planned_effort'].sum()
        
        idx_today = df_burndown['remaining_items'].notna().sum() - 1
        days_left = len(df_burndown['dates'][idx_today:])
        
        if days_left == 0:
            days_left = 1

        required_pace = round(current_remaining / days_left, 2)
        st.write(f"To accomplish the plan, we need to deliver **{required_pace}** PDs per day.")
        
        
        if len(df_burndown_pds) > 0:
            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=df_burndown_pds['dates'], 
                    y=df_burndown_pds['ideal_trend'], 
                    name='Ideal Trend',
                    line=dict(color="Black")
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=df_burndown_pds['dates'], 
                    y=df_burndown_pds['remaining_pds'], 
                    name = 'Remaining PDs',
                    fill='tozeroy',
                    line=dict(color="MediumSeaGreen")
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_burndown_pds['dates'], 
                    y=df_burndown_pds['pds_added'], 
                    name = 'PDs Added',
                    fill='tozeroy',
                    line=dict(color="Brown")
                    
                )
            )

            fig.update_layout(
                width=1000,
                height=500
            )

            fig.update_layout(legend=dict(orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                        legend_title_text='')

            st.plotly_chart(fig)

        else:
            print("There is no Active Sprint at this moment.")
    
    ############################################################################################################
    ############################################################################################################
    # ANCHOR WIP Analysis

    # Function to get all in progress items
    with st.spinner("Analyzing WIP..."):
        def prepareDataWIP(df):
            ######## Get the first log for 'Resolved Date' and first log for 'Started Date'
            ### Creates a Data Frame to store the Issue Keys

            df_change_log = pd.DataFrame(columns=['Issue key'])
            # df_resolved_change_log['Issue key'] = df[df['Resolved'].notnull()]['Issue key']
            df_change_log['Issue key'] = df['Issue key']
            df_change_log = df_change_log.reset_index(drop=True)

            df_transition_log = pd.DataFrame(columns=[
                'field', 'fieldtype', 'from', 'fromString', 'to', 'toString', 'created'
            ])


            for i in range(0, len(df_change_log), 1):
                get_change_log = jira_instance.get_issue_changelog(df_change_log['Issue key'][i])
                df_get_change_log = pd.json_normalize(get_change_log['histories'], 'items', 'created')
                df_get_change_log['issue_key'] = df_change_log['Issue key'][i]

                if len(df_get_change_log) > 0:
                    if len(df_get_change_log.loc[df_get_change_log['toString'] == 'In Progress']) > 0:
                        df_transition_log = pd.concat([df_transition_log, df_get_change_log.loc[df_get_change_log['toString'] == 'In Progress']],ignore_index=True)

            df_transition_log['created'] = df_transition_log['created'].str.split("T").str[0]

            df_transition_log = df_transition_log[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='last').reset_index(drop=True)

            df_transition_log['created'] = pd.to_datetime(df_transition_log['created'], dayfirst=True, format="mixed")

            df_transition_log = df_transition_log.rename(columns={'issue_key': 'Issue key', 'created': 'started'})

            df = df.merge(df_transition_log, how='left', on='Issue key')



            #### Dates preparation
            df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format='mixed')
            df['started'] = pd.to_datetime(df['started'], dayfirst=True, format='mixed')


            df['Created'] = df['Created'].dt.strftime('%Y-%m-%d')
            df['started'] = df['started'].dt.strftime('%Y-%m-%d')

            df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format="mixed")
            df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")


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


            df = df.rename(columns={'Custom field (Planned Effort)': 'planned_effort',
                                    'Original Estimate': 'original_estimate'})
            
            #### Time convertion
            df['planned_effort'] = df['planned_effort'].replace(np.nan, 0)
            df['original_estimate'] = df['original_estimate'].replace(np.nan, 0)
            
            df['planned_hours'] = df['planned_effort'].astype(int) * 8 * 0.9
            df['planned_effort'] = df['planned_effort'].astype(int)
            
            df['original_estimate'] = df['original_estimate'].astype(int)/86400 * 24 / 8
            
            df['planned_effort'] = df['planned_effort'] + df['original_estimate']

            ### Format the dates to get the Business Days

            df['Created'] = df['Created'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '')
            df['started'] = df['started'].apply(lambda x: x.strftime('%Y-%m-%d') if not pd.isnull(x) else '')


            return df

        # Define Today's Date
        today_wip = date.today()
        today_wip = today_wip.strftime("%Y-%m-%d")

        # Importing all data

        query_ide_wip = "project = IDEGENAI AND type in ('User Story', Activity, 'Backlog Item', 'Test Execution', 'Test Task', Bug) AND status not in (Open, 'In Refinement', 'Ref. Done', 'Ready for Dev.', Completed, Obsolete, Done)"
    
        df = pd.read_csv(BytesIO(jira_instance.csv(query_ide_wip, all_fields=True, delimiter=',')))
        df = prepareDataWIP(df)
        
        try:
            df = df[[
            'Priority', 'Issue Type', 'Issue key', 'Issue id', 'Status', 'Created',
            'Resolved','Custom field (Committed)', 'started',
            'Custom field (Parent Link)', 'Custom field (Flagged)', 'Assignee', 'Summary', 
            'Components', 'planned_effort', 'original_estimate'
            ]]
            
            df5 = df[['Issue key', 'started', 'Status', 'Components', 'planned_effort', 'Custom field (Flagged)']]
        
        except KeyError:
            df = df[[
            'Priority', 'Issue Type', 'Issue key', 'Issue id', 'Status', 'Created',
            'Resolved','Custom field (Committed)', 'started',
            'Custom field (Parent Link)', 'Assignee', 'Summary', 
            'Components', 'planned_effort', 'original_estimate'
            ]]
            
            df5 = df[['Issue key', 'started', 'Status', 'Components', 'planned_effort']]

        df5['time_wip'] = ''

        # Defining the Cycle Time in the Current Sprint
        
        if len(df_current_sprint[(df_current_sprint['Resolved'] >= st.session_state['resolved_date'])]) == 0:
            current_cycle_time = 0
        else: 
            current_cycle_time = int(df_current_sprint[(df_current_sprint['Resolved'] >= st.session_state['resolved_date'])]['cycle_time'].mean())


        status_to_ignore = ['Completed', 'Ready for Dev.']

        for i in range(0, len(df5), 1):
            if df5['started'][i] != '':
                if df5['Status'][i] not in status_to_ignore:
                    df5['time_wip'][i] = len(pd.bdate_range(start= df5['started'][i], end=today_wip, freq="B"))

        # Convert to Numpy int32 to match "Planned Effort" format
        df5['time_wip'] = df5['time_wip'].to_numpy('int32')

        df5 = df5.reset_index(drop=True)


        # Get the Real Cycle Time (excluding holidays and days off)
        for i in range(0, len(df5), 1):
            range_wip = pd.bdate_range(start = pd.to_datetime(df5['started'][i], dayfirst=False),
                                        end = pd.to_datetime(today_wip, dayfirst=False),
                                        freq="B")
            for j in range(0, len(st.session_state['non_working_brazil']), 1):
                if st.session_state['non_working_brazil'][j] in range_wip:
                    df5['time_wip'][i] -= 1        
                        
        df5['wip_higher'] = np.nan

        df5 = df5.sort_values(by='time_wip', ascending=False).reset_index(drop=True)
        


        for i in range(0, len(df5), 1):                
            if 'Custom field (Flagged)' in df5:
                if df5['Custom field (Flagged)'][i] == 'Impediment':
                    df5.at[i, 'wip_higher'] = 4
                    
                elif df5['time_wip'][i] <= df5['planned_effort'][i] and df5['time_wip'][i] <= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 0
                    
                elif df5['time_wip'][i] <= df5['planned_effort'][i] and df5['time_wip'][i] >= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 1
                    
                elif df5['time_wip'][i] >= df5['planned_effort'][i] and df5['time_wip'][i] <= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 2
                    
                elif df5['time_wip'][i] >= df5['planned_effort'][i] and df5['time_wip'][i] >= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 3
            
            else:
                if df5['time_wip'][i] <= df5['planned_effort'][i] and df5['time_wip'][i] <= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 0
                    
                elif df5['time_wip'][i] <= df5['planned_effort'][i] and df5['time_wip'][i] >= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 1
                    
                elif df5['time_wip'][i] >= df5['planned_effort'][i] and df5['time_wip'][i] <= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 2
                    
                elif df5['time_wip'][i] >= df5['planned_effort'][i] and df5['time_wip'][i] >= current_cycle_time:
                    df5.at[i, 'wip_higher'] = 3

        df5['wip_higher'] = df5['wip_higher'].fillna(0)
        df5['wip_higher'] = df5['wip_higher'].astype(int)

        df5['planned_effort'] = df5['planned_effort'].fillna(0)
        df5['planned_effort'] = df5['planned_effort'].astype(int)

        # Create the chart
        colors = df5['wip_higher'].to_list()

        for color in range(len(colors)):
            if colors[color] == 4:
                colors[color] = 'black'
            elif colors[color] == 3:
                colors[color] = 'salmon'
            elif colors[color] == 2:
                colors[color] = 'goldenrod'
            elif colors[color] == 1:
                colors[color] = 'moccasin'
            else:
                colors[color] = 'palegreen'


        st.write('## WIP Analysis')
        st.write("""
                Chart description:
                
                    - Gray bars: Indicate the planned effort for a specific item.
                    - Green bars: Indicate items with effort less than the planned effort and equal to or less than the mean sprint cycle time.
                    - Light Yellow bars: Represent items with effort less than the planned effort but greater than the mean sprint cycle time.
                    - Dark Yellow bars: Denote items with effort greater than the planned effort but less than the mean sprint cycle time.
                    - Red bars: Signify items where the effort is greater than both the planned effort and the mean sprint cycle time.
                    - Black bars: Indicate an item that is currently blocked.
                """)
        fig = px.bar(df5,
                    x='Issue key',
                    y=['planned_effort', 'time_wip'],
                    template='plotly_white',
                    text_auto=True,
                    barmode='group',
                    width=1000,
                    height=600,
                    labels={'y': 'Time in WIP (in Days)',
                        'x': 'Issue Key'})

        fig.update_layout(legend=dict(orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1),
                        legend_title_text='')

        fig.update_traces(marker_color=colors)

        fig.for_each_trace(lambda trace: trace.update(marker_color="lightgray") if trace.name == "planned_effort" else ())


        st.plotly_chart(fig)

    ############################################################################################################
    ############################################################################################################
    # ANCHOR Planned Effort X Cycle Time in the Current Sprint
    
    with st.spinner("Analyzing Planned Effort X Cycle Time in the Current Sprint..."):
        st.write("""--------------------""")
        st.write("### Planned Effort X Cycle Time in the Current Sprint")
        st.write("""
                Chart description:
                
                    - Gray bars: Indicate the planned effort for a specific item.
                    - Green bars: Indicate items with effort less than the planned effort.
                    - Light Yellow bars: Represent items with effort greater than the planned effort.
                    - Purple bars: Show items that were obsoleted after some effort was expended, indicating the number of days worked before cancellation.
                """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            
            # Create a Data Frame to analyze how the team behaves (Planned Effort X Cycle Time)
            df_perf_analy_bli = df_perf_analy[(df_perf_analy['Issue Type'] != 'Bug') & (df_perf_analy['Resolved'] >= df_sprints['startDate'][0])][['Issue key', 'Status', 'planned_effort', 'cycle_time', 'Summary', 'Issue Type', 'Resolved']]
            df_perf_analy_bli = df_perf_analy_bli.loc[~df_perf_analy_bli['cycle_time'].isna()].reset_index(drop=True)
            # df_perf_analy_bli = df_perf_analy_bli.loc[df_perf_analy['cycle_time'] - df_perf_analy['planned_effort'] >= 2].reset_index(drop=True)
            
            for i in range(0, len(df_perf_analy_bli), 1):
                if df_perf_analy_bli['cycle_time'][i] > df_perf_analy_bli['planned_effort'][i]:
                    df_perf_analy_bli.at[i, 'miss_plan_or_obsolete'] = 0
                elif df_perf_analy_bli['cycle_time'][i] <= df_perf_analy_bli['planned_effort'][i]:
                    df_perf_analy_bli.at[i, 'miss_plan_or_obsolete'] = 1
                
                if df_perf_analy_bli['Status'][i] == 'Obsolete':
                    df_perf_analy_bli.at[i, 'miss_plan_or_obsolete'] = 2
                
            colors = df_perf_analy_bli['miss_plan_or_obsolete'].to_list()

            for color in range(len(colors)):
                if colors[color] == 0:
                    colors[color] = 'moccasin'
                elif colors[color] == 1:
                    colors[color] = 'palegreen'
                else:
                    colors[color] = 'violet'
                


            fig = px.bar(df_perf_analy_bli,
                        x='Issue key',
                        y=['planned_effort', 'cycle_time'],
                        template='plotly_white',
                        text_auto=True,
                        barmode='group',
                        width=800,
                        title="Delta between Cycle Time and Planned Effort (Items with Estimations)")

            fig.update_traces(marker_color=colors)

            fig.for_each_trace(lambda trace: trace.update(marker_color="lightgray") if trace.name == "planned_effort" else ())
            
            st.plotly_chart(fig)
            
            df_perf_analy_bli['delta'] = round((df_perf_analy_bli['cycle_time'] - df_perf_analy_bli['planned_effort']), 2)
            st.write(f"The total deviation between the planned effort and the time spent to deliver the items is **{df_perf_analy_bli['delta'].sum()} PDs**")
        
        with col2:
            df_perf_analy_bug = df_perf_analy[(df_perf_analy['Issue Type'] == 'Bug')][['Issue key', 'Status', 'planned_effort', 'cycle_time', 'Summary', 'Issue Type', 'Resolved']]
            df_perf_analy_bug = df_perf_analy_bug.loc[~df_perf_analy_bug['cycle_time'].isna()].reset_index(drop=True)

            
            for i in range(0, len(df_perf_analy_bug), 1):
                if (df_perf_analy_bug['Status'][i] == 'Obsolete') & (df_perf_analy_bug['cycle_time'][i] > 0):
                    df_perf_analy_bug.at[i, 'miss_plan_or_obsolete'] = 2
                elif (df_perf_analy_bug['cycle_time'][i] > current_cycle_time):
                    df_perf_analy_bug.at[i, 'miss_plan_or_obsolete'] = 0
                else:
                    df_perf_analy_bug.at[i, 'miss_plan_or_obsolete'] = 1

            try:
                colors = df_perf_analy_bug['miss_plan_or_obsolete'].to_list()
            
            except KeyError:
                colors = []   
            
            for color in range(len(colors)):
                    if colors[color] == 0:
                        colors[color] = 'moccasin'
                    elif colors[color] == 1:
                        colors[color] = 'palegreen'
                    else:
                        colors[color] = 'violet'
                    


            fig = px.bar(df_perf_analy_bug,
                        x='Issue key',
                        y=['cycle_time'],
                        template='plotly_white',
                        text_auto=True,
                        barmode='group',
                        width=800,
                        title="Bugs Cycle Time in the Current Sprint")

            fig.update_traces(marker_color=colors)

            st.plotly_chart(fig)
            
            df_perf_analy_bug['delta'] = round(df_perf_analy_bug['cycle_time'], 2)
            st.write(f"The total time spent fixing bugs represents **{df_perf_analy_bug['delta'].sum()} PDs**")

        pd.set_option('display.max_colwidth', 100)  # Set the maximum width to 1000 characters
        df_deviation = df_perf_analy
        df_deviation = df_deviation.loc[(df_deviation['Status'].isin(['Completed', 'Obsolete']))]
        df_deviation['cycle_time'] = df_deviation['cycle_time'].fillna(0)
        df_deviation['delta'] = round((df_deviation['cycle_time'] - df_deviation['planned_effort']), 2)
        df_deviation = df_deviation[['Issue key', 'Summary', 'Issue Type', 'Status', 'planned_effort', 'cycle_time', 'delta']]
        df_deviation = df_deviation.sort_values(by='delta', ascending=False).reset_index(drop=True)
        
        st.dataframe(df_deviation)
        
    ############################################################################################################
    ############################################################################################################
    # ANCHOR Time Blocked
    with st.spinner("Analyzing Blocked Time"):
        st.write("""--------------------""")
        st.write("### Blocked Time by Item")
        
        df_time_blocked = df_time_blocked[pd.to_numeric(df_time_blocked['blocked_duration'], errors='coerce') > 0]

        fig = px.bar(df_time_blocked, 
                    x='Issue key', 
                    y='blocked_duration', 
                    title='Total Blocked Duration by Issue', 
                    text_auto=True)

        st.plotly_chart(fig)

if __name__ == '__main__':
    main()
