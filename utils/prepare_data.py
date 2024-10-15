import pandas as pd
import numpy as np
from atlassian import Jira
import streamlit as st


############################################################################################################
############################################################################################################
### ANCHOR Parameters - Instantiate Jira


jira_instance = Jira(
    url = st.secrets["db_url"],
    username = st.secrets["db_username"],
    password = st.secrets["db_password"]
)

def prepareData(df, components, non_working_days):
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


    df_resolved_transition_log['created'] = df_resolved_transition_log['created'].str.split("T").str[0]
    df_resolved_transition_log = df_resolved_transition_log[['issue_key', 'created']].drop_duplicates(subset=['issue_key'], keep='first').reset_index(drop=True)
    df_resolved_transition_log['created'] = pd.to_datetime(df_resolved_transition_log['created'], dayfirst=True, format="mixed")
    df_resolved_transition_log = df_resolved_transition_log.rename(columns={'issue_key': 'Issue key', 'created': 'Resolved'})
    df = df.merge(df_resolved_transition_log, how='left', on='Issue key')

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
    df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")
    df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")
    # df['blocked_at'] = pd.to_datetime(df['blocked_at'], dayfirst=True, format="mixed")
    # df['unblocked_at'] = pd.to_datetime(df['unblocked_at'], dayfirst=True, format="mixed")

    df['Created'] = df['Created'].dt.strftime('%Y-%m-%d')
    df['Resolved'] = df['Resolved'].dt.strftime('%Y-%m-%d')
    df['started'] = df['started'].dt.strftime('%Y-%m-%d')
    # df['blocked_at'] = df['blocked_at'].dt.strftime('%Y-%m-%d')
    # df['unblocked_at'] = df['unblocked_at'].dt.strftime('%Y-%m-%d')

    df['Created'] = pd.to_datetime(df['Created'], dayfirst=True, format="mixed")
    df['Resolved'] = pd.to_datetime(df['Resolved'], dayfirst=True, format="mixed")
    df['started'] = pd.to_datetime(df['started'], dayfirst=True, format="mixed")
    # df['blocked_at'] = pd.to_datetime(df['blocked_at'], dayfirst=True, format="mixed")
    # df['unblocked_at'] = pd.to_datetime(df['unblocked_at'], dayfirst=True, format="mixed")


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

    if components == 'Team Brazil':
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

    elif components == 'Team Integration':
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

    else:
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

    if components == 'Team Brazil':
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

    elif components == 'Team Integration':
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

    else:
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