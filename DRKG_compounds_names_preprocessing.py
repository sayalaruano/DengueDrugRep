#%%
# Import libraries
import pandas as pd
import csv
import re

#%% 
# Specify the file path
file_path = "Data/DRKG/compounds_DRKG.tsv"

# Open .tsv file
with open(file_path, 'r') as file:
    # Create a csv.reader object with tab delimiter
    tsv_reader = csv.reader(file, delimiter='\t')

    # Create a list to store rows
    rows = []

    # Read and append each row to the list
    for row in tsv_reader:
        # Concatenate all elements after the first element into the second element
        row = [row[0], ' '.join(row[1:])]

        rows.append(row)

# Create a DataFrame without specifying headers
compounds_DRKG = pd.DataFrame(rows, columns=["Entity", "Source"])
#%%
# Create a column with the compound id
compounds_DRKG['compound_id'] = compounds_DRKG['Entity'].str.split('::').str[1]

# Iterate over rows and create a column with the data source
# The CHEMBL, DrugBank and nmrshiftdb2 entries have the ID directly after the :: symbol, while 
# the rest of the entries have the name of the database and then the ID separated by a colon
for index, row in compounds_DRKG.iterrows():
    if row['compound_id'].startswith('CHEMBL'):
        compounds_DRKG.loc[index, 'data_source'] = "CHEMBL"
    elif row['compound_id'].startswith('DB'):
        compounds_DRKG.loc[index, 'data_source'] = "DrugBank"
    elif row['compound_id'].startswith('nmrshiftdb2'):
        compounds_DRKG.loc[index, 'data_source'] = "nmrshiftdb2"
    else:
        compounds_DRKG.loc[index, 'data_source'] = re.search(r'[:\s]*([A-Za-z]+)', compounds_DRKG.loc[index, 'compound_id']).group(1)

# Unify the CHEBI identidiers
compounds_DRKG['data_source'] = compounds_DRKG['data_source'].replace('chebi', 'CHEBI')
#%%
# Add a column with IDs without the data source 
for index, row in compounds_DRKG.iterrows():
    if ":" in row['compound_id']:
        compounds_DRKG.loc[index, 'compound_id_short'] = compounds_DRKG.loc[index, 'compound_id'].split(":")[1]
#%%
# Export the dataframe as a csv file
compounds_DRKG.to_csv('Data/DRKG/compounds_DRKG_datasource.csv', index=False)

# %%
# Create a list with the compound names
db_names = compounds_DRKG['data_source'].unique().tolist()

# Add the "_id" word in the end of each element of the list
db_names = [x + "_id" for x in db_names]

# Convert the list into a df and export it as a csv file
db_names = pd.DataFrame(db_names)
db_names.to_csv('Data/DRKG/compounds_DRKG_datasources_names.csv', index=False, header=False)

# %%
