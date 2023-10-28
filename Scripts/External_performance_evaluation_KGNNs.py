#%%
# Import libraries
import pandas as pd
import torch
from pykeen.datasets import DRKG
from pykeen import predict
import re

#%%
# Function to load the KGNNs pre-trained models
def load_model(model_name, parent_folder):
    '''
    Function to load the KGNNs pre-trained models
    Input: model_name (str) - name of the model to be loaded
           parent_folder (str) - name of the parent folder where the model is located
    Output: model (torch.nn.Module) - model loaded'''
    # Define the path and load the model
    model_path = 'Models/' + parent_folder + '/DRKG_' + model_name + '/trained_model.pkl' 
    model = torch.load(model_path, map_location=torch.device('cpu'))
    print(model)
    
    return model

# Function to add column with the compound IDs, extracted from the head_label column
def add_compound_ids(df):
    '''
    Function to add column with the compound IDs, extracted from the head_label column
    Input: df (pandas DataFrame) - dataframe with the predictions of the KGGNs model
    Output: df (pandas DataFrame) - dataframe with the compound IDs'''
    # Create a column with the compound id
    df['compound_id'] = df['head_label'].str.split('::').str[1]

    # Iterate over rows and create a column with the data source
    # The CHEMBL, DrugBank and nmrshiftdb2 entries have the ID directly after the :: symbol, while 
    # the rest of the entries have the name of the database and then the ID separated by a colon
    for index, row in df.iterrows():
        if row['compound_id'].startswith('CHEMBL'):
            df.loc[index, 'data_source'] = "CHEMBL"
        elif row['compound_id'].startswith('DB'):
            df.loc[index, 'data_source'] = "DrugBank"
        elif row['compound_id'].startswith('nmrshiftdb2'):
            df.loc[index, 'data_source'] = "nmrshiftdb2"
        else:
            df.loc[index, 'data_source'] = re.search(r'[:\s]*([A-Za-z]+)', df.loc[index, 'compound_id']).group(1)

    # Unify the CHEBI identidiers
    df['data_source'] = df['data_source'].replace('chebi', 'CHEBI')

    return df

# Function to make predictions  with a KGGNs model for a given disease and relation
def make_pred_and_compfilt(model, relation, disease, train_triples, test_triplets):
    '''
    Function to make predictions  with a KGGNs model for a given disease and relation
    and filter the predictions to only include the compounds.
    Input: model (torch.nn.Module) - KGGNs model to be used for the predictions
           relation (str) - relation to be used for the predictions
           disease (str) - disease to be used for the predictions
           train_triples (pykeen.triples.TriplesFactory) - training triples of the DRKG
           test_triplets (pykeen.triples.TriplesFactory) - testing triples of the DRKG
    Output: df (pandas DataFrame) - dataframe with the predictions of the KGGNs model'''
    # Make predictions
    predictions = predict.predict_target(model=model, 
                                         relation=relation, 
                                         tail=disease, 
                                         triples_factory=train_triples).add_membership_columns(testing = test_triplets)

    # Filter triplets that appear in the training set
    predictions_filt = predictions.filter_triples(train_triples)

    # Convert to dataframe
    predictions = predictions.df
    predictions_filt = predictions_filt.df

    # Create a df with the triplets that appear in the training set
    merged = pd.merge(predictions, predictions_filt, how='outer', indicator=True)
    predictions_train = merged[merged['_merge'] != 'both'].drop(columns=['_merge'])

    # Create column to define if prediction result is a compound or not
    predictions_filt['is_compound'] = ['yes' if 'Compound' in c else 'no' for c in predictions_filt['head_label']]

    # Filter only the compounds
    predictions_filt = predictions_filt[predictions_filt['is_compound'] == 'yes']

    # Add the compound IDs to the dataframe
    predictions_filt = add_compound_ids(predictions_filt)

    # Sort values by score and reset indices
    predictions_filt = predictions_filt.sort_values(by=['score'], ascending=False).reset_index(drop=True)

    # Return the predictions and the predictions that appear in the training set
    return predictions_filt, predictions_train

# Function to calculate the external validation rank metrics for a given KGGN model and ground truth data
def calc_rank_metrics(mode_name, model_pred, validated_drugs):
    '''
    Function to calculate the external validation rank metrics for a given KGGN model and ground truth data
    Input: mode_name (str) - name of the model to be loaded
           model_pred (pandas DataFrame) - dataframe with the predictions of the KGGNs model
           validated_drugs (pandas DataFrame) - dataframe with the validated drugs
    Output: rank_metrics (pandas DataFrame) - dataframe with the rank metrics'''

    # Create a dictionary with the drug column as keys and the rest of the columns as values
    validated_drugs_dict = validated_drugs.set_index('drug').T.to_dict('list')

    # Delete nan values from the values of the dictionary
    for key, value in validated_drugs_dict.items():
        validated_drugs_dict[key] = [x for x in value if str(x) != 'nan']

    # Now, for every drug, take the ids values and find the indices of rows from the predictions df
    # The positions in the sorted list represent the rank of the predictions to evalute the performance of the model
    # Create a dictionary to store the results
    validated_drugs_idxs_dict = {}

    # Iterate over the keys and values of the dictionary
    for key, value in validated_drugs_dict.items():
        # Create a list to store the indices
        indices = []
        # Iterate over the values of the dictionary
        for v in value:
            # Find the indices of the rows that match the values of the dictionary
            match = model_pred[model_pred['compound_id'] == v].index.values
            indices.append(match)
        # Flatten the list of lists
        indices = [item for sublist in indices for item in sublist]
        # Remove duplicates
        indices = list(dict.fromkeys(indices))
        # Add the indices to the dictionary
        validated_drugs_idxs_dict[key] = indices

    # Extract the lowest values for the array values from the dictionary
    validated_drugs_idxs_dict = {k: min(v) for k, v in validated_drugs_idxs_dict.items()}

    # Create a list with the values from the dictionary
    validated_drugs_idxs = sorted(list(validated_drugs_idxs_dict.values()))

    # Obatin the lowest, highest, and mean rank of the predictions and put them into a datframe
    lowest_rank = int(min(validated_drugs_idxs))
    highest_rank = int(max(validated_drugs_idxs))

    # Calculate the median rank
    middle_index = len(validated_drugs_idxs) // 2
    median_rank = validated_drugs_idxs[middle_index]

    # Create a dataframe with the results
    rank_metrics = pd.DataFrame(data={mode_name: [lowest_rank, median_rank, highest_rank]}, index=['First_hit', 'Median_hit', 'Last_hit']).T

    return rank_metrics

#%%
# Load knowledge graph 
drkg = DRKG()

# Create triples of Training set
drkg_train = drkg.training

# Create triples of Validation set
drkg_val = drkg.validation

# Create triples of Testing set
drkg_test = drkg.testing

# Load the KGNNs models trained on Google Colab
ERMLP_model_genev = load_model('ERMLP_50epochs', 'General_evaluation')
DistMult_model_genev = load_model('DISMULT_50epochs', 'General_evaluation')
PairE_model_genev = load_model('PairRE_50epochs', 'General_evaluation')
TransR_model_genev = load_model('TransR_50epochs', 'General_evaluation')

ERMLP_model_drev  = load_model('ERMLP_10epochs', 'Drug_rep_evaluation')
DistMult_model_drev = load_model('DISMULT_10epochs', 'Drug_rep_evaluation')
PairE_model_drev = load_model('PairRE_10epochs', 'Drug_rep_evaluation')
TransR_model_drev = load_model('TransR_10epochs', 'Drug_rep_evaluation')

#%%
# Perform head prediction for dengue disease using the GNBR compound-disease relation 
dengue_entity_drkg = 'Disease::MESH:D014355'
GNBR_compound_disease = 'GNBR::T::Compound:Disease'
Hetionet_compound_disease = 'Hetionet::CtD::Compound:Disease'
Drugbank_compound_disease = 'DRUGBANK::treats::Compound:Disease'

# ERMLP models predictions
# General evaluation
ERMLP_pred_dengue_genev, ERMLP_pred_dengue_genev_train = make_pred_and_compfilt(model=ERMLP_model_genev, 
                                                                                relation=GNBR_compound_disease, 
                                                                                disease=dengue_entity_drkg, 
                                                                                train_triples=drkg_train, 
                                                                                test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
ERMLP_pred_dengue_genev.to_csv('Results/CompoundDisease_predictions/General_evaluation/pred_dengue_emrlp_genev.csv', sep=',', index=False)
ERMLP_pred_dengue_genev_train.to_csv('Results/Triplets_in_train/General_evaluation/pred_dengue_emrlp_genev_train.csv', sep=',', index=False)

# Drug repurposing evaluation
ERMLP_pred_dengue_drev, ERMLP_pred_dengue_drev_train = make_pred_and_compfilt(model=ERMLP_model_drev,
                                                                            relation=GNBR_compound_disease, 
                                                                            disease=dengue_entity_drkg, 
                                                                            train_triples=drkg_train, 
                                                                            test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
ERMLP_pred_dengue_drev.to_csv('Results/CompoundDisease_predictions/Drug_rep_evaluation/pred_dengue_emrlp_drev.csv', sep=',', index=False)
ERMLP_pred_dengue_drev_train.to_csv('Results/Triplets_in_train/Drug_rep_evaluation/pred_dengue_emrlp_drev_train.csv', sep=',', index=False)

# DistMult models predictions
# General evaluation
DistMult_pred_dengue_genev, DistMult_pred_dengue_genev_train = make_pred_and_compfilt(model=DistMult_model_genev, 
                                                                                      relation=GNBR_compound_disease, 
                                                                                      disease=dengue_entity_drkg, 
                                                                                      train_triples=drkg_train, 
                                                                                      test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
DistMult_pred_dengue_genev.to_csv('Results/CompoundDisease_predictions/General_evaluation/pred_dengue_distmult_genev.csv', sep=',', index=False)
DistMult_pred_dengue_genev_train.to_csv('Results/Triplets_in_train/General_evaluation/pred_dengue_distmult_genev_train.csv', sep=',', index=False)

# Drug repurposing evaluation
DistMult_pred_dengue_drev, DistMult_pred_dengue_drev_train = make_pred_and_compfilt(model=DistMult_model_drev, 
                                                                                    relation=GNBR_compound_disease, 
                                                                                    disease=dengue_entity_drkg, 
                                                                                    train_triples=drkg_train, 
                                                                                    test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
DistMult_pred_dengue_drev.to_csv('Results/CompoundDisease_predictions/Drug_rep_evaluation/pred_dengue_distmult_drev.csv', sep=',', index=False)
DistMult_pred_dengue_drev_train.to_csv('Results/Triplets_in_train/Drug_rep_evaluation/pred_dengue_distmult_drev_train.csv', sep=',', index=False)

# PairE models predictions
# General evaluation
PairE_pred_dengue_genev, PairE_pred_dengue_genev_train = make_pred_and_compfilt(model=PairE_model_genev, 
                                                                                relation=GNBR_compound_disease, 
                                                                                disease=dengue_entity_drkg, 
                                                                                train_triples=drkg_train, 
                                                                                test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
PairE_pred_dengue_genev.to_csv('Results/CompoundDisease_predictions/General_evaluation/pred_dengue_paire_genev.csv', sep=',', index=False)
PairE_pred_dengue_genev_train.to_csv('Results/Triplets_in_train/General_evaluation/pred_dengue_paire_genev_train.csv', sep=',', index=False)

# Drug repurposing evaluation
PairE_pred_dengue_drev, PairE_pred_dengue_drev_train = make_pred_and_compfilt(model=PairE_model_drev, 
                                                                                relation=GNBR_compound_disease, 
                                                                                disease=dengue_entity_drkg, 
                                                                                train_triples=drkg_train, 
                                                                                test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
PairE_pred_dengue_drev.to_csv('Results/CompoundDisease_predictions/Drug_rep_evaluation/pred_dengue_paire_drev.csv', sep=',', index=False)
PairE_pred_dengue_drev_train.to_csv('Results/Triplets_in_train/Drug_rep_evaluation/pred_dengue_paire_drev_train.csv', sep=',', index=False)

# TransR models predictions
# General evaluation
TransR_pred_dengue_genev, TransR_pred_dengue_genev_train = make_pred_and_compfilt(model=TransR_model_genev, 
                                                                                    relation=GNBR_compound_disease, 
                                                                                    disease=dengue_entity_drkg, 
                                                                                    train_triples=drkg_train, 
                                                                                    test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
TransR_pred_dengue_genev.to_csv('Results/CompoundDisease_predictions/General_evaluation/pred_dengue_transr_genev.csv', sep=',', index=False)
TransR_pred_dengue_genev_train.to_csv('Results/Triplets_in_train/General_evaluation/pred_dengue_transr_genev_train.csv', sep=',', index=False)

# Drug repurposing evaluation
TransR_pred_dengue_drev, TransR_pred_dengue_drev_train = make_pred_and_compfilt(model=TransR_model_drev, 
                                                                                    relation=GNBR_compound_disease, 
                                                                                    disease=dengue_entity_drkg, 
                                                                                    train_triples=drkg_train, 
                                                                                    test_triplets=drkg_test)
# Export predictions and predictions that appear in the training set as csv files
TransR_pred_dengue_drev.to_csv('Results/CompoundDisease_predictions/Drug_rep_evaluation/pred_dengue_transr_drev.csv', sep=',', index=False)
TransR_pred_dengue_drev_train.to_csv('Results/Triplets_in_train/Drug_rep_evaluation/pred_dengue_transr_drev_train.csv', sep=',', index=False)

#%%
# Load the clilical trial compounds - ground truth data
clin_drugs = pd.read_csv('Data/Clinical_trials/dengue_validated_drugs_clin.csv', sep=',', dtype=str)

# Drop unnecesary columns from the dataframe
clin_drugs.drop(['start_yr', 'ClinVar_id'], axis=1, inplace=True)

#%%
# Calculate the external validation rank metrics for the KGGNs models
# General evaluation
ERMLP_rank_metrics_genev = calc_rank_metrics('ERMLP_genev', ERMLP_pred_dengue_genev, clin_drugs)
DistMult_rank_metrics_genev = calc_rank_metrics('DistMult_genev', DistMult_pred_dengue_genev, clin_drugs)
PairE_rank_metrics_genev = calc_rank_metrics('PairE_genev', PairE_pred_dengue_genev, clin_drugs)
TransR_rank_metrics_genev = calc_rank_metrics('TransR_genev', TransR_pred_dengue_genev, clin_drugs)

# Drug repurposing evaluation
ERMLP_rank_metrics_drev = calc_rank_metrics('ERMLP_drev', ERMLP_pred_dengue_drev, clin_drugs)
DistMult_rank_metrics_drev = calc_rank_metrics('DistMult_drev', DistMult_pred_dengue_drev, clin_drugs)
PairE_rank_metrics_drev = calc_rank_metrics('PairE_drev', PairE_pred_dengue_drev, clin_drugs)
TransR_rank_metrics_drev = calc_rank_metrics('TransR_drev', TransR_pred_dengue_drev, clin_drugs)

#%%
# Concatenate the results for general evaluation models into a single dataframe
ext_val_rank_met_genev = pd.concat([ERMLP_rank_metrics_genev, DistMult_rank_metrics_genev, 
                                    PairE_rank_metrics_genev, TransR_rank_metrics_genev], 
                                    axis=0)

# Rename indices of the dataframe
ext_val_rank_met_genev.index = ['ERMLP', 'DistMult', 'PairE', 'TransR']

# Concatenate the results for drug repurposing evaluation models into a single dataframe
ext_val_rank_met_drev = pd.concat([ERMLP_rank_metrics_drev, DistMult_rank_metrics_drev,
                                      PairE_rank_metrics_drev, TransR_rank_metrics_drev],
                                        axis=0)

# Rename indices of the dataframe
ext_val_rank_met_drev.index = ['ERMLP', 'DistMult', 'PairE', 'TransR']

# Export the results as a csv files
ext_val_rank_met_genev.to_csv('Results/External_evaluation/External_evaluation_Dengue_rank_metrics_genev.csv', sep=',', index=True)
ext_val_rank_met_drev.to_csv('Results/External_evaluation/External_evaluation_Dengue_rank_metrics_drev.csv', sep=',', index=True)

#%%