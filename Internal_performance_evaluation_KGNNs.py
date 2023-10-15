#%% 
# Import libraries
import pandas as pd
import json
import matplotlib.pyplot as plt
# Set the style for the plot
plt.style.use('seaborn-v0_8-deep')

#%% 
# Function to load the results of the the KGNNs pre-trained models 
def load_model_results(model_name, parent_folder):
    '''
    Function to load the KGNNs pre-trained models
    Input: model_name (str) - name of the model to be loaded
           parent_folder (str) - name of the parent folder where the model is located
    Output: results (dict) - dictionary with the results of the model
    '''
    # Load json file with the results of the model
    with open('Models/' + parent_folder + '/DRKG_' + model_name + '/results.json') as json_file:
        results = json.load(json_file)
    
    return results

# Function to plot the losses of the model
def plot_losses(ax, results_model, model_name, color=None):
    '''
    Function to plot the losses of the model across epochs
    Input: ax (matplotlib.axes._subplots.AxesSubplot) - axis for the plot
           results_model (dict) - dictionary with the results of the model
           model_name (str) - name of the model to be loaded
           color (str, optional) - color of the plot line
    '''
    # Convert losses to dataframe
    loss = pd.DataFrame(results_model['losses'])

    # Plot the losses with the specified color
    loss.plot(ax=ax, color=color)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')

# Function to extract and append performance metrics from models into a dataframe
def get_perf_metrics(results, model_name):
    '''
    Function to extract and append performance metrics from models into a dataframe
    Input: results (dict) - dictionary with the results of the model
              model_name (str) - name of the model to be loaded
    Output: perf_metrics (pandas.core.frame.DataFrame) - dataframe
    '''
    # Extract the performance metrics from the dictionary
    perf_metrics = pd.DataFrame([results["metrics"]["both"]["realistic"]]).T
    # Change the name of the column
    perf_metrics.columns = [model_name]
    return perf_metrics

#%%
# Load the results of the KGNNs models trained on Google Colab
ERMLP_results  = load_model_results('ERMLP_50epochs', 'General_evaluation')
DistMult_results = load_model_results('DISMULT_50epochs', 'General_evaluation')
PairE_results = load_model_results('PairRE_50epochs', 'General_evaluation')
TransR_results = load_model_results('TransR_50epochs', 'General_evaluation')

#%% 
# Plot the losses of the models in log scale
fig1, ax1 = plt.subplots()

plot_losses(ax1, ERMLP_results, 'ERMLP', color='mediumpurple')
plot_losses(ax1, PairE_results, 'PairRE', color='lightcoral')
plot_losses(ax1, TransR_results, 'TransR', color='steelblue')
plot_losses(ax1, DistMult_results, 'DISMULT', color='green')

# Add legends for each line
ax1.legend(['ERMLP', 'PairRE', 'TransR', 'DISMULT'])
ax1.set_yscale('log')
ax1.set_ylabel('Log(Loss)')
fig1.tight_layout()
fig1.set_facecolor('w')

# Save the figure
fig1.savefig('Results/Internal_evaluation/losses_models_50ep_genev.png', dpi=300, bbox_inches='tight')

#%%
# Load the results of the KGNNs models trained on Google Colab - 10 epochs
ERMLP10ep_results  = load_model_results('ERMLP_10epochs', 'Drug_rep_evaluation')
DistMult10ep_results = load_model_results('DISMULT_10epochs', 'Drug_rep_evaluation')
PairE10ep_results = load_model_results('PairRE_10epochs', 'Drug_rep_evaluation')
TransR10ep_results = load_model_results('TransR_10epochs', 'Drug_rep_evaluation')

#%%
# Plot the losses of the models in log scale
fig2, ax2 = plt.subplots()

plot_losses(ax2, ERMLP10ep_results, 'ERMLP', color='mediumpurple')
plot_losses(ax2, PairE10ep_results, 'PairRE', color='lightcoral')
plot_losses(ax2, TransR10ep_results, 'TransR', color='steelblue')
plot_losses(ax2, DistMult10ep_results, 'DISMULT', color='green')

# Add legends for each line
ax2.legend(['ERMLP', 'PairRE', 'TransR', 'DISMULT'])
ax2.set_yscale('log')
ax2.set_ylabel('Log(Loss)')
fig2.tight_layout()
fig2.set_facecolor('w')

# Save the figure
fig2.savefig('Results/Internal_evaluation/losses_models_10ep_drev.png', dpi=300, bbox_inches='tight')

#%%
# Obtain the performance metrics of the models
ERMLP_perf_met = get_perf_metrics(ERMLP_results, 'ERMLP')
DistMult_perf_met = get_perf_metrics(DistMult_results, 'DISMULT')
PairE_perf_met = get_perf_metrics(PairE_results, 'PairRE')
TransR_perf_met = get_perf_metrics(TransR_results, 'TransR')

# Append the performance metrics into a dataframe
perf_met = ERMLP_perf_met.join(DistMult_perf_met)
perf_met = perf_met.join(PairE_perf_met)
perf_met = perf_met.join(TransR_perf_met)

# Save the dataframe into a csv file
perf_met.to_csv('Results/Internal_evaluation/performance_metrics_all_50ep_genev.csv', sep=',')

#%%
# Obtain the performance metrics of the models 10 epochs
ERMLP10ep_perf_met = get_perf_metrics(ERMLP10ep_results, 'ERMLP')
DistMult10ep_perf_met = get_perf_metrics(DistMult10ep_results, 'DISMULT')
PairE10ep_perf_met = get_perf_metrics(PairE10ep_results, 'PairRE')
TransR10ep_perf_met = get_perf_metrics(TransR10ep_results, 'TransR')

# Append the performance metrics into a dataframe
perf_met10ep = ERMLP10ep_perf_met.join(DistMult10ep_perf_met)
perf_met10ep = perf_met10ep.join(PairE10ep_perf_met)
perf_met10ep = perf_met10ep.join(TransR10ep_perf_met)

# Save the dataframe into a csv file
perf_met10ep.to_csv('Results/Internal_evaluation/performance_metrics_all_10ep_drev.csv', sep=',')
#%%
# Filter the performance metric dataframe to obtain only the metrics of interest
perf_met_filt = perf_met.loc[['adjusted_arithmetic_mean_rank', 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10'], :]

# Transpose the dataframe
perf_met_filt = perf_met_filt.T

# Sort the dataframe by the adjusted arithmetic mean rank
perf_met_filt = perf_met_filt.sort_values(by='adjusted_arithmetic_mean_rank', ascending=True)

# Save the dataframe into a csv file
perf_met_filt.to_csv('Results/Internal_evaluation/performance_metrics_all_50ep_genev_filtered.csv', sep=',')

#%%
# Filter the performance metric dataframe to obtain only the metrics of interest
perf_met10ep_filt = perf_met10ep.loc[['adjusted_arithmetic_mean_rank', 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10'], :]

# Transpose the dataframe
perf_met10ep_filt = perf_met10ep_filt.T

# Sort the dataframe by the adjusted arithmetic mean rank
perf_met10ep_filt = perf_met10ep_filt.sort_values(by='adjusted_arithmetic_mean_rank', ascending=True)

# Save the dataframe into a csv file
perf_met10ep_filt.to_csv('Results/Internal_evaluation/performance_metrics_all_10ep_drev_filtered.csv', sep=',')
# %%
