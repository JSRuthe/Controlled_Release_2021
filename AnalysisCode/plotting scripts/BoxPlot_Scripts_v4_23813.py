import pandas as pd
import numpy as np
import datetime
from pytz import timezone
import pytz
from dateutil import parser
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.predstd import wls_prediction_std

import plotly
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib as mpl
mpl.use('tkagg')    #YAAA!!  this finally makes the Damn thing work
import matplotlib.pyplot as plt
#matplotlib inline
plt.rcParams['figure.figsize'] = (5, 5) # set default size of plots

from parity_plot_scripts_2385 import f_test, parity_plot, linreg_results, linreg_results_no_intercept

import seaborn as sns

# directory for storing graphs generated
# import os
# graph_dir = os.path.join('drive/My Drive/', root_path)+'graphs_SI/'

from datetime import date
today = date.today()
fdate = date.today().strftime('%m%d%Y')    # append the data today when exporting the graphs


# ignore some warnings
import warnings
warnings.filterwarnings('ignore')


font = {'family': 'Arial',
        'weight': 'normal',
        'size': 12}
font_small = {'family': 'Arial',
        'weight': 'normal',
        'size': 24}


import os.path
cwd = os.getcwd()
import pandas as pd

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_Bridger_23822.csv')
matchedDF_Bridger = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_GHGSat_23822.csv')
matchedDF_GHGSat = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_CarbonMapper_23822.csv')
matchedDF_CarbonMapper = pd.read_csv(csvPath)




## --------------------------CARBON MAPPER BOX AND WHISKER---------------------------------------------------------##


plt.subplots_adjust(hspace=0.5)
fig, axes = plt.subplots(3, 1, figsize=(10, 6), facecolor='w', edgecolor='k')

for i in range(3):

    plot_data = matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == (i + 1)) & (
            matchedDF_CarbonMapper['tc_Classification'] == 'TP') & (
            matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50)]

    plot_data["FlowError_percent"] = pd.to_numeric(plot_data.FlowError_percent, errors='coerce')

    plot_data["bin"] = pd.cut(plot_data["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])
    medians = plot_data.groupby(["bin"])["FlowError_percent"].median().values
    Q1, Q3 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.25), plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.75)
    IQR = Q3 - Q1
    IQR = IQR.to_numpy()	

    sns.boxplot(x="bin", y="FlowError_percent", data=plot_data, ax=axes[i], color='#8c1515')
    axes[i].set_ylim([-200, 150])

    nobs = plot_data["Wind_MPS_mean300"].value_counts(bins=[0, 1, 2, 3, 4, 5, 20]).sort_index(ascending=True).values
    #nobs = [str(x) for x in nobs.tolist()]
    #nobs = ["n: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, axes[i].get_xticklabels()):
        axes[i].text(pos[tick],
                     -225,
                     "{0} {1:.2f}".format('n: ', nobs[tick]) + '\n' + \
		     "{0} {1:.2f}".format('Q2: ', medians[tick]) + '\n' + \
		     "{0} {1:.2f}".format('IQR: ', IQR[tick]) + '\n',
                     horizontalalignment='center',
                     size='x-small',
                     color='k',
                     weight='normal',
		     verticalalignment = 'bottom')

axes[0].set_xlabel('', fontsize=6)
axes[1].set_xlabel('', fontsize=6)
axes[1].set_ylabel('Quantification Error [%]', fontsize=12)
axes[2].set_xlabel('Bin - 5 minute wind speed [mps]', fontsize=12)
axes[2].tick_params(labelsize=12)

axes[0].set_xticks(range(len(nobs)))
axes[1].set_xticks(range(len(nobs)))
axes[2].set_xticks(range(len(nobs)))
axes[0].set_xticklabels(['', '', '', '', '', ''])
axes[1].set_xticklabels(['', '', '', '', '', ''])
axes[2].set_xticklabels(['0-1', '1-2', '2-3', '3-4', '4-5', '5+'])

plt.rc('font', **font)

plt.savefig('CarbonMapper_Boxwhisker_23923.svg')
plt.close()

# ---------------------BRIDGER BOX AND WHISKER ---------------------------------------------------------------------#

plt.subplots_adjust(hspace=0.5)
fig, axes = plt.subplots(2, 2, figsize=(10, 6), facecolor='w', edgecolor='k')

# Unblinding stage 1 - HRRR

plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == (1)) & (
        matchedDF_Bridger['tc_Classification'] == 'TP') & (
                                      matchedDF_Bridger['WindType'] == 'HRRR')]

plot_data["FlowError_percent"] = pd.to_numeric(plot_data.FlowError_percent, errors='coerce')

plot_data["bin"] = pd.cut(plot_data["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])
medians = plot_data.groupby(["bin"])["FlowError_percent"].median().values
Q1 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.25)
Q3 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.75)
IQR = Q3 - Q1
IQR = IQR.to_numpy()

sns.boxplot(x="bin", y="FlowError_percent", data=plot_data, ax=axes[0, 0], color='#D2C295')
axes[0, 0].set_ylim([-200, 150])

nobs = plot_data["Wind_MPS_mean300"].value_counts(bins=[0, 1, 2, 3, 4, 5, 20]).sort_index(ascending=True).values
#nobs = [str(x) for x in nobs.tolist()]
#nobs = ["n: " + i for i in nobs]

# Add it to the plot
pos = range(len(nobs))
for tick, label in zip(pos, axes[0, 0].get_xticklabels()):
    axes[0, 0].text(pos[tick],
                 -175,
                 "{0} {1:.0f}".format('n: ', nobs[tick]) + '\n' + \
		 "{0} {1:.0f}".format('Q2: ', medians[tick]) + '\n' + \
		 "{0} {1:.0f}".format('IQR: ', IQR[tick]) + '\n',
                 horizontalalignment='center',
                 size='x-small',
                 color='k',
                 weight='normal',
		 verticalalignment = 'bottom')

# Unblinding stage 1 - NAM12

plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == (1)) & (
        matchedDF_Bridger['tc_Classification'] == 'TP') & (
                                      matchedDF_Bridger['WindType'] == 'NAM12')]

plot_data["FlowError_percent"] = pd.to_numeric(plot_data.FlowError_percent, errors='coerce')

plot_data["bin"] = pd.cut(plot_data["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])
medians = plot_data.groupby(["bin"])["FlowError_percent"].median().values
Q1 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.25)
Q3 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.75)
IQR = Q3 - Q1
IQR = IQR.to_numpy()

sns.boxplot(x="bin", y="FlowError_percent", data=plot_data, ax=axes[0, 1], color='#D2C295')
axes[0, 1].set_ylim([-200, 150])

nobs = plot_data["Wind_MPS_mean300"].value_counts(bins=[0, 1, 2, 3, 4, 5, 20]).sort_index(ascending=True).values
#nobs = [str(x) for x in nobs.tolist()]
#nobs = ["n: " + i for i in nobs]

# Add it to the plot
pos = range(len(nobs))
for tick, label in zip(pos, axes[0, 1].get_xticklabels()):
    axes[0, 1].text(pos[tick],
                 -175,
                 "{0} {1:.0f}".format('n: ', nobs[tick]) + '\n' + \
		 "{0} {1:.0f}".format('Q2: ', medians[tick]) + '\n' + \
		 "{0} {1:.0f}".format('IQR: ', IQR[tick]) + '\n',
                 horizontalalignment='center',
                 size='x-small',
                 color='k',
                 weight='normal',
		 verticalalignment = 'bottom')

# Unblinding stage 2

plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 2) & (
        matchedDF_Bridger['tc_Classification'] == 'TP') & (
                                      matchedDF_Bridger['WindType'] == 'Sonic')]

plot_data["FlowError_percent"] = pd.to_numeric(plot_data.FlowError_percent, errors='coerce')

plot_data["bin"] = pd.cut(plot_data["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])
medians = plot_data.groupby(["bin"])["FlowError_percent"].median().values
Q1 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.25)
Q3 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.75)
IQR = Q3 - Q1
IQR = IQR.to_numpy()

sns.boxplot(x="bin", y="FlowError_percent", data=plot_data, ax=axes[1, 0], color='#D2C295')
axes[1, 0].set_ylim([-200, 150])

nobs = plot_data["Wind_MPS_mean300"].value_counts(bins=[0, 1, 2, 3, 4, 5, 20]).sort_index(ascending=True).values
#nobs = [str(x) for x in nobs.tolist()]
#nobs = ["n: " + i for i in nobs]

# Add it to the plot
pos = range(len(nobs))
for tick, label in zip(pos, axes[1, 0].get_xticklabels()):
    axes[1, 0].text(pos[tick],
                 -175,
                 "{0} {1:.0f}".format('n: ', nobs[tick]) + '\n' + \
		 "{0} {1:.0f}".format('Q2: ', medians[tick]) + '\n' + \
		 "{0} {1:.0f}".format('IQR: ', IQR[tick]) + '\n',
                 horizontalalignment='center',
                 size='x-small',
                 color='k',
                 weight='normal',
		 verticalalignment = 'bottom')

# Unblinding stage 3

plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 3) & (
        matchedDF_Bridger['tc_Classification'] == 'TP') & (
                                      matchedDF_Bridger['WindType'] == 'Sonic')]

plot_data["FlowError_percent"] = pd.to_numeric(plot_data.FlowError_percent, errors='coerce')

plot_data["bin"] = pd.cut(plot_data["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])
medians = plot_data.groupby(["bin"])["FlowError_percent"].median().values
Q1 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.25)
Q3 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.75)
IQR = Q3 - Q1
IQR = IQR.to_numpy()

sns.boxplot(x="bin", y="FlowError_percent", data=plot_data, ax=axes[1, 1], color='#D2C295')
axes[1, 1].set_ylim([-200, 150])

nobs = plot_data["Wind_MPS_mean300"].value_counts(bins=[0, 1, 2, 3, 4, 5, 20]).sort_index(ascending=True).values
#nobs = [str(x) for x in nobs.tolist()]
#nobs = ["n: " + i for i in nobs]

# Add it to the plot
pos = range(len(nobs))
for tick, label in zip(pos, axes[1, 1].get_xticklabels()):
    axes[1, 1].text(pos[tick],
                 -175,
                 "{0} {1:.0f}".format('n: ', nobs[tick]) + '\n' + \
		 "{0} {1:.0f}".format('Q2: ', medians[tick]) + '\n' + \
		 "{0} {1:.0f}".format('IQR: ', IQR[tick]) + '\n',
                 horizontalalignment='center',
                 size='x-small',
                 color='k',
                 weight='normal',
		 verticalalignment = 'bottom')

axes[0, 0].set_xlabel('', fontsize=6)
axes[0, 1].set_xlabel('', fontsize=6)
axes[1, 0].set_xlabel('', fontsize=6)
axes[1, 1].set_xlabel('', fontsize=6)
axes[0, 1].set_ylabel('', fontsize=6)
axes[1, 1].set_ylabel('', fontsize=6)
axes[1, 0].set_ylabel('Quantification Error [%]', fontsize=12)
axes[1, 0].set_xlabel('Bin - 5 minute wind speed [mps]', fontsize=12)
axes[1, 0].tick_params(labelsize=12)

axes[0, 0].set_xticks(range(len(nobs)))
axes[0, 1].set_xticks(range(len(nobs)))
axes[1, 0].set_xticks(range(len(nobs)))
axes[1, 1].set_xticks(range(len(nobs)))
axes[0, 0].set_xticklabels(['', '', '', '', '', ''])
axes[0, 1].set_xticklabels(['', '', '', '', '', ''])
axes[1, 0].set_xticklabels(['0-1', '1-2', '2-3', '3-4', '4-5', '5+'])
axes[1, 1].set_xticklabels(['0-1', '1-2', '2-3', '3-4', '4-5', '5+'])

plt.rc('font', **font)

plt.savefig('Bridger_Boxwhisker_23822.svg')
plt.close()

## --------------------------GHGSAT-AV BOX AND WHISKER---------------------------------------------------------##


plt.subplots_adjust(hspace=0.5)
fig, axes = plt.subplots(3, 1, figsize=(10, 6), facecolor='w', edgecolor='k')

for i in range(3):

    plot_data = matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == (i + 1)) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]
            
    plot_data["FlowError_percent"] = pd.to_numeric(plot_data.FlowError_percent, errors='coerce')

    plot_data["bin"] = pd.cut(plot_data["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])
    medians = plot_data.groupby(["bin"])["FlowError_percent"].median().values
    Q1, Q3 = plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.25), plot_data.groupby(["bin"])["FlowError_percent"].quantile(0.75)
    IQR = Q3 - Q1
    IQR = IQR.to_numpy()	

    sns.boxplot(x="bin", y="FlowError_percent", data=plot_data, ax=axes[i], color='#8c1515')
    axes[i].set_ylim([-200, 200])

    nobs = plot_data["Wind_MPS_mean300"].value_counts(bins=[0, 1, 2, 3, 4, 5, 20]).sort_index(ascending=True).values
    #nobs = [str(x) for x in nobs.tolist()]
    #nobs = ["n: " + i for i in nobs]

    # Add it to the plot
    pos = range(len(nobs))
    for tick, label in zip(pos, axes[i].get_xticklabels()):
        axes[i].text(pos[tick],
                     -225,
                     "{0} {1:.2f}".format('n: ', nobs[tick]) + '\n' + \
		     "{0} {1:.2f}".format('Q2: ', medians[tick]) + '\n' + \
		     "{0} {1:.2f}".format('IQR: ', IQR[tick]) + '\n',
                     horizontalalignment='center',
                     size='x-small',
                     color='k',
                     weight='normal',
		     verticalalignment = 'bottom')

axes[0].set_xlabel('', fontsize=6)
axes[1].set_xlabel('', fontsize=6)
axes[1].set_ylabel('Quantification Error [%]', fontsize=12)
axes[2].set_xlabel('Bin - 5 minute wind speed [mps]', fontsize=12)
axes[2].tick_params(labelsize=12)

axes[0].set_xticks(range(len(nobs)))
axes[1].set_xticks(range(len(nobs)))
axes[2].set_xticks(range(len(nobs)))
axes[0].set_xticklabels(['', '', '', '', '', ''])
axes[1].set_xticklabels(['', '', '', '', '', ''])
axes[2].set_xticklabels(['0-1', '1-2', '2-3', '3-4', '4-5', '5+'])

plt.rc('font', **font)

plt.savefig('GHGSat_Boxwhisker_23822.svg')
plt.close()

