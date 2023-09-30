
import os.path
cwd = os.getcwd()
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

import seaborn as sns

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


csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_Bridger_23822.csv')
matchedDF_Bridger = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_GHGSat_23822.csv')
matchedDF_GHGSat = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_CarbonMapper_23822.csv')
matchedDF_CarbonMapper = pd.read_csv(csvPath)



cwd = os.getcwd()
plt.ion()
plt.subplots_adjust(hspace = 0.5)
fig, axes = plt.subplots(2,6, figsize=(10, 6), facecolor='w', edgecolor='k', gridspec_kw={'width_ratios': [1,2,4,1,2,4], 'height_ratios': [1,1]})
plt.rc('font', **font)

## PLOTTING - BRIDGER
# Subplot 1
# all data
Br_bar_dat = matchedDF_Bridger.loc[
    ((matchedDF_Bridger['UnblindingStage'] == 1) &
    (matchedDF_Bridger['WindType'] == 'HRRR') &
    (matchedDF_Bridger['tc_Classification'] == 'TN')) |
    ((matchedDF_Bridger['UnblindingStage'] == 1) &
    (matchedDF_Bridger['WindType'] == 'HRRR') &
    (matchedDF_Bridger['tc_Classification'] == 'FP')), 'tc_Classification']

Br_freq = Br_bar_dat.value_counts(normalize=False)

pd.DataFrame(Br_freq).T.plot.bar(stacked = True, ax = axes[0,0], legend = False, 
                                    color = ['#8c1515','#8c1515'], edgecolor='none', alpha = 0.5, linewidth=1.2)

# Round 3

Br_bar_dat = matchedDF_Bridger.loc[
    ((matchedDF_Bridger['UnblindingStage'] == 1) &
    (matchedDF_Bridger['WindType'] == 'HRRR') &
    (matchedDF_Bridger['Round 3 test set'] == 1) &
    (matchedDF_Bridger['tc_Classification'] == 'TN')) |
    ((matchedDF_Bridger['UnblindingStage'] == 1) &
    (matchedDF_Bridger['WindType'] == 'HRRR') &
    (matchedDF_Bridger['Round 3 test set'] == 1) &
    (matchedDF_Bridger['tc_Classification'] == 'FP')), 'tc_Classification']

Br_freq = Br_bar_dat.value_counts(normalize=False)

#pd.DataFrame(Br_freq).T.plot.bar(stacked = True, ax = axes[0,0], legend = False, 
#                                    color = ['#D2C295','#D2C295'], edgecolor='none', alpha = 0.5, linewidth=1.2)

axes[0,0].set_ylim([0, 35])
axes[0,0].set_yticks([0, 10, 20, 30, 35])
axes[0,0].set(xticklabels=[])
axes[0,0].set_xticks([])
# for minor ticks
axes[0,0].set_xticks([], minor=True)





# Subplot 2

## all data
Br_histo_dat = [matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'TP') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'FN') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAL') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAQ') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_MIS') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_Zero') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30']]
                #matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NE') &
                #                           (matchedDF_Bridger['UnblindingStage'] == 1) &
                #                           (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                #                           'cr_kgh_CH4_mean30'],
                #matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NS') &
                #                           (matchedDF_Bridger['UnblindingStage'] == 1) &
                #                           (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                #                           'cr_kgh_CH4_mean30']]

axes[0,1].hist(Br_histo_dat, bins = range(0,50+5,5), stacked=True, density = False,
                color=['#8c1515','#8c1515','#8c1515', '#8c1515', '#8c1515', '#8c1515'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)



# Round 3

Br_histo_dat = [matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'TP') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'FN') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAL') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &					     
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAQ') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_MIS') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NE') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NS') &
                                            (matchedDF_Bridger['UnblindingStage'] == 1) &
                            (matchedDF_Bridger['Round 3 test set'] == 1) &
                                            (matchedDF_Bridger['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30']]

axes[0,1].hist(Br_histo_dat, bins = range(0,50+5,5), stacked=True, density = False,
                color=['#D2C295','#D2C295','#D2C295', '#D2C295', '#D2C295', '#D2C295', '#D2C295'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)



axes[0,1].set_ylim([0, 35])
axes[0,1].set_xlim([0, 50])
axes[0,1].set(yticklabels=[])
axes[0,1].set_xticks([5, 25, 50])

# Subplot 3

#all data
Br_histo_dat = [matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'TP') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'FN') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAL') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAQ') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_MIS') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30']]
                #matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NE') &
                #(matchedDF_Bridger['UnblindingStage'] == 1) &
                #(matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                #(matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                #matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NS') &
                #(matchedDF_Bridger['UnblindingStage'] == 1) &
                #(matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                #(matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30']]

(n_Bridger, bins, patches) = axes[0,2].hist(Br_histo_dat, bins = range(50,1550+50,50), stacked=True, density = False, 
                color=['#8c1515','#8c1515','#8c1515', '#8c1515', '#8c1515'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)

# round 3
Br_histo_dat = [matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'TP') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'FN') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAL') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_FAQ') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER_MIS') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NE') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NS') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
            (matchedDF_Bridger['Round 3 test set'] == 1) &
                (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 50) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30']]

(n_Bridger, bins, patches) = axes[0,2].hist(Br_histo_dat, bins = range(50,1550+50,50), stacked=True, density = False, 
            color=['#D2C295','#D2C295','#D2C295', '#D2C295', '#D2C295', '#D2C295', '#D2C295'],
            edgecolor='none', alpha = 0.5, linewidth=1.2)



axes[0,2].set_ylim([0, 35])
axes[0,2].set_xlim([50, 1500])
axes[0,2].set(yticklabels=[])
axes[0,2].set_xticks([50, 500, 1000, 1500])

 ## PLOTTING - CARBON MAPPER
# Subplot 4

# All data
CM_bar_dat = matchedDF_CarbonMapper.loc[
    ((matchedDF_CarbonMapper['UnblindingStage'] == 1) &
    (matchedDF_CarbonMapper['tc_Classification'] == 'TN')) |
    ((matchedDF_CarbonMapper['UnblindingStage'] == 1) &
    (matchedDF_CarbonMapper['tc_Classification'] == 'FP')), 'tc_Classification']

CM_freq = CM_bar_dat.value_counts(normalize=False)

#axes[0,3].bar(1, CM_freq.values,color = '#999999',
#            edgecolor='black', linewidth=1.2)
pd.DataFrame(CM_freq).T.plot.bar(stacked = True, ax = axes[0,3], legend = False, 
                                 color = ['#8c1515','#8c1515'], edgecolor='none', alpha = 0.5, linewidth=1.2)

CM_bar_dat = matchedDF_CarbonMapper.loc[
    ((matchedDF_CarbonMapper['UnblindingStage'] == 1) &
    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
    (matchedDF_CarbonMapper['tc_Classification'] == 'TN')) |
    ((matchedDF_CarbonMapper['UnblindingStage'] == 1) &
    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
    (matchedDF_CarbonMapper['tc_Classification'] == 'FP')), 'tc_Classification']

CM_freq = CM_bar_dat.value_counts(normalize=False)

#axes[0,3].bar(1, CM_freq.values,color = '#999999',
#            edgecolor='black', linewidth=1.2)
#pd.DataFrame(CM_freq).T.plot.bar(stacked = True, ax = axes[0,3], legend = False, 
#                                 color = ['#D2C295','#D2C295'], edgecolor='none', alpha = 0.5, linewidth=1.2)

axes[0,3].set_ylim([0, 35])
axes[0,3].plot
axes[0,3].set(yticklabels=[])
axes[0,3].set(xticklabels=[])
axes[0,3].set_xticks([])
# for minor ticks
axes[0,3].set_xticks([], minor=True)

# Subplot 5

# All data

CM_histo_dat = [matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'FN') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAL') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAQ') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'ER_MIS') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30']]
                #matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NE') &
                #                           (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                #                           (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                #                           'cr_kgh_CH4_mean30'],
                #matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NS') &
                #                           (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                #                           (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                #                           'cr_kgh_CH4_mean30']]
axes[0,4].hist(CM_histo_dat, bins = range(0,50+5,5), stacked=True, density = False, 
               color=['#8c1515','#8c1515','#8c1515', '#8c1515', '#8c1515'], edgecolor='none', alpha = 0.5, linewidth=1.2)

# Round 3

CM_histo_dat = [matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'FN') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAL') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAQ') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'ER_MIS') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NE') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NS') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                            (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30']]
axes[0,4].hist(CM_histo_dat, bins = range(0,50+5,5), stacked=True, density = False, 
               color=['#D2C295','#D2C295','#D2C295', '#D2C295', '#D2C295', '#D2C295', '#D2C295'], edgecolor='none', alpha = 0.5, linewidth=1.2)


axes[0,4].set_ylim([0, 35])
axes[0,4].set_xlim([0, 50])
axes[0,4].set(yticklabels=[])
axes[0,4].set_xticks([5, 25, 50])

# Subplot 6

# All data
CM_histo_dat = [matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'FN') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                    (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAL') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                    (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAQ') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                    (matchedDF_CarbonMapper['tc_Classification'] == 'ER_MIS') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30']]
                #matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NE') &
                #                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                #                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                #matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NS') &
                #                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30']]

(n_CM, bins, patches) = axes[0,5].hist(CM_histo_dat, bins = range(50,1550+50,50), stacked=True, density = False, 
                                       color=['#8c1515','#8c1515','#8c1515', '#8c1515', '#8c1515'], edgecolor='none', alpha = 0.5, linewidth=1.2)

# Round 3

CM_histo_dat = [matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'FN') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                    (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAL') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                    (matchedDF_CarbonMapper['tc_Classification'] == 'ER_FAQ') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[
                                    (matchedDF_CarbonMapper['tc_Classification'] == 'ER_MIS') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NE') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NS') &
                    (matchedDF_CarbonMapper['Round 3 test set'] == 1) &
                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30']]

(n_CM, bins, patches) = axes[0,5].hist(CM_histo_dat, bins = range(50,1550+50,50), stacked=True, density = False, 
                                       color=['#D2C295','#D2C295','#D2C295', '#D2C295', '#D2C295', '#D2C295', '#D2C295'], edgecolor='none', alpha = 0.5, linewidth=1.2)

axes[0,5].set_ylim([0, 35])
axes[0,5].set_xlim([50, 1550])
axes[0,5].set(yticklabels=[])
axes[0,5].set_xticks([50, 500, 1000, 1500])

plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
plt.show()
plt.savefig('All_histo_23822_pt1.svg', dpi=300)


plt.ion()

fig, axes = plt.subplots(2,3, figsize=(10, 6), facecolor='w', edgecolor='k', gridspec_kw={'width_ratios': [1,4,22], 'height_ratios': [1,1]})
plt.rc('font', **font)



# Subplot 7

GHGS_bar_dat = matchedDF_GHGSat.loc[
    ((matchedDF_GHGSat['UnblindingStage'] == 1) &
    (matchedDF_GHGSat['tc_Classification'] == 'TN')) |
    ((matchedDF_GHGSat['UnblindingStage'] == 1) &
    (matchedDF_GHGSat['tc_Classification'] == 'ER_FAL')  &
    (matchedDF_GHGSat['cr_kgh_CH4_mean30'] == 0)) |
    ((matchedDF_GHGSat['UnblindingStage'] == 1) &
    (matchedDF_GHGSat['tc_Classification'] == 'ER_FAQ')  &
    (matchedDF_GHGSat['cr_kgh_CH4_mean30'] == 0)) |
    ((matchedDF_GHGSat['UnblindingStage'] == 1) &
    (matchedDF_GHGSat['tc_Classification'] == 'ER_MIS')  &
    (matchedDF_GHGSat['cr_kgh_CH4_mean30'] == 0)), 'tc_Classification']

GHGS_freq = GHGS_bar_dat.value_counts(normalize=False)

#axes[1,0].bar(1, GHGS_freq.values,color = ['#999999','#59B3A9', '#A6B168'],
#            edgecolor='black', linewidth=1.2, stacked = True)
pd.DataFrame(GHGS_freq).T.plot.bar(stacked = True, ax = axes[1,0], legend = False, 
                                   color = ['#8c1515','#8c1515', '#8c1515'], edgecolor='none', alpha = 0.5, linewidth=1.2)


axes[1,0].set_ylim([0, 35])
axes[1,0].set_yticks([0, 10, 20, 30, 35])
axes[1,0].set(xticklabels=[])
axes[1,0].set_xticks([])
# for minor ticks
axes[1,0].set_xticks([], minor=True)

# Subplot 8

# All data
GHGS_histo_dat = [matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'TP') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'FN') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[
                                            (matchedDF_GHGSat['tc_Classification'] == 'ER_FAL') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50) &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[
                                            (matchedDF_GHGSat['tc_Classification'] == 'ER_FAQ') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50) &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[
                                            (matchedDF_GHGSat['tc_Classification'] == 'ER_MIS') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50) &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0),
                                            'cr_kgh_CH4_mean30']]
                #matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'NE') &
                #                           (matchedDF_GHGSat['UnblindingStage'] == 1) &
                #                           (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                #                           'cr_kgh_CH4_mean30'],
                #matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'NS') &
                #                           (matchedDF_GHGSat['UnblindingStage'] == 1) &
                #                           (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                #                           'cr_kgh_CH4_mean30']]

axes[1,1].hist(GHGS_histo_dat, bins = range(0,50+5,5), stacked=True, density = False,
                color=['#8c1515','#8c1515','#8c1515', '#8c1515', '#8c1515'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)

# Round 3 only

GHGS_histo_dat = [matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'TP') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'FN') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[
                                            (matchedDF_GHGSat['tc_Classification'] == 'ER_FAL') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50) &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[
                                            (matchedDF_GHGSat['tc_Classification'] == 'ER_FAQ') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50) &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[
                                            (matchedDF_GHGSat['tc_Classification'] == 'ER_MIS') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50) &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'NE') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'NS') &
                                            (matchedDF_GHGSat['UnblindingStage'] == 1) &
                            (matchedDF_GHGSat['Round 3 test set'] == 1) &
                                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 50),
                                            'cr_kgh_CH4_mean30']]

axes[1,1].hist(GHGS_histo_dat, bins = range(0,50+5,5), stacked=True, density = False,
                color=['#D2C295','#D2C295','#D2C295', '#D2C295', '#D2C295', '#D2C295', '#D2C295'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)


axes[1,1].set_ylim([0, 35])
axes[1,1].set_xlim([0, 50])
axes[1,1].set(yticklabels=[])
axes[1,1].set_xticks([5, 25, 50])

# Subplot 9

# All data
GHGS_histo_dat = [matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'TP') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'FN') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'ER_FAL') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'ER_FAQ') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'ER_MIS') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30']]
                    #matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'NE') &
                    #(matchedDF_GHGSat['UnblindingStage'] == 1) &
                    #(matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30']]

(n_GHGSat, bins, patches) = axes[1,2].hist(GHGS_histo_dat, bins = range(50,7550+100,100), stacked=True, density = False,
                color=['#8c1515','#8c1515','#8c1515', '#8c1515', '#8c1515'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)

# Round 3 data
GHGS_histo_dat = [matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'TP') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                (matchedDF_GHGSat['Round 3 test set'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'FN') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                (matchedDF_GHGSat['Round 3 test set'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'ER_FAL') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                (matchedDF_GHGSat['Round 3 test set'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'ER_FAQ') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                (matchedDF_GHGSat['Round 3 test set'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'ER_MIS') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                (matchedDF_GHGSat['Round 3 test set'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30'],
                    matchedDF_GHGSat.loc[(matchedDF_GHGSat['tc_Classification'] == 'NE') &
                        (matchedDF_GHGSat['UnblindingStage'] == 1) &
                (matchedDF_GHGSat['Round 3 test set'] == 1) &
                        (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 50), 'cr_kgh_CH4_mean30']]

(n_GHGSat, bins, patches) = axes[1,2].hist(GHGS_histo_dat, bins = range(50,7550+100,100), stacked=True, density = False,
                color=['#D2C295','#D2C295','#D2C295', '#D2C295', '#D2C295', '#D2C295'],
                edgecolor='none', alpha = 0.5, linewidth=1.2)


axes[1,2].set_ylim([0, 35])
axes[1,2].set_xlim([50, 7550])
axes[1,2].set_xticks([50, 2000, 4000, 6000, 7500])

axes[1,2].set(yticklabels=[])

plt.subplots_adjust(hspace = 0.2, wspace = 0.1)
plt.show()
plt.savefig('All_histo_pt2_23822.svg', dpi=300)
