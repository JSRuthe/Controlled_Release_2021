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


plt.subplots_adjust(hspace = 0.5)
fig, axes = plt.subplots(2,3, figsize=(10, 6), facecolor='w', edgecolor='k')
plt.rc('font', **font)

matchedDF_CarbonMapper['FacilityEmissionRateUpper'] = matchedDF_CarbonMapper['FacilityEmissionRateUpper'].replace('#VALUE!',np.NaN)
matchedDF_CarbonMapper['FacilityEmissionRateLower'] = matchedDF_CarbonMapper['FacilityEmissionRateLower'].replace('#VALUE!',np.NaN)

for i in range(3):
        if i == 0:
                plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == (i + 1)) & 
                                            (matchedDF_Bridger['tc_Classification'] == 'TP') & 
                                            (matchedDF_Bridger['WindType'] == 'HRRR')]

                x = plot_data['cr_kgh_CH4_mean90'].values
                y = plot_data['FlowError_kgh'].fillna(0).values 

                axes[0,i].scatter(x,y,s=10, color = '#8c1515', alpha = 0.2)

                y = plot_data['FlowError_percent'].fillna(0).values 
                axes[1,i].scatter(x,y,s=10, color = '#8c1515', alpha = 0.2)

                axes[0,i].set_xlim([0,2000])
                axes[0,i].set_ylim([-1000,1000])
                #axes[0,i].set_yticks([0, 10, 20, 30, 35])
                axes[0,i].set(xticklabels=[])
                #axes[0,i].set_xticks([])
                # for minor ticks
                #axes[0,i].set_xticks([], minor=True)	

                axes[1,i].set_xlim([0,2000])
                axes[1,i].set_ylim([-100,200])
        

        elif i ==1:
                plot_data = matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == (i + 1)) & 
                                            (matchedDF_CarbonMapper['tc_Classification'] == 'TP')]

                x = plot_data['cr_kgh_CH4_mean90'].values
                y = plot_data['FlowError_kgh'].fillna(0).values 

                axes[0,i].scatter(x,y,s=10, color = '#8c1515', alpha = 0.2)

                y = plot_data['FlowError_percent'].fillna(0).values 
                axes[1,i].scatter(x,y,s=10, color = '#8c1515', alpha = 0.2)

                axes[0,i].set_xlim([0,2000])
                axes[0,i].set_ylim([-1000,1000])
                #axes[0,i].set_yticks([0, 10, 20, 30, 35])
                axes[0,i].set(yticklabels=[])
                #axes[0,i].set_yticks([])
                # for minor ticks
                #axes[0,i].set_yticks([], minor=True)	

                axes[0,i].set(xticklabels=[])
                #axes[0,i].set_xticks([])
                # for minor ticks
                #axes[0,i].set_xticks([], minor=True)	

                axes[1,i].set_xlim([0,2000])
                axes[1,i].set_ylim([-100,200])
                axes[1,i].set(yticklabels=[])
                #axes[1,i].set_yticks([])
                # for minor ticks
                #axes[1,i].set_yticks([], minor=True)	

        elif i == 2:
                plot_data = matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == (i + 1)) & 
                            (matchedDF_GHGSat['tc_Classification'] == 'TP') &
                            (matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]

                x = plot_data['cr_kgh_CH4_mean90'].values
                y = plot_data['FlowError_kgh'].fillna(0).values 

                axes[0,i].scatter(x,y,s=10, color = '#8c1515', alpha = 0.2)

                y = plot_data['FlowError_percent'].fillna(0).values 
                axes[1,i].scatter(x,y,s=10, color = '#8c1515', alpha = 0.2)

                axes[0,i].set_xlim([0,2000])
                axes[0,i].set_ylim([-1000,1000])
                #axes[0,i].set_yticks([0, 10, 20, 30, 35])
                axes[0,i].set(yticklabels=[])
                #axes[0,i].set_yticks([])
                # for minor ticks
                #axes[0,i].set_yticks([], minor=True)	

                axes[0,i].set(xticklabels=[])
                #axes[0,i].set_xticks([])
                # for minor ticks
                #axes[0,i].set_xticks([], minor=True)	

                axes[1,i].set_xlim([0,2000])
                axes[1,i].set_ylim([-100,200])
                axes[1,i].set(yticklabels=[])
                #axes[1,i].set_yticks([])
                # for minor ticks
                #axes[1,i].set_yticks([], minor=True)	

                
plt.savefig('Allteams_residuals_23825.svg', dpi = 300)