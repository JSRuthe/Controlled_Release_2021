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
fig, axes = plt.subplots(3,4, figsize=(10, 6), facecolor='w', edgecolor='k')
## CARBON MAPPER - PARITY
matchedDF_CarbonMapper['FacilityEmissionRateUpper'] = matchedDF_CarbonMapper['FacilityEmissionRateUpper'].replace('#VALUE!',np.NaN)
matchedDF_CarbonMapper['FacilityEmissionRateLower'] = matchedDF_CarbonMapper['FacilityEmissionRateLower'].replace('#VALUE!',np.NaN)

regress_stats = []
cols = ['n','R2','slope','intercept','Fstat','p-value']
for i in range(4):
    for j in range(3):
        if i == 0:
            if j == 0:
                plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == (j + 1)) & 
                                            (matchedDF_Bridger['tc_Classification'] == 'TP') & 
                                            (matchedDF_Bridger['WindType'] == 'HRRR')]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i,j], plot_data, 'Bridger', plot_color = '#D2C295')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('Bridger stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)
            else:
                plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == (j + 1)) & (matchedDF_Bridger['tc_Classification'] == 'TP')]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i,j], plot_data, 'Bridger', plot_color = '#8c1515')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('Bridger stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)
        elif i ==1:
            if j == 0:
                plot_data = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == (j + 1)) & 
                                            (matchedDF_Bridger['tc_Classification'] == 'TP') & 
                                            (matchedDF_Bridger['WindType'] == 'NAM12')]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i-1,j], plot_data, 'Bridger', plot_color = '#53284f')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('Bridger (NAM12) stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)
        elif i == 2:
            if j == 0:
                plot_data = matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == (j + 1)) & 
                            (matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50)]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i-1,j], plot_data, 'CarbonMapper', plot_color = '#D2C295')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('CM stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)
            else:
                plot_data = matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == (j + 1)) & 
                            (matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50)]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i-1,j], plot_data, 'CarbonMapper', plot_color = '#8c1515')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('CM stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)
        elif i == 3:	 
            if j == 0:
                plot_data = matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == (j + 1)) & 
                            (matchedDF_GHGSat['tc_Classification'] == 'TP')]
                            #(matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i-1,j], plot_data, 'GHGSat', plot_color = '#175e54')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('GHGSat stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)
            else:
                plot_data = matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == (j + 1)) & 
                            (matchedDF_GHGSat['tc_Classification'] == 'TP')]
                            #(matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]
                ax, n, r_value,slope,intercept, f, p = parity_plot(axes[i-1,j], plot_data, 'GHGSat', plot_color = '#8c1515')
                x = plot_data['cr_kgh_CH4_mean60'].values
                y = plot_data['FacilityEmissionRate'].fillna(0).values
                #f, p = f_test(x, y)
                print('GHGSat stage ', j,' F statistic = ',f)
                print('Best fit $R^{2}$ = %0.2f \n$y$ = %0.2f$x+$%0.2f \n n = %0.2f' % (r_value**2,slope,intercept,n))
                regress_add = [n, r_value**2,slope, intercept,f,p]
                regress_stats.append(regress_add)

dfRegressStats = pd.DataFrame(regress_stats, columns=cols)
#plt.savefig('Allteams_parity_230923_95CI_noint.svg', dpi = 300)

x = 1
