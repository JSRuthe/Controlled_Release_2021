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
from statsmodels.stats.weightstats import ttest_ind

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

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_Bridger_23822.csv')
matchedDF_Bridger = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_GHGSat_23822.csv')
matchedDF_GHGSat = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_CarbonMapper_23822.csv')
matchedDF_CarbonMapper = pd.read_csv(csvPath)


#################################################################################################################

#classification statistics Bridger  
df_counts_Bridger = matchedDF_Bridger.pivot_table( 
                        index='UnblindingStage', 
                        columns='tc_Classification', 
                        values = 'QC filter',
                        aggfunc = len)


#classification statistics Carbon Mapper
df_counts_CM = matchedDF_CarbonMapper.pivot_table( 
                        index='UnblindingStage', 
                        columns='tc_Classification', 
                        values = 'QC filter',
                        aggfunc = len)    

#classification statistics GHGSat
df_counts_GHGSat = matchedDF_GHGSat.pivot_table( 
                        index='UnblindingStage', 
                        columns='tc_Classification', 
                        values = 'PerformerExperimentID',
                        aggfunc = len)   

df_GHGSat_zero = pd.pivot_table(matchedDF_GHGSat[matchedDF_GHGSat['cr_kgh_CH4_mean30'] == 0],
                        index = 'UnblindingStage',
                        columns = 'tc_Classification',
                        values = 'PerformerExperimentID',
                        aggfunc = len)

df_GHGSat_nonzero = pd.pivot_table(matchedDF_GHGSat[matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0],
                        index = 'UnblindingStage',
                        columns = 'tc_Classification',
                        values = 'PerformerExperimentID',
                        aggfunc = len)


df_Bridger_rnd3 = pd.pivot_table(matchedDF_Bridger[(matchedDF_Bridger['WindType'] == 'HRRR') &
                                                (matchedDF_Bridger['Round 3 test set'] == 1)],
                        index = 'UnblindingStage',
                        columns = 'tc_Classification',
                        values = 'PerformerExperimentID',
                        aggfunc = len)


df_GHGSat_rnd3 = pd.pivot_table(matchedDF_GHGSat[matchedDF_GHGSat['Round 3 test set'] == 1],
                        index = 'UnblindingStage',
                        columns = 'tc_Classification',
                        values = 'PerformerExperimentID',
                        aggfunc = len)


df_CM_rnd3 = pd.pivot_table(matchedDF_CarbonMapper[matchedDF_CarbonMapper['Round 3 test set'] == 1],
                        index = 'UnblindingStage',
                        columns = 'tc_Classification',
                        values = 'PerformerExperimentID',
                        aggfunc = len)


# Range of releases Bridger

matchedDF_Bridger_filter = matchedDF_Bridger.drop(matchedDF_Bridger[(matchedDF_Bridger['tc_Classification'] == 'NE') |
                                                                (matchedDF_Bridger['tc_Classification'] == 'NS')
                                                ].index)

print('Bridger, min = ',
        matchedDF_Bridger_filter['cr_kgh_CH4_mean60'][(matchedDF_Bridger_filter['cr_kgh_CH4_mean60'] > 0.01) &
                                                (matchedDF_Bridger_filter['UnblindingStage'] == 1)].min())
print('Bridger, max = ',
        matchedDF_Bridger_filter['cr_kgh_CH4_mean60'][matchedDF_Bridger_filter['UnblindingStage'] == 1].max())
print('Bridger min detect = ',
        matchedDF_Bridger_filter['cr_kgh_CH4_mean60'][(matchedDF_Bridger_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_Bridger_filter['tc_Classification'] == 'TP')
                                                & (matchedDF_Bridger_filter['UnblindingStage'] == 1)].min())
count_nzero_Bridger = matchedDF_Bridger_filter['cr_kgh_CH4_mean60'][(matchedDF_Bridger_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_Bridger_filter['UnblindingStage'] == 1)].count()
count_gt100 = matchedDF_Bridger_filter['cr_kgh_CH4_mean60'][(matchedDF_Bridger_filter['cr_kgh_CH4_mean60'] > 100)
                                                & (matchedDF_Bridger_filter['UnblindingStage'] == 1)].count()
count_gt500_Bridger = matchedDF_Bridger_filter['cr_kgh_CH4_mean60'][(matchedDF_Bridger_filter['cr_kgh_CH4_mean60'] > 1000)
                                                & (matchedDF_Bridger_filter['UnblindingStage'] == 1)].count()


print('Bridger, frac > 100 = ', count_gt100/count_nzero_Bridger)

CI_data =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) &
                                                (matchedDF_Bridger['tc_Classification'] == 'TP') &
                                                (matchedDF_Bridger['WindType'] == 'HRRR')]

CI_data = CI_data['FlowError_percent'].to_numpy()
CI_error = np.percentile(CI_data,[2.5, 97.5])
print('Bridger, error (95% CI) = ', CI_error)
print('Bridger, error (mean) = ', np.mean(CI_data))

matchedDF_Bridger_filter = matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) & (matchedDF_Bridger['WindType'] == 'HRRR')]

matchedDF_Bridger_filter['Stanford_timestamp'] = pd.to_datetime(matchedDF_Bridger_filter['Stanford_timestamp'])
matchedDF_Bridger_filter['Time_diff'] = matchedDF_Bridger_filter[
'Stanford_timestamp'].diff().dropna().dt.total_seconds()

difference = matchedDF_Bridger_filter[
'Stanford_timestamp'].diff().dropna().dt.total_seconds()
difference = difference.reset_index(drop=True)

filter_average = matchedDF_Bridger_filter['Stanford_timestamp'].dt.date == matchedDF_Bridger_filter[
'Stanford_timestamp'].shift(-1).dt.date
filter_average = filter_average.reset_index(drop=True)


matchedDF_Bridger_filter['Time_diff'][
matchedDF_Bridger_filter['Stanford_timestamp'].dt.date != matchedDF_Bridger_filter[
'Stanford_timestamp'].shift(1).dt.date] = np.nan

print('Bridger, revisit average', np.mean(difference.loc[filter_average]))

matchedDF_Bridger_filter['Day'] = matchedDF_Bridger_filter['Stanford_timestamp'].dt.day

Bridger_daily = pd.pivot_table(matchedDF_Bridger_filter,
                                index='Day',
                                values=['Time_diff', 'Altitude (feet)'],
                                aggfunc={'Time_diff':('count', 'mean', 'sum'), 'Altitude (feet)': 'mean'})


# Range of releases Carbon Mapper

matchedDF_CarbonMapper_filter = matchedDF_CarbonMapper.drop(matchedDF_CarbonMapper[
                                                (matchedDF_CarbonMapper['tc_Classification'] == 'NE') |
                                                (matchedDF_CarbonMapper['tc_Classification'] == 'NS')
                                                ].index)

print('CM, min = ',
        matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].min())
print('CM, max = ',
        matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][matchedDF_CarbonMapper_filter['UnblindingStage'] == 1].max())
print('CM min detect = ',
        matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_CarbonMapper_filter['tc_Classification'] == 'TP')
                                                & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].min())
print('CM max FN = ',
        matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_CarbonMapper_filter['tc_Classification'] == 'FN')
                                                & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].max())
count_nzero_CM = matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].count()
count_gt100 = matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] > 100)
                                                & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].count()
count_gt500_CM = matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] > 1000)
                                                & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].count()

print('CM, frac > 100 = ', count_gt100/count_nzero_CM)

countTP_lt100 = matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'][(matchedDF_CarbonMapper_filter['cr_kgh_CH4_mean60'] < 100)
                                        & (matchedDF_CarbonMapper_filter['tc_Classification'] == 'TP')
                                        & (matchedDF_CarbonMapper_filter['UnblindingStage'] == 1)].count()
print('CM, # < 100 TP = ', countTP_lt100)
print('CM, frac < 100 TP', countTP_lt100 / (count_nzero_CM - count_gt100))

CI_data =  matchedDF_CarbonMapper_filter[(matchedDF_CarbonMapper_filter['UnblindingStage'] == 1) &
                                                (matchedDF_CarbonMapper_filter['tc_Classification'] == 'TP')]

CI_data = CI_data['FlowError_percent'].to_numpy()
CI_error = np.percentile(CI_data,[2.5, 97.5])
print('CM, error (95% CI) = ', CI_error)
print('CM, error (mean) = ', np.mean(CI_data))

matchedDF_CarbonMapper_filter = matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 1)]

matchedDF_CarbonMapper_filter['Stanford_timestamp'] = pd.to_datetime(matchedDF_CarbonMapper_filter['Stanford_timestamp'])

matchedDF_CarbonMapper_filter['Time_diff'] = matchedDF_CarbonMapper_filter[
'Stanford_timestamp'].diff().dropna().dt.total_seconds()

difference = matchedDF_CarbonMapper_filter[
'Stanford_timestamp'].diff().dropna().dt.total_seconds()
difference = difference.reset_index(drop=True)

filter_average = matchedDF_CarbonMapper_filter['Stanford_timestamp'].dt.date == matchedDF_CarbonMapper_filter[
'Stanford_timestamp'].shift(-1).dt.date
filter_average = filter_average.reset_index(drop=True)


matchedDF_CarbonMapper_filter['Time_diff'][
matchedDF_CarbonMapper_filter['Stanford_timestamp'].dt.date != matchedDF_CarbonMapper_filter[
'Stanford_timestamp'].shift(1).dt.date] = np.nan

print('Carbon Mapper, revisit average', np.mean(difference.loc[filter_average]))

matchedDF_CarbonMapper_filter['Day'] = matchedDF_CarbonMapper_filter['Stanford_timestamp'].dt.day

CarbonMapper_daily = pd.pivot_table(matchedDF_CarbonMapper_filter,
                                index='Day',
                                values=['Time_diff', 'Altitude (feet)'],
                                aggfunc={'Time_diff':('count', 'mean', 'sum'), 'Altitude (feet)': 'mean'})

# Range of releases GHGSat-AV

matchedDF_GHGSat_filter = matchedDF_GHGSat.drop(matchedDF_GHGSat[
                                                (matchedDF_GHGSat['tc_Classification'] == 'NE') |
                                                (matchedDF_GHGSat['tc_Classification'] == 'NS')
                                                ].index)

print('GHGSat, min = ',
        matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] > 0.01)
                                        & (matchedDF_GHGSat_filter['UnblindingStage'] == 1)].min())
print('GHGSat, max = ',
        matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][matchedDF_GHGSat_filter['UnblindingStage'] == 1].max())
print('GHGSat min detect = ',
        matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_GHGSat_filter['tc_Classification'] == 'TP')].min())
print('GHGSat max FN = ',
        matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_GHGSat_filter['tc_Classification'] == 'FN')
                                                & (matchedDF_GHGSat_filter['UnblindingStage'] == 1)].max())
count_nzero_GHGSat = matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] > 0.01)
                                                & (matchedDF_GHGSat_filter['UnblindingStage'] == 1)].count()
count_gt100 =matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] > 100)
                                                & (matchedDF_GHGSat_filter['UnblindingStage'] == 1)].count()
count_gt500_GHGSat =matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] > 1000)
                                                & (matchedDF_GHGSat_filter['UnblindingStage'] == 1)].count()

print('GHGSat, frac > 100 = ', count_gt100/count_nzero_GHGSat)
print('All, count > 1000 = ', (count_gt500_GHGSat + count_gt500_Bridger + count_gt500_CM)) #/(count_nzero_GHGSat + count_nzero_Bridger + count_nzero_CM))

countTP_lt100 = matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'][(matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] < 100)
                                        & (matchedDF_GHGSat_filter['tc_Classification'] == 'TP')
                                        & (matchedDF_GHGSat_filter['UnblindingStage'] == 1)].count()
print('GHGSat, # < 100 TP = ', countTP_lt100)
print('GHGSat, frac < 100 TP', countTP_lt100 / (count_nzero_GHGSat - count_gt100))

CI_data = matchedDF_GHGSat_filter[(matchedDF_GHGSat_filter['UnblindingStage'] == 1) &
                                        (matchedDF_GHGSat_filter['tc_Classification'] == 'TP')]

CI_data = CI_data['FlowError_percent'].to_numpy()
CI_error = np.percentile(CI_data, [2.5, 97.5])
print('GHGSat (all), error (95% CI) = ', CI_error)
print('GHGSat(all), error (mean) = ', np.mean(CI_data))

CI_data = matchedDF_GHGSat_filter[(matchedDF_GHGSat_filter['UnblindingStage'] == 1) &
                                        (matchedDF_GHGSat_filter['tc_Classification'] == 'TP') &
                                        (matchedDF_GHGSat_filter['cr_kgh_CH4_mean60'] < 2000)]

CI_data = CI_data['FlowError_percent'].to_numpy()
CI_error = np.percentile(CI_data, [2.5, 97.5])
print('GHGSat (<2000), error (95% CI) = ', CI_error)
print('GHGSat (<2000), error (mean) = ', np.mean(CI_data))

matchedDF_GHGSat_filter = matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1)]

matchedDF_GHGSat_filter['Stanford_timestamp'] = pd.to_datetime(matchedDF_GHGSat_filter['Stanford_timestamp'])

matchedDF_GHGSat_filter['Time_diff'] = matchedDF_GHGSat_filter['Stanford_timestamp'].diff().dropna().dt.total_seconds()
matchedDF_GHGSat_filter = matchedDF_GHGSat_filter.drop(matchedDF_GHGSat_filter[
                                                        (matchedDF_GHGSat_filter['PerformerExperimentID'] == '1496-1-115-657-856-28')].index)
difference = matchedDF_GHGSat_filter[
'Stanford_timestamp'].diff().dropna().dt.total_seconds()
difference = difference.reset_index(drop=True)

filter_average = matchedDF_GHGSat_filter['Stanford_timestamp'].dt.date == matchedDF_GHGSat_filter[
'Stanford_timestamp'].shift(-1).dt.date
filter_average = filter_average.reset_index(drop=True)

matchedDF_GHGSat_filter['Time_diff'][matchedDF_GHGSat_filter['Stanford_timestamp'].dt.date != matchedDF_GHGSat_filter[
'Stanford_timestamp'].shift(1).dt.date] = np.nan

print('GHGSat, revisit average', np.mean(difference.loc[filter_average]))

matchedDF_GHGSat_filter['Day'] = matchedDF_GHGSat_filter['Stanford_timestamp'].dt.day

GHGSat_daily = pd.pivot_table(matchedDF_GHGSat_filter,
                                index='Day',
                                values=['Time_diff', 'Altitude (feet)'],
                                aggfunc={'Time_diff':('count', 'mean', 'sum'), 'Altitude (feet)': 'mean'})

#################################################################################################################

## CARBON MAPPER - PARITY

matchedDF_CarbonMapper['FacilityEmissionRateUpper'] = matchedDF_CarbonMapper['FacilityEmissionRateUpper'].replace('#VALUE!',np.NaN)
matchedDF_CarbonMapper['FacilityEmissionRateLower'] = matchedDF_CarbonMapper['FacilityEmissionRateLower'].replace('#VALUE!',np.NaN)


CM_histo_dat = [matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'FN') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'ER') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NE') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1), 'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NS') &
                                    (matchedDF_CarbonMapper['UnblindingStage'] == 1), 'cr_kgh_CH4_mean30']]

CM_bar_dat = matchedDF_CarbonMapper.loc[
    ((matchedDF_CarbonMapper['UnblindingStage'] == 1) &
    (matchedDF_CarbonMapper['tc_Classification'] == 'TN')) |
    ((matchedDF_CarbonMapper['UnblindingStage'] == 1) &
    (matchedDF_CarbonMapper['tc_Classification'] == 'FP')), 'tc_Classification']

CM_freq = CM_bar_dat.value_counts(normalize=False)


CM_histo_dat = [matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 60),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'FN') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 60),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'ER') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 60),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NE') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 60),
                                            'cr_kgh_CH4_mean30'],
                matchedDF_CarbonMapper.loc[(matchedDF_CarbonMapper['tc_Classification'] == 'NS') &
                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                            (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] <= 60),
                                            'cr_kgh_CH4_mean30']]

    

## BRIDGER HISTOGRAM

Br_histo_dat = [matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'TP') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'FN') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'ER') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NE') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30'],
                matchedDF_Bridger.loc[(matchedDF_Bridger['tc_Classification'] == 'NS') &
                (matchedDF_Bridger['UnblindingStage'] == 1) &
                (matchedDF_Bridger['WindType'] == 'HRRR'), 'cr_kgh_CH4_mean30']]

Br_bar_dat = matchedDF_Bridger.loc[
    ((matchedDF_Bridger['UnblindingStage'] == 1) &
    (matchedDF_Bridger['WindType'] == 'HRRR') &
    (matchedDF_Bridger['tc_Classification'] == 'TN')) |
    ((matchedDF_Bridger['UnblindingStage'] == 1) &
    (matchedDF_Bridger['WindType'] == 'HRRR') &
    (matchedDF_Bridger['tc_Classification'] == 'FP')), 'tc_Classification']

Br_freq = Br_bar_dat.value_counts(normalize=False)



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

df_Bridger_rnd3_lvls = pd.pivot_table(matchedDF_Bridger[(matchedDF_Bridger['WindType'] == 'HRRR') &
                                                            (matchedDF_Bridger['tc_Classification'] == 'TP') &
                                                            (matchedDF_Bridger['UnblindingStage'] == 1)],
                                index = 'cr_start',
                                #columns='Round 3 test set',
                                values = 'cr_kgh_CH4_mean30',
                                aggfunc = ('count','mean'))

df_Bridger_rnd3_lvls_all = pd.pivot_table(matchedDF_Bridger[(matchedDF_Bridger['WindType'] == 'HRRR') &
                                                            (matchedDF_Bridger['UnblindingStage'] == 1)],
                                index = 'cr_start',
                                #columns='Round 3 test set',
                                values = 'cr_kgh_CH4_mean30',
                                aggfunc = ('count','mean'))

df_Bridger_rnd3_lvls['datetime_col'] = pd.to_datetime(df_Bridger_rnd3_lvls.index)
df_Bridger_rnd3_lvls['Time_diff_min'] =  df_Bridger_rnd3_lvls['datetime_col'].diff().dropna().dt.total_seconds()/60

df_GHGSat_rnd3_lvls = pd.pivot_table(matchedDF_GHGSat[(matchedDF_GHGSat['tc_Classification'] == 'TP') &
                                                            (matchedDF_GHGSat['UnblindingStage'] == 1)],
                                index = 'cr_start',
                                #columns='Round 3 test set',
                                values = 'cr_kgh_CH4_mean30',
                                aggfunc = ('count','mean'))

df_GHGSat_rnd3_lvls_all = pd.pivot_table(matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1)],
                                index = 'cr_start',
                                #columns='Round 3 test set',
                                values = 'cr_kgh_CH4_mean30',
                                aggfunc = ('count','mean'))

df_GHGSat_rnd3_lvls['datetime_col'] = pd.to_datetime(df_GHGSat_rnd3_lvls.index)
df_GHGSat_rnd3_lvls['Time_diff_min'] =  df_GHGSat_rnd3_lvls['datetime_col'].diff().dropna().dt.total_seconds()/60

df_CM_rnd3_lvls = pd.pivot_table(matchedDF_CarbonMapper[(matchedDF_CarbonMapper['tc_Classification'] == 'TP') &
                                                            (matchedDF_CarbonMapper['UnblindingStage'] == 1)],
                                index = 'cr_start',
                                #columns='Round 3 test set',
                                values = 'cr_kgh_CH4_mean30',
                                aggfunc = ('count','mean'))

df_CM_rnd3_lvls_all = pd.pivot_table(matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 1)],
                                index = 'cr_start',
                                #columns='Round 3 test set',
                                values = 'cr_kgh_CH4_mean30',
                                aggfunc = ('count','mean'))

df_CM_rnd3_lvls['datetime_col'] = pd.to_datetime(df_CM_rnd3_lvls.index)
df_CM_rnd3_lvls['Time_diff_min'] =  df_CM_rnd3_lvls['datetime_col'].diff().dropna().dt.total_seconds()/60

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

print('Bridger, Total levels = ',
          df_Bridger_rnd3_lvls['count'][(df_Bridger_rnd3_lvls['Time_diff_min'] < 720)].count())
print('Carbon Mapper, Total levels = ',
          df_CM_rnd3_lvls['count'][(df_CM_rnd3_lvls['Time_diff_min'] < 720)].count())
print('GHGSat, Total levels = ',
          df_GHGSat_rnd3_lvls['count'][(df_GHGSat_rnd3_lvls['Time_diff_min'] < 720)].count())

print('Bridger, Level held for, on average (min) = ',
          df_Bridger_rnd3_lvls['Time_diff_min'][(df_Bridger_rnd3_lvls['Time_diff_min'] < 720)].mean())
print('Carbon Mapper, Level held for, on average (min) = ',
          df_CM_rnd3_lvls['Time_diff_min'][(df_CM_rnd3_lvls['Time_diff_min'] < 720)].mean())
print('GHGSat, Level held for, on average (min) = ',
          df_GHGSat_rnd3_lvls['Time_diff_min'][(df_GHGSat_rnd3_lvls['Time_diff_min'] < 720)].mean())

#print('Bridger, Total levels rnd 3 = ',
#          df_Bridger_rnd3_lvls['count'][(df_Bridger_rnd3_lvls['Round 3 test set'] == 1) &
#                                        (df_Bridger_rnd3_lvls['Time_diff_min'] < 720)].count())
#print('Carbon Mapper, Total levels rnd 3 = ',
#          df_CM_rnd3_lvls['count'][(df_CM_rnd3_lvls['Round 3 test set'] == 1) &
#                                   (df_CM_rnd3_lvls['Time_diff_min'] < 720)].count())
#print('GHGSat, Total levels rnd 3 = ',
#          df_GHGSat_rnd3_lvls['count'][(df_GHGSat_rnd3_lvls['Round 3 test set'] == 1)
#                                       (df_GHGSat_rnd3_lvls['Time_diff_min'] < 720)].count())

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 









#classification statistics Bridger  
df_Bridger_zero = pd.pivot_table(matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) &
                                                    (matchedDF_Bridger['WindType'] == 'HRRR') & 
                                                   (matchedDF_Bridger['cr_kgh_CH4_mean30'] == 0)],
                            index = 'Round 3 test set',
                            values = 'PerformerExperimentID',
                            aggfunc = len)

df_Bridger_nonzero = pd.pivot_table(matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) &
                                                    (matchedDF_Bridger['WindType'] == 'HRRR') & 
                                                   (matchedDF_Bridger['cr_kgh_CH4_mean30'] > 0)],
                            index = 'Round 3 test set',
                            values = 'cr_kgh_CH4_mean30',
                            aggfunc = ('count','mean'))


#classification statistics Carbon Mapper
df_CM_zero = pd.pivot_table(matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] == 0)],
                            index = 'Round 3 test set',
                            values = 'PerformerExperimentID',
                            aggfunc = len)

df_CM_nonzero = pd.pivot_table(matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 1) &
                                                    (matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 0)],
                            index='Round 3 test set', 
                            values = 'cr_kgh_CH4_mean30',
                            aggfunc = ('count','mean'))  

#classification statistics GHGSat
df_GHGSat_zero = pd.pivot_table(matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1) &
                                                (matchedDF_GHGSat['cr_kgh_CH4_mean30'] == 0)],
                            index = 'Round 3 test set',
                            values = 'PerformerExperimentID',
                            aggfunc = len)

df_GHGSat_nonzero = pd.pivot_table(matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1) &
                                                    (matchedDF_GHGSat['cr_kgh_CH4_mean30'] > 0)],
                            index = 'Round 3 test set',
                            values = 'cr_kgh_CH4_mean30',
                            aggfunc = ('count','mean')) 


df_Bridger_rnd3 = pd.pivot_table(matchedDF_Bridger[(matchedDF_Bridger['WindType'] == 'HRRR') &
                                                    (matchedDF_Bridger['Round 3 test set'] == 1)],
                            index = 'UnblindingStage',
                            columns = 'tc_Classification',
                            values = 'cr_kgh_CH4_mean30',
                            aggfunc = ('count','mean'))


df_GHGSat_rnd3 = pd.pivot_table(matchedDF_GHGSat[matchedDF_GHGSat['Round 3 test set'] == 1],
                            index = 'UnblindingStage',
                            columns = 'tc_Classification',
                            values = 'cr_kgh_CH4_mean30',
                            aggfunc = ('count','mean'))


df_CM_rnd3 = pd.pivot_table(matchedDF_CarbonMapper[matchedDF_CarbonMapper['Round 3 test set'] == 1],
                            index = 'UnblindingStage',
                            columns = 'tc_Classification',
                            values = 'cr_kgh_CH4_mean30',
                            aggfunc = ('count','mean'))

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# CARBON MAPPER QUANT ERROR

df_CM_TP =  matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 1) & (
            matchedDF_CarbonMapper['tc_Classification'] == 'TP') & (
            matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50)]

df_CM_TP["FlowError_percent"] = pd.to_numeric(df_CM_TP.FlowError_percent, errors='coerce')

df_CM_TP["bin"] = pd.cut(df_CM_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1 Count"
        "R1 Mean error",
        "R1 P2.5 error",
        "R1 P50 error",
        "R1 P97.5 error"]

df_CM_wind1 = pd.DataFrame(columns = column_names)

df_CM_wind1["R1 Mean error"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].mean()
df_CM_wind1["R1 P2.5 error"], df_CM_wind1["R1 P50 error"], df_CM_wind1["R1 P97.5 error"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_CM_wind1.columns))[None, :], 
                  columns=df_CM_wind1.columns,
                  index=[10])

df_CM_wind1 = pd.concat([df_CM_wind1, df1])

df_CM_wind1.loc[10,"R1 Mean error"] = np.mean(df_CM_TP["FlowError_percent"])
df_CM_wind1.loc[10,"R1 P2.5 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.025)
df_CM_wind1.loc[10,"R1 P50 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.5)
df_CM_wind1.loc[10,"R1 P97.5 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.975)

df_CM_wind1["R1 Count"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].size()

df_CM_TP =  matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 2) & (
            matchedDF_CarbonMapper['tc_Classification'] == 'TP') & (
            matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50)]

df_CM_TP["FlowError_percent"] = pd.to_numeric(df_CM_TP.FlowError_percent, errors='coerce')

df_CM_TP["bin"] = pd.cut(df_CM_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R2 Count",
        "R2 Mean error",
        "R2 P2.5 error",
        "R2 P50 error",
        "R2 P97.5 error"]

df_CM_wind2 = pd.DataFrame(columns = column_names)

df_CM_wind2["R2 Mean error"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].mean()
df_CM_wind2["R2 P2.5 error"], df_CM_wind2["R2 P50 error"], df_CM_wind2["R2 P97.5 error"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df2 = pd.DataFrame(np.repeat(0, len(df_CM_wind2.columns))[None, :], 
                  columns=df_CM_wind2.columns,
                  index=[10])

df_CM_wind2 = pd.concat([df_CM_wind2, df2])

df_CM_wind2.loc[10,"R2 Mean error"] = np.mean(df_CM_TP["FlowError_percent"])
df_CM_wind2.loc[10,"R2 P2.5 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.025)
df_CM_wind2.loc[10,"R2 P50 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.5)
df_CM_wind2.loc[10,"R2 P97.5 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.975)

df_CM_wind2["R2 Count"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].size()

df_CM_TP =  matchedDF_CarbonMapper[(matchedDF_CarbonMapper['UnblindingStage'] == 3) & (
            matchedDF_CarbonMapper['tc_Classification'] == 'TP') & (
            matchedDF_CarbonMapper['cr_kgh_CH4_mean30'] > 50)]

df_CM_TP["FlowError_percent"] = pd.to_numeric(df_CM_TP.FlowError_percent, errors='coerce')

df_CM_TP["bin"] = pd.cut(df_CM_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R3 Count",
        "R3 Mean error",
        "R3 P2.5 error",
        "R3 P50 error",
        "R3 P97.5 error"]

df_CM_wind3 = pd.DataFrame(columns = column_names)

df_CM_wind3["R3 Mean error"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].mean()
df_CM_wind3["R3 P2.5 error"], df_CM_wind3["R3 P50 error"], df_CM_wind3["R3 P97.5 error"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_CM_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df3 = pd.DataFrame(np.repeat(0, len(df_CM_wind3.columns))[None, :], 
                  columns=df_CM_wind3.columns,
                  index=[10])

df_CM_wind3 = pd.concat([df_CM_wind3, df3])

df_CM_wind3.loc[10,"R3 Mean error"] = np.mean(df_CM_TP["FlowError_percent"])
df_CM_wind3.loc[10,"R3 P2.5 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.025)
df_CM_wind3.loc[10,"R3 P50 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.5)
df_CM_wind3.loc[10,"R3 P97.5 error"] = np.quantile(df_CM_TP["FlowError_percent"], 0.975)

df_CM_wind3["R2 Count"] = df_CM_TP.groupby(["bin"])["FlowError_percent"].size()


# BRIDGER QUANT ERROR

df_Bridger_TP =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) & (
            matchedDF_Bridger['tc_Classification'] == 'TP') & (
            matchedDF_Bridger['WindType'] == 'HRRR')]

df_Bridger_TP["FlowError_percent"] = pd.to_numeric(df_Bridger_TP.FlowError_percent, errors='coerce')

df_Bridger_TP["bin"] = pd.cut(df_Bridger_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1 Count",
        "R1 Mean error",
        "R1 P2.5 error",
        "R1 P50 error",
        "R1 P97.5 error"]

df_Bridger_wind1 = pd.DataFrame(columns = column_names)

df_Bridger_wind1["R1 Mean error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].mean()
df_Bridger_wind1["R1 P2.5 error"], df_Bridger_wind1["R1 P50 error"], df_Bridger_wind1["R1 P97.5 error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_Bridger_wind1.columns))[None, :], 
                  columns=df_Bridger_wind1.columns,
                  index=[10])

df_Bridger_wind1 = pd.concat([df_Bridger_wind1, df1])

df_Bridger_wind1.loc[10,"R1 Mean error"] = np.mean(df_Bridger_TP["FlowError_percent"])
df_Bridger_wind1.loc[10,"R1 P2.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.025)
df_Bridger_wind1.loc[10,"R1 P50 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.5)
df_Bridger_wind1.loc[10,"R1 P97.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.975)

df_Bridger_wind1["R1 Count"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].size()

df_Bridger_TP =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) & (
            matchedDF_Bridger['tc_Classification'] == 'TP') & (
            matchedDF_Bridger['WindType'] == 'NAM12')]

df_Bridger_TP["FlowError_percent"] = pd.to_numeric(df_Bridger_TP.FlowError_percent, errors='coerce')

df_Bridger_TP["bin"] = pd.cut(df_Bridger_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1_2 Count",
        "R1_2 Mean error",
        "R1_2 P2.5 error",
        "R1_2 P50 error",
        "R1_2 P97.5 error"]

df_Bridger_wind1_2 = pd.DataFrame(columns = column_names)

df_Bridger_wind1_2["R1_2 Mean error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].mean()
df_Bridger_wind1_2["R1_2 P2.5 error"], df_Bridger_wind1_2["R1_2 P50 error"], df_Bridger_wind1_2["R1_2 P97.5 error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_Bridger_wind1_2.columns))[None, :], 
                  columns=df_Bridger_wind1_2.columns,
                  index=[10])

df_Bridger_wind1_2 = pd.concat([df_Bridger_wind1_2, df1])

df_Bridger_wind1_2.loc[10,"R1_2 Mean error"] = np.mean(df_Bridger_TP["FlowError_percent"])
df_Bridger_wind1_2.loc[10,"R1_2 P2.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.025)
df_Bridger_wind1_2.loc[10,"R1_2 P50 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.5)
df_Bridger_wind1_2.loc[10,"R1_2 P97.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.975)

df_Bridger_wind1_2["R1_2 Count"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].size()

df_Bridger_TP =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 2) & (
            matchedDF_Bridger['tc_Classification'] == 'TP')]

df_Bridger_TP["FlowError_percent"] = pd.to_numeric(df_Bridger_TP.FlowError_percent, errors='coerce')

df_Bridger_TP["bin"] = pd.cut(df_Bridger_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R2 Count",
        "R2 Mean error",
        "R2 P2.5 error",
        "R2 P50 error",
        "R2 P97.5 error"]

df_Bridger_wind2 = pd.DataFrame(columns = column_names)

df_Bridger_wind2["R2 Mean error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].mean()
df_Bridger_wind2["R2 P2.5 error"], df_Bridger_wind2["R2 P50 error"], df_Bridger_wind2["R2 P97.5 error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df2 = pd.DataFrame(np.repeat(0, len(df_Bridger_wind2.columns))[None, :], 
                  columns=df_Bridger_wind2.columns,
                  index=[10])

df_Bridger_wind2 = pd.concat([df_Bridger_wind2, df2])

df_Bridger_wind2.loc[10,"R2 Mean error"] = np.mean(df_Bridger_TP["FlowError_percent"])
df_Bridger_wind2.loc[10,"R2 P2.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.025)
df_Bridger_wind2.loc[10,"R2 P50 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.5)
df_Bridger_wind2.loc[10,"R2 P97.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.975)

df_Bridger_wind2["R2 Count"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].size()

df_Bridger_TP =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 3) & (
            matchedDF_Bridger['tc_Classification'] == 'TP')]

df_Bridger_TP["FlowError_percent"] = pd.to_numeric(df_Bridger_TP.FlowError_percent, errors='coerce')

df_Bridger_TP["bin"] = pd.cut(df_Bridger_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R3 Count",
        "R3 Mean error",
        "R3 P2.5 error",
        "R3 P50 error",
        "R3 P97.5 error"]

df_Bridger_wind3 = pd.DataFrame(columns = column_names)

df_Bridger_wind3["R3 Mean error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].mean()
df_Bridger_wind3["R3 P2.5 error"], df_Bridger_wind3["R3 P50 error"], df_Bridger_wind3["R3 P97.5 error"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_Bridger_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df3 = pd.DataFrame(np.repeat(0, len(df_Bridger_wind3.columns))[None, :], 
                  columns=df_Bridger_wind3.columns,
                  index=[10])

df_Bridger_wind3 = pd.concat([df_Bridger_wind3, df3])

df_Bridger_wind3.loc[10,"R3 Mean error"] = np.mean(df_Bridger_TP["FlowError_percent"])
df_Bridger_wind3.loc[10,"R3 P2.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.025)
df_Bridger_wind3.loc[10,"R3 P50 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.5)
df_Bridger_wind3.loc[10,"R3 P97.5 error"] = np.quantile(df_Bridger_TP["FlowError_percent"], 0.975)

df_Bridger_wind3["R3 Count"] = df_Bridger_TP.groupby(["bin"])["FlowError_percent"].size()

# GHGSAT QUANT ERROR

df_GHGsat_TP =  matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]

df_GHGsat_TP["FlowError_percent"] = pd.to_numeric(df_GHGsat_TP.FlowError_percent, errors='coerce')

df_GHGsat_TP["bin"] = pd.cut(df_GHGsat_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1 Count",
        "R1 Mean error",
        "R1 P2.5 error",
        "R1 P50 error",
        "R1 P97.5 error"]

df_GHGsat_wind1 = pd.DataFrame(columns = column_names)

df_GHGsat_wind1["R1 Mean error"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].mean()
df_GHGsat_wind1["R1 P2.5 error"], df_GHGsat_wind1["R1 P50 error"], df_GHGsat_wind1["R1 P97.5 error"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_GHGsat_wind1.columns))[None, :], 
                  columns=df_GHGsat_wind1.columns,
                  index=[10])

df_GHGsat_wind1 = pd.concat([df_GHGsat_wind1, df1])

df_GHGsat_wind1.loc[10,"R1 Mean error"] = np.mean(df_GHGsat_TP["FlowError_percent"])
df_GHGsat_wind1.loc[10,"R1 P2.5 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.025)
df_GHGsat_wind1.loc[10,"R1 P50 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.5)
df_GHGsat_wind1.loc[10,"R1 P97.5 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.975)

df_GHGsat_wind1["R1 Count"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].size()

df_GHGsat_TP =  matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 2) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]

df_GHGsat_TP["FlowError_percent"] = pd.to_numeric(df_GHGsat_TP.FlowError_percent, errors='coerce')

df_GHGsat_TP["bin"] = pd.cut(df_GHGsat_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R2 Count",
        "R2 Mean error",
        "R2 P2.5 error",
        "R2 P50 error",
        "R2 P97.5 error"]

df_GHGsat_wind2 = pd.DataFrame(columns = column_names)

df_GHGsat_wind2["R2 Mean error"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].mean()
df_GHGsat_wind2["R2 P2.5 error"], df_GHGsat_wind2["R2 P50 error"], df_GHGsat_wind2["R2 P97.5 error"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df2 = pd.DataFrame(np.repeat(0, len(df_GHGsat_wind2.columns))[None, :], 
                  columns=df_GHGsat_wind2.columns,
                  index=[10])

df_GHGsat_wind2 = pd.concat([df_GHGsat_wind2, df2])

df_GHGsat_wind2.loc[10,"R2 Mean error"] = np.mean(df_GHGsat_TP["FlowError_percent"])
df_GHGsat_wind2.loc[10,"R2 P2.5 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.025)
df_GHGsat_wind2.loc[10,"R2 P50 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.5)
df_GHGsat_wind2.loc[10,"R2 P97.5 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.975)

df_GHGsat_wind2["R2 Count"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].size()

df_GHGsat_TP =  matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 3) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]

df_GHGsat_TP["FlowError_percent"] = pd.to_numeric(df_GHGsat_TP.FlowError_percent, errors='coerce')

df_GHGsat_TP["bin"] = pd.cut(df_GHGsat_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R3 Count",
        "R3 Mean error",
        "R3 P2.5 error",
        "R3 P50 error",
        "R3 P97.5 error"]

df_GHGsat_wind3 = pd.DataFrame(columns = column_names)

df_GHGsat_wind3["R3 Mean error"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].mean()
df_GHGsat_wind3["R3 P2.5 error"], df_GHGsat_wind3["R3 P50 error"], df_GHGsat_wind3["R3 P97.5 error"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.025), df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.5), df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].quantile(0.975)


df3 = pd.DataFrame(np.repeat(0, len(df_GHGsat_wind3.columns))[None, :], 
                  columns=df_GHGsat_wind3.columns,
                  index=[10])

df_GHGsat_wind3 = pd.concat([df_GHGsat_wind3, df3])

df_GHGsat_wind3.loc[10,"R3 Mean error"] = np.mean(df_GHGsat_TP["FlowError_percent"])
df_GHGsat_wind3.loc[10,"R3 P2.5 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.025)
df_GHGsat_wind3.loc[10,"R3 P50 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.5)
df_GHGsat_wind3.loc[10,"R3 P97.5 error"] = np.quantile(df_GHGsat_TP["FlowError_percent"], 0.975)

df_GHGsat_wind3["R3 Count"] = df_GHGsat_TP.groupby(["bin"])["FlowError_percent"].size()

# Wind Error


column_names = [
        "R1 Count"
        "R1 Mean error",
        "R1 P2.5 error",
        "R1 P50 error",
        "R1 P97.5 error"]

df_CM_wind1 = pd.DataFrame(columns = column_names)

df_CM_wind1["R1 Mean error"] = df_CM_TP.groupby(["bin"])["WindError_percent"].mean()
df_CM_wind1["R1 P2.5 error"], df_CM_wind1["R1 P50 error"], df_CM_wind1["R1 P97.5 error"] = df_CM_TP.groupby(["bin"])["WindError_percent"].quantile(0.025), df_CM_TP.groupby(["bin"])["WindError_percent"].quantile(0.5), df_CM_TP.groupby(["bin"])["WindError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_CM_wind1.columns))[None, :], 
                  columns=df_CM_wind1.columns,
                  index=[10])

df_CM_wind1 = pd.concat([df_CM_wind1, df1])

df_CM_wind1.loc[10,"R1 Mean error"] = np.mean(df_CM_TP["WindError_percent"])
df_CM_wind1.loc[10,"R1 P2.5 error"] = np.quantile(df_CM_TP["WindError_percent"], 0.025)
df_CM_wind1.loc[10,"R1 P50 error"] = np.quantile(df_CM_TP["WindError_percent"], 0.5)
df_CM_wind1.loc[10,"R1 P97.5 error"] = np.quantile(df_CM_TP["WindError_percent"], 0.975)

df_CM_wind1["R1 Count"] = df_CM_TP.groupby(["bin"])["WindError_percent"].size()



# BRIDGER QUANT ERROR

df_Bridger_TP =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) & (
            matchedDF_Bridger['tc_Classification'] == 'TP') & (
            matchedDF_Bridger['WindType'] == 'HRRR')]

df_Bridger_TP["WindError_percent"] = pd.to_numeric(df_Bridger_TP.WindError_percent, errors='coerce')

df_Bridger_TP["bin"] = pd.cut(df_Bridger_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1 Count",
        "R1 Mean error",
        "R1 P2.5 error",
        "R1 P50 error",
        "R1 P97.5 error"]

df_Bridger_wind1 = pd.DataFrame(columns = column_names)

df_Bridger_wind1["R1 Mean error"] = df_Bridger_TP.groupby(["bin"])["WindError_percent"].mean()
df_Bridger_wind1["R1 P2.5 error"], df_Bridger_wind1["R1 P50 error"], df_Bridger_wind1["R1 P97.5 error"] = df_Bridger_TP.groupby(["bin"])["WindError_percent"].quantile(0.025), df_Bridger_TP.groupby(["bin"])["WindError_percent"].quantile(0.5), df_Bridger_TP.groupby(["bin"])["WindError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_Bridger_wind1.columns))[None, :], 
                  columns=df_Bridger_wind1.columns,
                  index=[10])

df_Bridger_wind1 = pd.concat([df_Bridger_wind1, df1])

df_Bridger_wind1.loc[10,"R1 Mean error"] = np.mean(df_Bridger_TP["WindError_percent"])
df_Bridger_wind1.loc[10,"R1 P2.5 error"] = np.quantile(df_Bridger_TP["WindError_percent"], 0.025)
df_Bridger_wind1.loc[10,"R1 P50 error"] = np.quantile(df_Bridger_TP["WindError_percent"], 0.5)
df_Bridger_wind1.loc[10,"R1 P97.5 error"] = np.quantile(df_Bridger_TP["WindError_percent"], 0.975)

df_Bridger_wind1["R1 Count"] = df_Bridger_TP.groupby(["bin"])["WindError_percent"].size()

df_Bridger_TP =  matchedDF_Bridger[(matchedDF_Bridger['UnblindingStage'] == 1) & (
            matchedDF_Bridger['tc_Classification'] == 'TP') & (
            matchedDF_Bridger['WindType'] == 'NAM12')]

df_Bridger_TP["WindError_percent"] = pd.to_numeric(df_Bridger_TP.WindError_percent, errors='coerce')

df_Bridger_TP["bin"] = pd.cut(df_Bridger_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1_2 Count",
        "R1_2 Mean error",
        "R1_2 P2.5 error",
        "R1_2 P50 error",
        "R1_2 P97.5 error"]

df_Bridger_wind1_2 = pd.DataFrame(columns = column_names)

df_Bridger_wind1_2["R1_2 Mean error"] = df_Bridger_TP.groupby(["bin"])["WindError_percent"].mean()
df_Bridger_wind1_2["R1_2 P2.5 error"], df_Bridger_wind1_2["R1_2 P50 error"], df_Bridger_wind1_2["R1_2 P97.5 error"] = df_Bridger_TP.groupby(["bin"])["WindError_percent"].quantile(0.025), df_Bridger_TP.groupby(["bin"])["WindError_percent"].quantile(0.5), df_Bridger_TP.groupby(["bin"])["WindError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_Bridger_wind1_2.columns))[None, :], 
                  columns=df_Bridger_wind1_2.columns,
                  index=[10])

df_Bridger_wind1_2 = pd.concat([df_Bridger_wind1_2, df1])

df_Bridger_wind1_2.loc[10,"R1_2 Mean error"] = np.mean(df_Bridger_TP["WindError_percent"])
df_Bridger_wind1_2.loc[10,"R1_2 P2.5 error"] = np.quantile(df_Bridger_TP["WindError_percent"], 0.025)
df_Bridger_wind1_2.loc[10,"R1_2 P50 error"] = np.quantile(df_Bridger_TP["WindError_percent"], 0.5)
df_Bridger_wind1_2.loc[10,"R1_2 P97.5 error"] = np.quantile(df_Bridger_TP["WindError_percent"], 0.975)

df_Bridger_wind1_2["R1_2 Count"] = df_Bridger_TP.groupby(["bin"])["WindError_percent"].size()


# GHGSAT QUANT ERROR

df_GHGsat_TP =  matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]

df_GHGsat_TP["WindError_percent"] = pd.to_numeric(df_GHGsat_TP.WindError_percent, errors='coerce')

df_GHGsat_TP["bin"] = pd.cut(df_GHGsat_TP["Wind_MPS_mean300"], [0, 1, 2, 3, 4, 5, 20])

column_names = [
        "R1 Count",
        "R1 Mean error",
        "R1 P2.5 error",
        "R1 P50 error",
        "R1 P97.5 error"]

df_GHGsat_wind1 = pd.DataFrame(columns = column_names)

df_GHGsat_wind1["R1 Mean error"] = df_GHGsat_TP.groupby(["bin"])["WindError_percent"].mean()
df_GHGsat_wind1["R1 P2.5 error"], df_GHGsat_wind1["R1 P50 error"], df_GHGsat_wind1["R1 P97.5 error"] = df_GHGsat_TP.groupby(["bin"])["WindError_percent"].quantile(0.025), df_GHGsat_TP.groupby(["bin"])["WindError_percent"].quantile(0.5), df_GHGsat_TP.groupby(["bin"])["WindError_percent"].quantile(0.975)


df1 = pd.DataFrame(np.repeat(0, len(df_GHGsat_wind1.columns))[None, :], 
                  columns=df_GHGsat_wind1.columns,
                  index=[10])

df_GHGsat_wind1 = pd.concat([df_GHGsat_wind1, df1])

df_GHGsat_wind1.loc[10,"R1 Mean error"] = np.mean(df_GHGsat_TP["WindError_percent"])
df_GHGsat_wind1.loc[10,"R1 P2.5 error"] = np.quantile(df_GHGsat_TP["WindError_percent"], 0.025)
df_GHGsat_wind1.loc[10,"R1 P50 error"] = np.quantile(df_GHGsat_TP["WindError_percent"], 0.5)
df_GHGsat_wind1.loc[10,"R1 P97.5 error"] = np.quantile(df_GHGsat_TP["WindError_percent"], 0.975)

df_GHGsat_wind1["R1 Count"] = df_GHGsat_TP.groupby(["bin"])["WindError_percent"].size()




# comparing mean flow percent at different flight altitudes

df_GHGsat_alt1 =  matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['Altitude (feet)'] > 9000) & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]

df_GHGsat_alt2 =  matchedDF_GHGSat[(matchedDF_GHGSat['UnblindingStage'] == 1) & (
            matchedDF_GHGSat['tc_Classification'] == 'TP') & (
            matchedDF_GHGSat['Altitude (feet)'] < 9000) & (
            matchedDF_GHGSat['cr_kgh_CH4_mean30'] <= 2000)]


column_names = [
        "Count",
        "Mean error",
        "P2.5 error",
        "P50 error",
        "P97.5 error"]

df_alt1 = pd.DataFrame(columns = column_names)

df_alt1.loc[0,"Mean error"] = np.mean(df_GHGsat_alt1["FlowError_percent"])
df_alt1.loc[0,"P2.5 error"] = np.quantile(df_GHGsat_alt1["FlowError_percent"], 0.025)
df_alt1.loc[0,"P50 error"] = np.quantile(df_GHGsat_alt1["FlowError_percent"], 0.5)
df_alt1.loc[0,"P97.5 error"] = np.quantile(df_GHGsat_alt1["FlowError_percent"], 0.975)
df_alt1.loc[0,"Count"] = np.size(df_GHGsat_alt1["FlowError_percent"])

df_alt2 = pd.DataFrame(columns = column_names)

df_alt2.loc[0,"Mean error"] = np.mean(df_GHGsat_alt2["FlowError_percent"])
df_alt2.loc[0,"P2.5 error"] = np.quantile(df_GHGsat_alt2["FlowError_percent"], 0.025)
df_alt2.loc[0,"P50 error"] = np.quantile(df_GHGsat_alt2["FlowError_percent"], 0.5)
df_alt2.loc[0,"P97.5 error"] = np.quantile(df_GHGsat_alt2["FlowError_percent"], 0.975)
df_alt2.loc[0,"Count"] = np.size(df_GHGsat_alt2["FlowError_percent"])


t_stat, p_val = stats.ttest_ind(df_GHGsat_alt1["FlowError_percent"], df_GHGsat_alt2["FlowError_percent"])  
print("t-statistic = " + str(t_stat))  
print("p-value = " + str(p_val))

KS_D, KS_p = stats.ks_2samp(df_GHGsat_alt1["FlowError_percent"], df_GHGsat_alt2["FlowError_percent"])
print("k-statistic = " + str(KS_D))  
print("p-value = " + str(KS_p))

sns.distplot(df_GHGsat_alt1["FlowError_percent"], hist=True, rug=False)
sns.distplot(df_GHGsat_alt2["FlowError_percent"], hist=True, rug=False)

plt.show()

def statistic(x, y, axis):
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

rng = np.random.default_rng()

from scipy.stats import permutation_test
# because our statistic is vectorized, we pass `vectorized=True`
# `n_resamples=np.inf` indicates that an exact test is to be performed
res = permutation_test((df_GHGsat_alt1["FlowError_percent"], df_GHGsat_alt2["FlowError_percent"]), statistic, vectorized=True,
                       n_resamples=100000, alternative='two-sided')
print(res.pvalue)


x = 1