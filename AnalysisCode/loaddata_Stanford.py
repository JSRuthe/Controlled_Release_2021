# imports
import datetime
import pytz
import math
import os.path
import pathlib
import pandas as pd
import bisect
import numpy as np

def loaddata_Stanford(cr_averageperiod_sec):
    """Load all data from Midland testing"""
    
    cwd = os.getcwd()    
    DataPath = os.path.join(cwd, 'EhrenbergTestData')    
    
    
    # load Bridger AZ data (Stanford) processed with HRRR wind
    print("Loading Bridger HRRR data (Stanford)...")
    bridgerHRRR_path = os.path.join(DataPath, 'XOM0011 Stanford CR - HRRR.xlsx')
    bridgerHRRRDF = loadBridgerData(bridgerHRRR_path)
    bridgerHRRRDF['WindType'] = 'HRRR'
    bridgerHRRRDF['ExperimentSet'] = 'AZ'
    bridgerHRRRDF = bridgerHRRRDF[bridgerHRRRDF['Emission Location Id'] == 33931]

    # load Bridger AZ data (Stanford) processed with NAM12 wind
    print("Loading Bridger NAM12 data (Stanford)...")
    bridgerNAM12_path = os.path.join(DataPath, 'XOM0011 Stanford CR - NAM12.xlsx')
    bridgerNAM12DF = loadBridgerData(bridgerNAM12_path)
    bridgerNAM12DF['WindType'] = 'NAM12'
    bridgerNAM12DF['ExperimentSet'] = 'AZ'
    bridgerNAM12DF = bridgerNAM12DF[bridgerNAM12DF['Emission Location Id'] == 33931]

    # load Bridger AZ data (Stanford) processed with sonic wind
    print("Loading Bridger Sonic data (Stanford)...")
    bridgerSonic_path = os.path.join(DataPath, 'XOM0011 Stanford CR - Anemometer.xlsx')
    bridgerSonicDF = loadBridgerData(bridgerSonic_path)
    bridgerSonicDF['WindType'] = 'Sonic'
    bridgerSonicDF['ExperimentSet'] = 'AZ'
    bridgerSonicDF = bridgerSonicDF[bridgerSonicDF['Emission Location Id'] == 33931]

    # append Bridger data into single DF
    bridgerDF = pd.concat([bridgerHRRRDF, bridgerNAM12DF, bridgerSonicDF], ignore_index=True)

    
    # Delete rows where Bridger passed over before Stanford was prepared to release
    bridgerDF = bridgerDF.drop(bridgerDF[(bridgerDF['Flight Feature Time (UTC)'] < '2021.11.03 17:38:06')].index)
    #bridgerDF = bridgerDF.drop(bridgerDF.index[[0,1,116,117]])
    bridgerDF = bridgerDF.reset_index()    
    
    # load quadratherm data
    print("Loading Quadratherm data...")
    quadrathermDF = loadQuadrathermData_Stanford(DataPath, cr_averageperiod_sec)
    
    # load anemometer data
    print("Loading anemometer data...")
    sonic_path = os.path.join(DataPath, 'Sonic\\')
    sonicDF = combineAnemometer_Stanford(sonic_path)
    
    return bridgerDF, quadrathermDF, sonicDF
    
def loadBridgerData(filepath):
    """Load bridger data from report and format datetimes."""
    df = pd.read_excel(filepath, sheet_name='emitter_group_scan', skiprows=4, engine='openpyxl')
    # convert datetime data to a datetime object; format: 04-Oct-2021 18:34:31
    df['Flight Feature Time (UTC)'] = df.apply(
        lambda x: datetime.datetime.strptime(x['Flight Feature Time (UTC)'], '%d-%b-%Y %H:%M:%S'), axis=1)
    df['Flight Feature Time (UTC)'] = df.apply(
        lambda x: x['Flight Feature Time (UTC)'].replace(tzinfo=pytz.timezone("UTC")), axis=1)

    df['Detection Time (UTC)'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Detection Time (UTC)']) else
        datetime.datetime.strptime(x['Detection Time (UTC)'], '%d-%b-%Y %H:%M:%S'), axis=1)
    df['Detection Time (UTC)'] = df.apply(
        lambda x: pd.NA if pd.isna(x['Detection Time (UTC)']) else
        x['Detection Time (UTC)'].replace(tzinfo=pytz.timezone("UTC")), axis=1)

    return df


def loadQuadrathermData_Stanford(DataPath, cr_averageperiod_sec):
    
    # Load time series data from Nanodac:
    # (1) Select data from correct channel depending on which meter was used
    # (2) Delete additional channels
    nano_1_path = os.path.join(DataPath, 'nano_21113_1_exp.csv')
    Quad_data_1 = pd.read_csv(nano_1_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_1['datetime_UTC'] = pd.to_datetime(Quad_data_1['datetime_UTC'])
    Quad_data_1.set_index('datetime_UTC', inplace = True)
    Quad_data_1['instantaneous_scfh'] = np.nan
    Quad_data_1['instantaneous_scfh'][(Quad_data_1.index < '2021.11.03 17:33:18')] = Quad_data_1['channel_1'][(Quad_data_1.index < '2021.11.03 17:33:18')]
    Quad_data_1['instantaneous_scfh'][(Quad_data_1.index > '2021.11.03 17:38:37')] = Quad_data_1['channel_2'][(Quad_data_1.index > '2021.11.03 17:38:37')]
    del Quad_data_1['channel_1'] 
    del Quad_data_1['channel_2']
    del Quad_data_1['channel_3']
    del Quad_data_1['channel_4']
    nano_2_path = os.path.join(DataPath, 'nano_21113_2_exp.csv')    
    Quad_data_2 = pd.read_csv(nano_2_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_2['datetime_UTC'] = pd.to_datetime(Quad_data_2['datetime_UTC'])
    Quad_data_2.set_index('datetime_UTC', inplace = True)
    Quad_data_2['instantaneous_scfh'] = np.nan
    Quad_data_2['instantaneous_scfh'][(Quad_data_2.index < '2021.11.03 21:32:11')] = Quad_data_2['channel_2'][(Quad_data_2.index < '2021.11.03 21:32:11')]
    del Quad_data_2['channel_1'] 
    del Quad_data_2['channel_2']
    del Quad_data_2['channel_3']
    del Quad_data_2['channel_4']
    nano_3_path = os.path.join(DataPath, 'nano_21114_exp.csv')    
    Quad_data_3 = pd.read_csv(nano_3_path, skiprows=1, usecols=[0,1,2,3,4],names=['datetime_UTC','channel_1','channel_2','channel_3','channel_4'], parse_dates=True)
    Quad_data_3['datetime_UTC'] = pd.to_datetime(Quad_data_3['datetime_UTC'])
    Quad_data_3.set_index('datetime_UTC', inplace = True)
    Quad_data_3['instantaneous_scfh'] = np.nan
    Quad_data_3['instantaneous_Coriolis_gps'] = np.nan
    Quad_data_3['instantaneous_scfh'][(Quad_data_3.index < '2021.11.04 19:27:38')] = Quad_data_3['channel_2'][(Quad_data_3.index < '2021.11.04 19:27:38')]
    Quad_data_3['instantaneous_Coriolis_gps'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')] = Quad_data_3['channel_4'][(Quad_data_3.index > '2021.11.04 19:28:02') & (Quad_data_3.index < '2021.11.04 20:24:37')]
    Quad_data_3['instantaneous_scfh'][(Quad_data_3.index > '2021.11.04 20:24:45')] = Quad_data_3['channel_2'][(Quad_data_3.index > '2021.11.04 20:24:45')]
    del Quad_data_3['channel_1'] 
    del Quad_data_3['channel_2']
    del Quad_data_3['channel_3']
    del Quad_data_3['channel_4']
    #Nanodac appears to have froze for the period 21.11.4 19:28:02 : 19:57:12. Replace data with handwritten notes
    hand_data_21114_path = os.path.join(DataPath, '21114_releasedat_Coriolis.csv')
    hand_data_21114 = pd.read_csv(hand_data_21114_path, skiprows=0, usecols=[0,1],names=['datetime_UTC','instantaneous_Coriolis_gps'], parse_dates=True)
    hand_data_21114['datetime_UTC'] = pd.to_datetime(hand_data_21114['datetime_UTC'])
    hand_data_21114.set_index('datetime_UTC', inplace = True)

    # Concatenate all time series data
    Quad_data_all = pd.concat([Quad_data_1, Quad_data_2, Quad_data_3])
    
    # Overwrite Nanodac Coriolis data with hand recorded Coriolis data
    Quad_data_all = Quad_data_all.drop(Quad_data_all[(Quad_data_all.index > '2021.11.04 19:28:02') & (Quad_data_all.index < '2021.11.04 19:57:12')].index)
    
    Quad_data_all = pd.concat([Quad_data_all, hand_data_21114])
    
    Quad_date_range_1  = pd.date_range("2021.11.03 16:25:04", periods = 18431, freq = "s")
    Quad_date_range_1 = Quad_date_range_1.to_frame(index = True)
    Quad_date_range_2  = pd.date_range("2021.11.04 16:39:41", periods = 17705, freq = "s")
    Quad_date_range_2 = Quad_date_range_2.to_frame(index = True)
    Quad_date_range = pd.concat([Quad_date_range_1, Quad_date_range_2])
    

    # Perform outer join between date range and Quadratherm data
    quadrathermDF = Quad_date_range.join(Quad_data_all, how='outer')
    time_series = quadrathermDF[0]
    del quadrathermDF[0]
    
    # Back-fill missing data
    quadrathermDF = quadrathermDF.bfill()
    
    # nan data where the Quadratherm isn't being used and where the nanodac isn't being used
    quadrathermDF['instantaneous_Coriolis_gps'][(quadrathermDF.index < '2021.11.04 19:28:02')] = np.NaN
    quadrathermDF['instantaneous_Coriolis_gps'][(quadrathermDF.index > '2021.11.04 20:24:37')] = np.NaN
    quadrathermDF['instantaneous_scfh'][(quadrathermDF.index > '2021.11.04 19:28:02') & (quadrathermDF.index < '2021.11.04 20:24:37')] = np.NaN
    
    # Localize the datetime index
    time_series = time_series.dt.tz_localize(pytz.utc)
    quadrathermDF.index = time_series

    # Convert from SCFH to KGH
    quadrathermDF['instantaneous_scfh'] = pd.to_numeric(quadrathermDF['instantaneous_scfh'],errors = 'coerce')
    quadrathermDF['instantaneous_Coriolis_gps'] = pd.to_numeric(quadrathermDF['instantaneous_Coriolis_gps'],errors = 'coerce')
    # For Coriolis data we are converting from grams per second
    # Edit - perform unit conversion later
    # quadrathermDF['instantaneous_scfh'][(quadrathermDF.index > '2021.11.04 19:28:02') & (quadrathermDF.index < '2021.11.04 20:24:37')] = quadrathermDF['instantaneous_scfh'][(quadrathermDF.index > '2021.11.04 19:28:02') & (quadrathermDF.index < '2021.11.04 20:24:37')]*(1/(16.043*1.202*CH4_frac))*3600

    # Add a column for moving average    
    quadrathermDF['cr_scfh_mean'] = quadrathermDF['instantaneous_scfh'].rolling(window=cr_averageperiod_sec).mean()
    quadrathermDF['cr_scfh_std'] = quadrathermDF['instantaneous_scfh'].rolling(window=cr_averageperiod_sec).std()
    quadrathermDF['cr_coriolis_gps_mean'] = quadrathermDF['instantaneous_Coriolis_gps'].rolling(window=cr_averageperiod_sec).mean()
    quadrathermDF['cr_coriolis_gps_std'] = quadrathermDF['instantaneous_Coriolis_gps'].rolling(window=cr_averageperiod_sec).std()
    quadrathermDF['cr_avg_start'] = quadrathermDF.index - datetime.timedelta(seconds = cr_averageperiod_sec)
    quadrathermDF['cr_avg_end'] = quadrathermDF.index
    
    so_path = os.path.join(DataPath, 'shut_off_stamps.csv')
    shutoff_points = pd.read_csv(so_path, skiprows=0, usecols=[0,1],names=['start_UTC', 'end_UTC'], parse_dates=True)
    shutoff_points['start_UTC'] = pd.to_datetime(shutoff_points['start_UTC'])
    shutoff_points['end_UTC'] = pd.to_datetime(shutoff_points['end_UTC'])
    shutoff_points['start_UTC'] = shutoff_points['start_UTC'].dt.tz_localize(pytz.utc)
    shutoff_points['end_UTC'] = shutoff_points['end_UTC'].dt.tz_localize(pytz.utc)
    
    for i in range(shutoff_points.shape[0]):
        quadrathermDF['cr_scfh_mean'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_coriolis_gps_mean'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_scfh_std'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
        quadrathermDF['cr_coriolis_gps_std'][(quadrathermDF.index > shutoff_points['start_UTC'][i]) & (quadrathermDF.index < shutoff_points['end_UTC'][i])] = 0
                
    del quadrathermDF['instantaneous_scfh']  
    del quadrathermDF['instantaneous_Coriolis_gps']
    
    # Delete all rows with NaT
    quadrathermDF["TMP"] = quadrathermDF.index.values                   # index is a DateTimeIndex
    quadrathermDF = quadrathermDF[quadrathermDF.TMP.notnull()]          # remove all NaT values
    quadrathermDF.drop(["TMP"], axis=1, inplace=True)                   # delete TMP again
    
    return quadrathermDF


def combineAnemometer_Stanford(sonic_path):
    
    # Process data for November 3
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.11.3'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 19
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_1  = pd.date_range("2021.11.03 16:59:08", periods = 17115, freq = "s")
    sonic_date_range_1 = sonic_date_range_1.to_frame(index = True)
    sonic_date_range_1  = sonic_date_range_1.tz_localize(pytz.utc)

    sonicDF_temp1 = processAnemometer_Stanford(path_lookup, localtz, cols, offset) 
    sonicDF_temp1 = sonicDF_temp1.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp1 = sonic_date_range_1.merge(sonicDF_temp1, how='outer', left_index=True, right_index=True)
    sonicDF_temp1 = sonicDF_temp1.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp1 = sonicDF_temp1.bfill()
    
    # Process data for November 4
    # Location = Ehrenberg
    # Timezone = MST
    
    # Sonic data is in Palo Alto time
    localtz = pytz.timezone("US/Pacific")
    date_string = '21.11.4'
    path_lookup = sonic_path + date_string + '\\'
    #path_export = path_compiled + date_string
    cols = [1,2,6]
    AZ_day = 20
    offset = 0.8438*AZ_day + 54.865
    offset = int(round(offset))

    sonic_date_range_2  = pd.date_range("2021.11.04 15:57:57", periods = 20367, freq = "s")
    sonic_date_range_2 = sonic_date_range_2.to_frame(index = True)
    sonic_date_range_2  = sonic_date_range_2.tz_localize(pytz.utc)
    
    sonicDF_temp2 = processAnemometer_Stanford(path_lookup, localtz, cols, offset)     
    sonicDF_temp2 = sonicDF_temp2.set_index('time')
    
    # Perform outer join between date range and Quadratherm data
    #sonicDF_temp1 = sonic_date_range_1.join(sonicDF_temp1, how='outer')
    sonicDF_temp2 = sonic_date_range_2.merge(sonicDF_temp2, how='outer', left_index=True, right_index=True)
    sonicDF_temp2 = sonicDF_temp2.iloc[: , 2:]
    
     # Back-fill missing data
    sonicDF_temp2 = sonicDF_temp2.bfill()
    
    sonicDF = pd.concat([sonicDF_temp1, sonicDF_temp2])



    return sonicDF


def processAnemometer_Stanford(path_lookup, localtz, cols, offset):

    #os.chdir(path_lookup)    
    #os.listdir()[2:]

    data = pd.DataFrame()
    for     file in os.listdir(path_lookup)[0:]:
          file_data = pd.read_csv(os.path.join(path_lookup, file),skiprows=4,
                        usecols=cols,names=['Direction','Speed_MPS','time'], index_col='time', parse_dates=True)
          file_data = file_data.dropna()
          data = data.append(file_data)
    
    data = data.reset_index()

    df = data.copy()
    df['Speed_MPS'] = df['Speed_MPS'].astype(float)

    # Set timezone
    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].dt.tz_localize(localtz)
    df = df.sort_values("time").reset_index(drop=True)    
    df['time'] = df['time'].apply(lambda x: x.astimezone(pytz.utc))

    # Apply time offset    
    df['time'] = df['time'] - datetime.timedelta(seconds = offset)

    # Calculate moving average of wind speed
    df['Speed_Moving_MPS'] = df['Speed_MPS'].rolling(window =300).mean()
    
    
    return df
    