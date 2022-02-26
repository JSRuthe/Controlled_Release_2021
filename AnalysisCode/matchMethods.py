import datetime
import math
import anemometerMethods
from UnitConversion import convertUnits
from matchMethods import  calculatePassSpecificData
import numpy as np
import pandas as pd
from scipy import integrate
import pytz
import os.path


def performMatching_Stanford(bridgerDF, quadrathermDF, sonicDF, minPlumeLength, cr_averageperiod_sec, CH4_frac):
    # (1) Matches Bridger passes to controlled releases in Stanford Quadratherm time series
    # (2) calculates plume length for each pass and determines if plume is established,
    # (3) classifies each pass as TP, FN, or NE,
    # (4) 

    cwd = os.getcwd()    
    DataPath = os.path.join(cwd, 'EhrenbergTestData') 
    
    # Bridger reported additional rows for emissions from the Rawhide trailer. Select only rows where emission Location
    # ID = 33931 (the release point) and ignore rows where emission point is the leaky trailer
    bridgerDF = bridgerDF.loc[bridgerDF['Emission Location Id'] == 33931] 
    
    print("Matching bridger passes to release events...")
    matchedDF_Stanford = matchPassToQuadratherm(bridgerDF, quadrathermDF)  # match each pass to release event

    print("Checking plume lengths...")
    matchedDF_Stanford = checkPlumes_Stanford(DataPath, matchedDF_Stanford, sonicDF, mThreshold=minPlumeLength)  # determine plume lengths

    print("Classifying detections...")
    matchedDF_Stanford = classifyDetections_Stanford(matchedDF_Stanford)  # assign TP, FN, and NE classifications

    print("Setting flight feature wind stats...")
    matchedDF_Stanford = anemometerMethods_Stanford.appendFlightFeatureMetStats_Stanford(matchedDF_Stanford, sonicDF) #, dt=cr_averageperiod_sec)

    print("Setting nominal altitude...")
    matchedDF_Stanford = setNominalAltitude(matchedDF_Stanford)

    print("Applying unit conversions...")
    matchedDF_Stanford = convertUnits(matchedDF_Stanford, CH4_frac)

    del matchedDF_Stanford['cr_coriolis_gps_mean']  
    del matchedDF_Stanford['cr_coriolis_gps_std']
    del matchedDF_Stanford['Match Time'] 
    
    print("Setting errors in flow estimates...")
    matchedDF_Stanford = setFlowError(matchedDF_Stanford)

    return matchedDF_Stanford


def matchPassToQuadratherm(bridgerDF, quadrathermDF):

    bridgerDF['Match Time'] = bridgerDF['Detection Time (UTC)']
    bridgerDF.loc[bridgerDF["Match Time"].isnull(),'Match Time'] = bridgerDF["Flight Feature Time (UTC)"]    

    bridgerDF['Match Time'] = pd.to_datetime(bridgerDF['Match Time'])  
    
    matchedDF_Stanford = pd.DataFrame()  # makae empty df to store results
    matchedDF_Stanford = bridgerDF.merge(quadrathermDF, left_on = ['Match Time'], right_index = True)

    return matchedDF_Stanford

def classifyDetections_Stanford(matchedDF_Stanford):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF_Stanford.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF_Stanford.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF_Stanford.loc[idx, 'Detection'] = -1
        # False negatives occur if Bridger does not record a detection 
        # AND Stanford is releasing
        elif pd.isna(row['Detection Id']) and row['cr_scfh_mean'] > 0:
            matchedDF_Stanford.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF_Stanford.loc[idx, 'Detection'] = 0
        # False positives occur if Bridger does record a detection 
        # AND Stanford is not releasing
        #todo: check with Jeff if cr_SCFH_mean would actually be zero in FP or if he should be checking setpoint instead of metered value.
        #2/21/2022 note, there are no FP results in data set
        elif pd.notna(row['Detection Id']) and row['cr_scfh_mean'] == 0:
            matchedDF_Stanford.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF_Stanford.loc[idx, 'Detection'] = 0
        else:
            matchedDF_Stanford.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF_Stanford.loc[idx, 'Detection'] = 1
    return matchedDF_Stanford


def checkPlumes_Stanford(DataPath, matchedDF_Stanford, sonicDF, mThreshold):
    """Calculates a plume length and compares to threshold for established plume
    :param matchedDF = dataframe of aircraft passes matched with events
    :param metDF = dataframe from anemometer
    :param mThreshold = minimum plume length in meters to consider "established"
    :return matchedDF = updated dataframe with plume length and plume established added to each row"""

    # Calculate time since last pass    
    # Quad_new_setpoint = a list of timestamps where Stanford adjusted the gas flow to a new level
    ts_path = os.path.join(DataPath, 'transition_stamps_v2.csv')
    Quad_new_setpoint = pd.read_csv(ts_path, skiprows=0, usecols=[0],names=['datetime_UTC'], parse_dates=True)
    Quad_new_setpoint['datetime_UTC'] = pd.to_datetime(Quad_new_setpoint['datetime_UTC'])
    Quad_new_setpoint['datetime_UTC'] = Quad_new_setpoint.apply(
            lambda x: x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    
    matchedDF_Stanford['cr_start'] = np.nan
    matchedDF_Stanford['cr_end'] = np.nan
    matchedDF_Stanford['cr_idx'] = np.nan
    matchedDF_Stanford['PlumeLength_m'] = np.nan

    for i in range(matchedDF_Stanford.shape[0]):
        matchedDF_Stanford['cr_start'][i] = min(Quad_new_setpoint['datetime_UTC'], key = lambda datetime :
                                                  ((matchedDF_Stanford['Match Time'][i] - datetime).total_seconds() < 0,
                                                   (matchedDF_Stanford['Match Time'][i] - datetime).total_seconds()))
        idx = Quad_new_setpoint[Quad_new_setpoint['datetime_UTC'] == matchedDF_Stanford['cr_start'][i]].index[0]    
        matchedDF_Stanford['cr_end'][i] = Quad_new_setpoint['datetime_UTC'][idx+1]
        
    idx = 2001
    for i in range(Quad_new_setpoint.shape[0]):    
        matchedDF_Stanford['cr_idx'][matchedDF_Stanford['cr_start'] == Quad_new_setpoint['datetime_UTC'][i]] = idx
        idx = idx + 1
    
    # calculate plume lengths
    matchedDF_Stanford['PlumeLength_m'] = matchedDF_Stanford.apply(
        lambda x: calcPlumeLength(x['cr_start'], x['Match Time'], sonicDF), axis=1)
    # check if plume is established
    matchedDF_Stanford['PlumeEstablished'] = matchedDF_Stanford.apply(lambda x: establishedPlume(x['PlumeLength_m'], mThreshold), axis=1)
    
    return matchedDF_Stanford


def calcPlumeLength(t1, t2, sonicDF):
    """integrate wind speed from t1 to t2 to determine plume length in meters
    :param t1 = start time for integration
    :param t2 = end time for integration
    :param metDF = met station dataframe.
    :return plumeLength = length of plume in meters"""
    if pd.isna(t1) | pd.isna(t2):
        plumeLength = 0
    else:
        #mph_to_ms = 0.44704  # 1 mph = 0.44704 m/s
        #sonicDF = sonicDF.set_index('datetime')  # set datetime as index
        sonicDF = sonicDF[t1.astimezone(pytz.utc):t2.astimezone(
            pytz.utc)]  # subset to only data in the timeframe we're interested in
        # import pdb; pdb.set_trace()
        integrated = sonicDF.apply(integrate.trapz)  # integrate all fields
        plumeLength = integrated['Speed_MPS'] #* mph_to_ms  # convert integrated windspeed (mph*s) to m
    return plumeLength

def establishedPlume(plumeLength_m, mThreshold):
    """Plume is established if integral of wind speed from eStart to pStart is > mThreshold
    :param plumeLength_m = length of plume since last change
    :param mThreshold = minimum plume length in meters to consider plume "established"
    :return True if plumeLength_m > mThreshold, else False"""
    if plumeLength_m >= mThreshold:
        return True
    else:
        return False

def setNominalAltitude(df):
    """set nominal altitude to 500' agl or 675' agl. """
    df['Nominal Altitude (ft)'] = df.apply(lambda x: 500 if x['Flight Feature Agl (m)'] < 180 else 675, axis=1)
    return df



def setFlowError(df):
    """set flow error for all TP detections """
    # Bridger estimates using HRRR wind
    df['FlowError_kgh'] = df.apply(
        lambda x: pd.NA if (pd.isna(x['b_kgh'])) & (pd.isna(x['cr_kgh_CH4_mean']))
        else x['b_kgh'] - x['cr_kgh_CH4_mean'], axis=1)
    df['FlowError_percent'] = df.apply(
        lambda x: pd.NA if pd.isna(x['FlowError_kgh']) else x['FlowError_kgh'] / x['cr_kgh_CH4_mean'] * 100, axis=1)

    return df