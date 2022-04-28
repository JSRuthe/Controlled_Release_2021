import datetime
import math
import anemometerMethods
from UnitConversion import convertUnits
#from matchMethods import  calculatePassSpecificData
import numpy as np
import pandas as pd
from scipy import integrate
import pytz
import os.path
from meterUncertainty import meterUncertainty

def performMatching(operatorDF, meterDF_All, sonicDF_All):
    # (1) Matches Bridger passes to controlled releases in Stanford Quadratherm time series
    # (2) calculates plume length for each pass and determines if plume is established,
    # (3) classifies each pass as TP, FN, or NE,
    # (4) 


    cwd = os.getcwd()  
    
    print("Matching bridger passes to release events...")
    
    matchedDF = matchPassToQuadratherm(operatorDF, meterDF_All, sonicDF_All)  # match each pass to release event 
    
    DataPath = os.path.join(cwd, 'BridgerTestData') 
    print("Checking plume lengths Bridger...")
    matchedDF_Bridger = matchedDF[matchedDF['OperatorSet'] == 'Bridger']
    matchedDF_Bridger = matchedDF_Bridger.reset_index()  
    sonicDF_Bridger = sonicDF_All[sonicDF_All['OperatorSet'] == 'Bridger']
    matchedDF_Bridger = checkPlumes(DataPath, matchedDF_Bridger, sonicDF_Bridger, 
                                                                                tstamp_file = 'transition_stamps_v2.csv', 
                                                                                minPlumeLength = 150,
                                                                                Operator = 'Bridger')
    print("Classifying detections Bridger...")
    matchedDF_Bridger = classifyDetections_Bridger(matchedDF_Bridger)  # assign TP, FN, and NE classifications

    matchedDF_Bridger = assessUncertainty(matchedDF_Bridger)

    DataPath = os.path.join(cwd, 'GHGSatTestData') 
    print("Checking plume lengths GHGSat...")
    matchedDF_GHGSat = matchedDF[matchedDF['OperatorSet'] == 'GHGSat']
    matchedDF_GHGSat = matchedDF_GHGSat.reset_index()  
    sonicDF_GHGSat = sonicDF_All[sonicDF_All['OperatorSet'] == 'GHGSat']    
    matchedDF_GHGSat = checkPlumes(DataPath, matchedDF_GHGSat, sonicDF_GHGSat, 
                                                                                tstamp_file = 'transition_stamps_v2.csv',                                                 
                                                                                minPlumeLength = 150,
                                                                                Operator = 'GHGSat')
    print("Classifying detections GHGSat...")
    matchedDF_GHGSat = classifyDetections_GHGSat(matchedDF_GHGSat)  # assign TP, FN, and NE classifications
            
    matchedDF_GHGSat = assessUncertainty(matchedDF_GHGSat)
                                                                
    DataPath = os.path.join(cwd, 'CarbonMapperTestData') 
    print("Checking plume lengths CarbonMapper...")
    matchedDF_CarbonMapper = matchedDF[matchedDF['OperatorSet'] == 'CarbonMapper']
    matchedDF_CarbonMapper = matchedDF_CarbonMapper.reset_index()  
    sonicDF_CarbonMapper = sonicDF_All[sonicDF_All['OperatorSet'] == 'CarbonMapper']    
    matchedDF_CarbonMapper = checkPlumes(DataPath, matchedDF_CarbonMapper, sonicDF_CarbonMapper, 
                                                                                tstamp_file = 'transition_stamps.csv',                                          
                                                                                minPlumeLength = 150,
                                                                                Operator = 'CarbonMapper')

    print("Classifying detections CarbonMapper...")
    matchedDF_CarbonMapper = classifyDetections_CarbonMapper(matchedDF_CarbonMapper)  # assign TP, FN, and NE classifications

    matchedDF_CarbonMapper = assessUncertainty(matchedDF_CarbonMapper)



    DataPath = os.path.join(cwd, 'MAIRTestData') 
    print("Checking plume lengths MAIR...")
    matchedDF_MAIR = matchedDF[matchedDF['OperatorSet'] == 'MAIR']
    matchedDF_MAIR = matchedDF_MAIR.reset_index()  
    #Use same Sonic data for MAIR as we generated for Carbon Mapper
    sonicDF_MAIR = sonicDF_All[sonicDF_All['OperatorSet'] == 'CarbonMapper']    
    matchedDF_MAIR = checkPlumes(DataPath, matchedDF_MAIR, sonicDF_MAIR, 
                                                                                tstamp_file = 'transition_stamps.csv',                                          
                                                                                minPlumeLength = 150,
                                                                                Operator = 'MAIR')

    print("Classifying detections MAIR...")
    matchedDF_MAIR = classifyDetections_MAIR(matchedDF_MAIR)  # assign TP, FN, and NE classifications

    matchedDF_MAIR = assessUncertainty(matchedDF_MAIR)

    DataPath = os.path.join(cwd, 'SOOFIETestData')
    print("Checking plume lengths SOOFIE...")
    matchedDF_SOOFIE = matchedDF[matchedDF['OperatorSet'] == 'SOOFIE']
    matchedDF_SOOFIE = matchedDF_SOOFIE.reset_index()
    #Use the full sonic dataset

    sonicDF_SOOFIE = sonicDF_All

    #matchedDF_SOOFIE_new = pd.DataFrame(columns=matchedDF_SOOFIE.columns)

    #start_date = pd.to_datetime('2021.10.16 00:00:00')
    #for single_date in (start_date + datetime.timedelta(days = n) for n in range(20)):
    #    print(single_date.date())

    # Extract SOOFIE data where we have wind data
    #sonic_days = pd.Series(sonicDF_All.index).apply(lambda x: x.date).unique()
    #sonic_days_all = pd.Series(sonicDF_All.index).apply(lambda x: x.date)
    #SOOFIE_days = pd.Series(matchedDF_SOOFIE['Stanford_timestamp']).apply(lambda x: x.date)
    #is_sonic = SOOFIE_days.isin(sonic_days)
    #matchedDF_SOOFIE = matchedDF_SOOFIE[is_sonic]
    #for single_date in sonic_days:
    #    isin_single_date = (sonic_days_all == single_date)
    #    earliest = min(sonicDF_SOOFIE.loc[isin_single_date.values,:].index)
    #    latest = max(sonicDF_SOOFIE.loc[isin_single_date.values,:].index)
    #    new_rows = matchedDF_SOOFIE[(matchedDF_SOOFIE['Stanford_timestamp'] > earliest) &
    #                                (matchedDF_SOOFIE['Stanford_timestamp'] < latest)]
    #    matchedDF_SOOFIE_new = pd.concat([matchedDF_SOOFIE_new, new_rows])
    #
    #matchedDF_SOOFIE = matchedDF_SOOFIE_new
    #
    matchedDF_SOOFIE = checkPlumes(DataPath, matchedDF_SOOFIE, sonicDF_SOOFIE,
                                                                                tstamp_file = 'transition_stamps.csv',
                                                                                minPlumeLength = 150,
                                                                                Operator = 'SOOFIE')

    print("Classifying detections SOOFIE...")
    matchedDF_SOOFIE = classifyDetections_SOOFIE(matchedDF_SOOFIE)  # assign TP, FN, and NE classifications

    matchedDF_SOOFIE = assessUncertainty(matchedDF_SOOFIE)


    DataPath = os.path.join(cwd, 'SatelliteTestData') 
    print("Checking plume lengths Satellite data...")
    matchedDF_Satellites = matchedDF[(matchedDF['OperatorSet'] != 'MAIR') &
                               (matchedDF['OperatorSet'] != 'CarbonMapper') &
                               (matchedDF['OperatorSet'] != 'Bridger') &
                               (matchedDF['OperatorSet'] != 'GHGSat')]
    matchedDF_Satellites = matchedDF_Satellites.reset_index()  
    sonicDF_Satellites = sonicDF_All    
    matchedDF_Satellites = checkPlumes(DataPath, matchedDF_Satellites, sonicDF_Satellites, 
                                                                                tstamp_file = 'transition_stamps.csv',                                          
                                                                                minPlumeLength = 150,
                                                                                Operator = 'Satellite')

    print("Classifying detections Satellites...")
    matchedDF_Satellites = classifyDetections_Satellites(matchedDF_Satellites)  # assign TP, FN, and NE classifications

    matchedDF_Satellites = assessUncertainty(matchedDF_Satellites)
    


    # Set flow error


    #print("Setting flight feature wind stats...")
    #matchedDF = anemometerMethods.appendFlightFeatureMetStats(matchedDF, sonicDF) #, dt=cr_averageperiod_sec)

    #print("Setting nominal altitude...")
    #matchedDF = setNominalAltitude(matchedDF)

    #print("Setting errors in flow estimates...")
    #matchedDF = setFlowError(matchedDF)

    return matchedDF_Bridger, matchedDF_GHGSat, matchedDF_CarbonMapper, matchedDF_MAIR, matchedDF_Satellites, matchedDF_SOOFIE


def matchPassToQuadratherm(operatorDF, meterDF_All, sonicDF_All):

    # TEMPORARY: If this 
    #operatorDF['Match Time'] = operatorDF['Detection Time (UTC)']
    #operatorDF.loc[operatorDF["Match Time"].isnull(),'Match Time'] = operatorDF["Flight Feature Time (UTC)"]    
    # operatorDF['Match Time'] = pd.to_datetime(operatorDF['Match Time'])  
    
    matchedDF = pd.DataFrame()  # makae empty df to store results
    #operatorDF['Timestamp'] = pd.to_datetime(operatorDF['Timestamp'])
    matchedDF = operatorDF.merge(meterDF_All, left_on = ['Stanford_timestamp'], right_index = True)

    # Add wind speed MPS moving average column
    matchedDF = matchedDF.merge(sonicDF_All[['Wind_MPS_mean300']], left_on = ['Stanford_timestamp'], right_index = True)



    return matchedDF

def checkPlumes(DataPath, matchedDF, sonicDF,       
                                                                    tstamp_file,                                                     
                                                                    minPlumeLength,
                                                                    Operator):
    
    
    """Calculates a plume length and compares to threshold for established plume
    :param matchedDF = dataframe of aircraft passes matched with events
    :param metDF = dataframe from anemometer
    :param mThreshold = minimum plume length in meters to consider "established"
    :return matchedDF = updated dataframe with plume length and plume established added to each row"""

    # Calculate time since last pass    
    # Quad_new_setpoint = a list of timestamps where Stanford adjusted the gas flow to a new level
    
    # GHGSat:           Oct  21: # No time series available for morning of Oct  21, Assume that plume was stabilized two minutes prior to setpoints
    # Carbon Mapper:    July 31: # No time series available for morning of July 31, Assume that plume was stabilized two minutes prior to setpoints
    
    
    ts_path = os.path.join(DataPath, tstamp_file)
    Quad_new_setpoint = pd.read_csv(ts_path, skiprows=0, usecols=[0],names=['datetime_UTC'], parse_dates=True)
    Quad_new_setpoint['datetime_UTC'] = pd.to_datetime(Quad_new_setpoint['datetime_UTC'])
    Quad_new_setpoint['datetime_UTC'] = Quad_new_setpoint.apply(
            lambda x: x['datetime_UTC'].replace(tzinfo=pytz.timezone("UTC")), axis=1)
    
    matchedDF['cr_start'] = np.nan
    matchedDF['cr_end'] = np.nan
    matchedDF['cr_idx'] = np.nan
    matchedDF['PlumeLength_m'] = np.nan
    matchedDF['PlumeEstablished'] = np.nan
    matchedDF['PlumeDevelopTime'] = np.nan

    for i in range(matchedDF.shape[0]):
        matchedDF['cr_start'][i] = min(Quad_new_setpoint['datetime_UTC'], key = lambda datetime :
                                                  ((matchedDF['Stanford_timestamp'][i] - datetime).total_seconds() < 0,
                                                   (matchedDF['Stanford_timestamp'][i] - datetime).total_seconds()))
        idx = Quad_new_setpoint[Quad_new_setpoint['datetime_UTC'] == matchedDF['cr_start'][i]].index[0] 
        
        if Operator != 'Satellite':
            matchedDF['cr_end'][i] = Quad_new_setpoint['datetime_UTC'][idx+1]
        else:
            matchedDF['cr_end'][i] = 0
            
    matchedDF['PlumeDevelopTime'] = (matchedDF['Stanford_timestamp'] - pd.to_datetime(matchedDF['cr_start'])).dt.total_seconds()/60
        
    idx = 2001
    for i in range(Quad_new_setpoint.shape[0]):    
        matchedDF['cr_idx'][matchedDF['cr_start'] == Quad_new_setpoint['datetime_UTC'][i]] = idx
        idx = idx + 1
    
    # calculate plume lengths
    matchedDF['PlumeLength_m'] = matchedDF.apply(
        lambda x: calcPlumeLength(x['cr_start'], x['Stanford_timestamp'], sonicDF), axis=1)
    # check if plume is established
    matchedDF['PlumeEstablished'] = matchedDF.apply(lambda x: establishedPlume(x['PlumeLength_m'], minPlumeLength), axis=1)
    
    matchedDF['PlumeSteady'] = matchedDF.apply(lambda x: steadyPlume(x['cr_scfh_mean60'], x['cr_allmeters_scfh']), axis=1)
    
    return matchedDF

def DivZeroCheck(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 0

def classifyDetections_Bridger(matchedDF):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF.loc[idx, 'Detection'] = -1
        # False negatives occur if Bridger does not record a detection 
        # AND Stanford is releasing
        # Bridger reports "NA" for non-detects
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] > 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF.loc[idx, 'Detection'] = 0
        # False positives occur if Bridger does record a detection 
        # AND Stanford is not releasing
        elif pd.notna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF.loc[idx, 'Detection'] = 0
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'TN'  # TN = True Negative
            matchedDF.loc[idx, 'Detection'] = 0     
        elif not row['PlumeSteady']:
            matchedDF.loc[idx, 'tc_Classification'] = 'NS'  # NS = Not Steady
            matchedDF.loc[idx, 'Detection'] = 0  
        else:
            matchedDF.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF.loc[idx, 'Detection'] = 1
            
    return matchedDF

def classifyDetections_CarbonMapper(matchedDF):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF.loc[idx, 'Detection'] = -1
        # False negatives occur if Carbon Mapper does not record a detection
        # AND Stanford is releasing
        # For Carbon Mapper we use the QC filter to differentiate between detects and non-detects
        elif row['QC filter'] == 2 and row['cr_allmeters_scfh'] > 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF.loc[idx, 'Detection'] = 0
        # False positives occur if Carbon Mapper does record a detection
        # AND Stanford is not releasing
        elif row['QC filter'] == 1 and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF.loc[idx, 'Detection'] = 0
        elif row['QC filter'] == 0 :
            matchedDF.loc[idx, 'tc_Classification'] = 'ER'  # ER = Error
            matchedDF.loc[idx, 'Detection'] = -1
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'TN'  # TN = True Negative
            matchedDF.loc[idx, 'Detection'] = 0 
        elif not row['PlumeSteady']:
            matchedDF.loc[idx, 'tc_Classification'] = 'NS'  # NS = Not Steady
            matchedDF.loc[idx, 'Detection'] = 0  
        else:
            matchedDF.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF.loc[idx, 'Detection'] = 1
            
    return matchedDF

def classifyDetections_MAIR(matchedDF):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF.loc[idx, 'Detection'] = -1
        # False negatives occur if MAIR does not record a detection
        # AND Stanford is releasing
        # MAIR reports "NA" for non-detects
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] > 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF.loc[idx, 'Detection'] = 0
        # False positives occur if MAIR does record a detection
        # AND Stanford is not releasing
        elif pd.notna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF.loc[idx, 'Detection'] = 0
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'TN'  # TN = True Negative
            matchedDF.loc[idx, 'Detection'] = 0     
        elif not row['PlumeSteady']:
            matchedDF.loc[idx, 'tc_Classification'] = 'NS'  # NS = Not Steady
            matchedDF.loc[idx, 'Detection'] = 0  
        else:
            matchedDF.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF.loc[idx, 'Detection'] = 1
            
    return matchedDF

def classifyDetections_GHGSat(matchedDF):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF.loc[idx, 'Detection'] = -1
        # False negatives occur if GHGSat does not record a detection
        # AND Stanford is releasing
        # For GHGSat we have added a QC filter column to identify detects and non-detects
        elif row['QC filter'] == 2 and row['cr_allmeters_scfh'] > 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF.loc[idx, 'Detection'] = 0
        # False positives occur if GHGSat does record a detection
        # AND Stanford is not releasing
        elif row['QC filter'] == 1 and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF.loc[idx, 'Detection'] = 0
        elif row['QC filter'] == 0 :
            matchedDF.loc[idx, 'tc_Classification'] = 'ER'  # ER = Error
            matchedDF.loc[idx, 'Detection'] = -1
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'TN'  # TN = True Negative
            matchedDF.loc[idx, 'Detection'] = 0 
        elif not row['PlumeSteady']:
            matchedDF.loc[idx, 'tc_Classification'] = 'NS'  # NS = Not Steady
            matchedDF.loc[idx, 'Detection'] = 0  
        else:
            matchedDF.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF.loc[idx, 'Detection'] = 1
            
    return matchedDF

def classifyDetections_Satellites(matchedDF):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF.loc[idx, 'Detection'] = -1
        # False negatives occur if Bridger does not record a detection 
        # AND Stanford is releasing
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] > 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF.loc[idx, 'Detection'] = 0
        # False positives occur if Bridger does record a detection 
        # AND Stanford is not releasing
        elif pd.notna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF.loc[idx, 'Detection'] = 0
        elif pd.isna(row['FacilityEmissionRate']) and row['cr_allmeters_scfh'] <= 0:
            matchedDF.loc[idx, 'tc_Classification'] = 'TN'  # TN = True Negative
            matchedDF.loc[idx, 'Detection'] = 0     
        elif not row['PlumeSteady']:
            matchedDF.loc[idx, 'tc_Classification'] = 'NS'  # NS = Not Steady
            matchedDF.loc[idx, 'Detection'] = 0  
        else:
            matchedDF.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF.loc[idx, 'Detection'] = 1
            
    return matchedDF


def classifyDetections_SOOFIE(matchedDF):
    """ Classify each pass as TP (True Positive), FN (False Negative), or NE (Not Established)
    :param matchedDF =  dataframe with passes matched to release events
    :return matchedDF = updated dataframe with each row classified (TP, FN, or NE)"""

    for idx, row in matchedDF.iterrows():
        if not row['PlumeEstablished']:
            # tc_Classification is a categorical string describing the classification, Detection is describes same thing with -1, 0, 1
            matchedDF.loc[idx, 'tc_Classification'] = 'NE'  # NE = Not Established
            matchedDF.loc[idx, 'Detection'] = -1
        # False negatives occur if Bridger does not record a detection
        # AND Stanford is releasing
        elif (pd.to_numeric(row['FacilityEmissionRate']) == 0) and (row['cr_allmeters_scfh'] > 0):
            matchedDF.loc[idx, 'tc_Classification'] = 'FN'  # FN = False Negative
            matchedDF.loc[idx, 'Detection'] = 0
        # False positives occur if Bridger does record a detection
        # AND Stanford is not releasing
        elif (pd.to_numeric(row['FacilityEmissionRate']) > 0) and (row['cr_allmeters_scfh'] <= 0):
            matchedDF.loc[idx, 'tc_Classification'] = 'FP'  # FP = False Positive
            matchedDF.loc[idx, 'Detection'] = 0
        elif (pd.to_numeric(row['FacilityEmissionRate']) == 0) and (row['cr_allmeters_scfh'] <= 0):
            matchedDF.loc[idx, 'tc_Classification'] = 'TN'  # TN = True Negative
            matchedDF.loc[idx, 'Detection'] = 0
        elif pd.isna(row['FacilityEmissionRate']):
            matchedDF.loc[idx, 'tc_Classification'] = 'ER'  # ER = Error
            matchedDF.loc[idx, 'Detection'] = 0
        elif not row['PlumeSteady']:
            matchedDF.loc[idx, 'tc_Classification'] = 'NS'  # NS = Not Steady
            matchedDF.loc[idx, 'Detection'] = 0
        else:
            matchedDF.loc[idx, 'tc_Classification'] = 'TP'  # TP = True Positive
            matchedDF.loc[idx, 'Detection'] = 1

    return matchedDF


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
        sonicDF_temp = sonicDF.drop('OperatorSet', 1)
        integrated = sonicDF_temp.apply(integrate.trapz)  # integrate all fields
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

def steadyPlume(meanPlume, instantPlume):
    #x = abs(1 - DivZeroCheck(meanPlume,instantPlume))
    #return x
    if abs(1 - DivZeroCheck(meanPlume,instantPlume)) > 0.1:
        return False
    else:
        return True

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

def assessUncertainty(df):
    # (InputReleaseRate, MeterOption, PipeDiamOption, TestLocation, NumberMonteCarloDraws, hist=0, units='kgh'):
    df['cr_kgh_CH4_mean30'] = np.nan
    df['cr_kgh_CH4_lower30'] = np.nan
    df['cr_kgh_CH4_upper30'] = np.nan
    df['cr_kgh_CH4_mean60'] = np.nan
    df['cr_kgh_CH4_lower60'] = np.nan
    df['cr_kgh_CH4_upper60'] = np.nan
    df['cr_kgh_CH4_mean90'] = np.nan
    df['cr_kgh_CH4_lower90'] = np.nan
    df['cr_kgh_CH4_upper90'] = np.nan    
    df['cr_kgh_CH4_mean300'] = np.nan
    df['cr_kgh_CH4_lower300'] = np.nan
    df['cr_kgh_CH4_upper300'] = np.nan 
    df['cr_kgh_CH4_mean600'] = np.nan
    df['cr_kgh_CH4_lower600'] = np.nan
    df['cr_kgh_CH4_upper600'] = np.nan 
    df['cr_kgh_CH4_mean900'] = np.nan
    df['cr_kgh_CH4_lower900'] = np.nan
    df['cr_kgh_CH4_upper900'] = np.nan
    #df['cr_noise_low'] = np.nan
    #df['cr_noise_high'] = np.nan

    mean30, std30, mean60, std60, mean90, std90, mean300, std300, mean600, std600, mean900, std900 = assessVariability(df)
    
    for idx, row in df.iterrows():
        # If row was hand written add an additional uncertainty term
        if row['Flag_field_recorded'] == True:
            field_recorded_mean = mean30
            field_recorded_std = std30
        else:
            field_recorded_mean = 1
            field_recorded_std = 0            
        
        ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty(row['cr_scfh_mean30'], row['MeterCode'], row['PipeSize_inch'], row['TestLocation'], row['OperatorSet'],
                                                                                                  field_recorded_mean,
                                                                                                  field_recorded_std,
                                                                                                  NumberMonteCarloDraws = 10000,
                                                                                                  hist=0, 
                                                                                                  units='kgh')
        df.loc[idx, 'cr_kgh_CH4_mean30'] = ObservationStats[0]
        df.loc[idx, 'cr_kgh_CH4_lower30'] = ObservationStats[1]
        df.loc[idx, 'cr_kgh_CH4_upper30'] = ObservationStats[2]

    for idx, row in df.iterrows():
        # If row was hand written add an additional uncertainty term
        if row['Flag_field_recorded'] == True:
            field_recorded_mean = mean60
            field_recorded_std = std60
        else:
            field_recorded_mean = 1
            field_recorded_std = 0     
        
        ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty(
                                                                                            row['cr_scfh_mean60'],
                                                                                            row['MeterCode'],
                                                                                            row['PipeSize_inch'],
                                                                                            row['TestLocation'],
                                                                                            row['OperatorSet'],
                                                                                            field_recorded_mean,
                                                                                            field_recorded_std,
                                                                                            NumberMonteCarloDraws = 500,
                                                                                            hist=0,
                                                                                            units='kgh')
        df.loc[idx, 'cr_kgh_CH4_mean60'] = ObservationStats[0]
        df.loc[idx, 'cr_kgh_CH4_lower60'] = ObservationStats[1]
        df.loc[idx, 'cr_kgh_CH4_upper60'] = ObservationStats[2]
        #df.loc[idx, 'cr_noise_low'] = UncertaintyStats[1]
        #df.loc[idx, 'cr_noise_high'] = UncertaintyStats[2]

    for idx, row in df.iterrows():
        # If row was hand written add an additional uncertainty term
        if row['Flag_field_recorded'] == True:
            field_recorded_mean = mean90
            field_recorded_std = std90
        else:
            field_recorded_mean = 1
            field_recorded_std = 0     
        
        ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty(row['cr_scfh_mean90'], row['MeterCode'], row['PipeSize_inch'], row['TestLocation'], row['OperatorSet'],
                                                                                                  field_recorded_mean,
                                                                                                  field_recorded_std,
                                                                                                  NumberMonteCarloDraws = 500,
                                                                                                  hist=0, 
                                                                                                  units='kgh')
        df.loc[idx, 'cr_kgh_CH4_mean90'] = ObservationStats[0]
        df.loc[idx, 'cr_kgh_CH4_lower90'] = ObservationStats[1]
        df.loc[idx, 'cr_kgh_CH4_upper90'] = ObservationStats[2]

    for idx, row in df.iterrows():
        # If row was hand written add an additional uncertainty term
        if row['Flag_field_recorded'] == True:
            field_recorded_mean = mean300
            field_recorded_std = std300
        else:
            field_recorded_mean = 1
            field_recorded_std = 0     
        
        ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty(row['cr_scfh_mean300'], row['MeterCode'], row['PipeSize_inch'], row['TestLocation'], row['OperatorSet'],
                                                                                                  field_recorded_mean,
                                                                                                  field_recorded_std,
                                                                                                  NumberMonteCarloDraws = 500,
                                                                                                  hist=0, 
                                                                                                  units='kgh')
        df.loc[idx, 'cr_kgh_CH4_mean300'] = ObservationStats[0]
        df.loc[idx, 'cr_kgh_CH4_lower300'] = ObservationStats[1]
        df.loc[idx, 'cr_kgh_CH4_upper300'] = ObservationStats[2]

    for idx, row in df.iterrows():
        # If row was hand written add an additional uncertainty term
        if row['Flag_field_recorded'] == True:
            field_recorded_mean = mean600
            field_recorded_std = std600
        else:
            field_recorded_mean = 1
            field_recorded_std = 0     
        
        ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty(row['cr_scfh_mean600'], row['MeterCode'], row['PipeSize_inch'], row['TestLocation'], row['OperatorSet'],
                                                                                                  field_recorded_mean,
                                                                                                  field_recorded_std,
                                                                                                  NumberMonteCarloDraws = 500,
                                                                                                  hist=0, 
                                                                                                  units='kgh')
        df.loc[idx, 'cr_kgh_CH4_mean600'] = ObservationStats[0]
        df.loc[idx, 'cr_kgh_CH4_lower600'] = ObservationStats[1]
        df.loc[idx, 'cr_kgh_CH4_upper600'] = ObservationStats[2]

    for idx, row in df.iterrows():
        # If row was hand written add an additional uncertainty term
        if row['Flag_field_recorded'] == True:
            field_recorded_mean = mean900
            field_recorded_std = std900
        else:
            field_recorded_mean = 1
            field_recorded_std = 0

        ObservationStats, ObservationStatsNormed, ObservationRealizationHolder = meterUncertainty(row['cr_scfh_mean900'], row['MeterCode'], row['PipeSize_inch'], row['TestLocation'], row['OperatorSet'],
                                                                                                  field_recorded_mean,
                                                                                                  field_recorded_std,
                                                                                                  NumberMonteCarloDraws=500,
                                                                                                  hist=0,
                                                                                                  units='kgh')
        df.loc[idx, 'cr_kgh_CH4_mean900'] = ObservationStats[0]
        df.loc[idx, 'cr_kgh_CH4_lower900'] = ObservationStats[1]
        df.loc[idx, 'cr_kgh_CH4_upper900'] = ObservationStats[2]

    return df


def assessVariability(df):
        # THis function calculates variability between the rolling average release rate and the instantaneous release rate and returns the mean and the standard deviation
        
        df = df[(df['PlumeEstablished'] == True) & (df['PlumeSteady'] == True) & (df['cr_allmeters_scfh'] >0)]
        
        frac30 = df['cr_scfh_mean30']/df['cr_allmeters_scfh']
        mean30 = frac30.mean()
        std30 = frac30.std()
        
        frac60 = df['cr_scfh_mean60']/df['cr_allmeters_scfh']
        mean60 = frac60.mean()
        std60 = frac60.std()
        
        frac90 = df['cr_scfh_mean90']/df['cr_allmeters_scfh']
        mean90 = frac90.mean()
        std90 = frac90.std() 

        frac300 = df['cr_scfh_mean300']/df['cr_allmeters_scfh']
        mean300 = frac300.mean()
        std300 = frac300.std() 

        frac600 = df['cr_scfh_mean600']/df['cr_allmeters_scfh']
        mean600 = frac600.mean()
        std600 = frac600.std() 

        frac900 = df['cr_scfh_mean900']/df['cr_allmeters_scfh']
        mean900 = frac900.mean()
        std900 = frac900.std()

        return mean30, std30, mean60, std60, mean90, std90, mean300, std300, mean600, std600, mean900, std900
    
        
        