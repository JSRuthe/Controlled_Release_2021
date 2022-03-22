
import datetime
import pytz
import pandas as pd
import os

from loaddata import loaddata
from matchMethods import performMatching

#def main():

operatorDF, meterDF_All, sonicDF_All = loaddata()
    
matchedDF_Bridger, matchedDF_GHGSat, matchedDF_CarbonMapper, matchedDF_MAIR, matchedDF_Satellites = performMatching(operatorDF, meterDF_All, sonicDF_All)

cwd = os.getcwd()


#csvPath = os.path.join(cwd, 'matchedDF_Bridger_warning_fix.csv')
#matchedDF_Bridger.to_csv(csvPath)

#csvPath = os.path.join(cwd, 'matchedDF_GHGSat_update.csv')
#matchedDF_GHGSat.to_csv(csvPath)

#csvPath = os.path.join(cwd, 'matchedDF_CarbonMapper_1sigma.csv')
#matchedDF_CarbonMapper.to_csv(csvPath)

csvPath = os.path.join(cwd, 'matchedDF_MAIR.csv')
matchedDF_MAIR.to_csv(csvPath)

#csvPath = os.path.join(cwd, 'meterDF_All.csv')
#meterDF_All.to_csv(csvPath)

#csvPath = os.path.join(cwd, 'matchedDF_Satellites_22315.csv')
#matchedDF_Satellites.to_csv(csvPath)

# COlumn names for export to teams:
    
cols = [
    "Stanford_timestamp",
    "cr_kgh_CH4_mean90", 
    "cr_kgh_CH4_lower90", 
    "cr_kgh_CH4_upper90", 
    "FacilityEmissionRate", 
    "FacilityEmissionRateUpper", 
    "FacilityEmissionRateLower", 
    "UnblindingStage", 
    "PipeSize_inch", 
    "MeterCode", 
    "PlumeEstablished", 
    "PlumeSteady", 
    "cr_kgh_CH4_mean30", 
    "cr_kgh_CH4_lower30", 
    "cr_kgh_CH4_upper30", 
    "cr_kgh_CH4_mean60", 
    "cr_kgh_CH4_lower60", 
    "cr_kgh_CH4_upper60", 
    "Operator_Timestamp"]

matchedDF_MAIR_toTeam = matchedDF_MAIR.reindex(columns = cols)
csvPath = os.path.join(cwd, 'matchedDF_MAIR_unblindedToMAIR.csv')
matchedDF_MAIR_toTeam.to_csv(csvPath)

date_start = pd.to_datetime('2021.07.30 00:00:00')
date_start = date_start.tz_localize('UTC')
date_end = pd.to_datetime('2021.08.04 00:00:00')
date_end = date_end.tz_localize('UTC')
meterDF_MAIR_toTeam = meterDF_All[(meterDF_All.index > date_start) & (meterDF_All.index < date_end)]
csvPath = os.path.join(cwd, 'meterDF_MAIR_unblindedToMAIR.csv')
meterDF_MAIR_toTeam.to_csv(csvPath)

#matchedDF_CarbonMapper_toTeam = matchedDF_CarbonMapper.reindex(columns = cols)
#csvPath = os.path.join(cwd, 'matchedDF_CarbonMapper_unblindedToCM.csv')
#matchedDF_CarbonMapper_toTeam.to_csv(csvPath)

#date_start = pd.to_datetime('2021.07.30 00:00:00')
#date_start = date_start.tz_localize('UTC')
#date_end = pd.to_datetime('2021.08.04 00:00:00')
#date_end = date_end.tz_localize('UTC')
#meterDF_CarbonMapper_toTeam = meterDF_All[(meterDF_All.index > date_start) & (meterDF_All.index < date_end)]
#csvPath = os.path.join(cwd, 'meterDF_CarbonMapper_unblindedToCM.csv')
#meterDF_CarbonMapper_toTeam.to_csv(csvPath)

# matchedDF_GHGSat_toTeam = matchedDF_GHGSat.reindex(columns = cols)
# csvPath = os.path.join(cwd, 'matchedDF_GHGSat_unblindedToGHGSat.csv')
# matchedDF_GHGSat_toTeam.to_csv(csvPath)

# date_start = pd.to_datetime('2021.10.18 00:00:00')
# date_start = date_start.tz_localize('UTC')
# date_end = pd.to_datetime('2021.10.23 00:00:00')
# date_end = date_end.tz_localize('UTC')
# meterDF_GHGSat_toTeam = meterDF_All[(meterDF_All.index > date_start) & (meterDF_All.index < date_end)]
# csvPath = os.path.join(cwd, 'meterDF_GHGSat_unblindedToGHGSat.csv')
# meterDF_GHGSat_toTeam.to_csv(csvPath)

#matchedDF_Bridger_toTeam = matchedDF_Bridger.reindex(columns = cols)
#csvPath = os.path.join(cwd, 'matchedDF_Bridger_unblindedToBridger.csv')
#matchedDF_Bridger_toTeam.to_csv(csvPath)

#date_start = pd.to_datetime('2021.11.03 00:00:00')
#date_start = date_start.tz_localize('UTC')
#date_end = pd.to_datetime('2021.11.05 00:00:00')
#date_end = date_end.tz_localize('UTC')
#meterDF_Bridger_toTeam = meterDF_All[(meterDF_All.index > date_start) & (meterDF_All.index < date_end)]
#csvPath = os.path.join(cwd, 'meterDF_Bridger_unblindedToBridger.csv')
#meterDF_Bridger_toTeam.to_csv(csvPath)

    # write matched results to csv
#    cwd = os.getcwd()

    #csvPath = os.path.join(cwd, 'MidlandTestAnalysisResults', 'StanfordMatchedPasses.csv')
    #matchedDF.to_csv(csvPath)




#if __name__ == '__main__':
#    main()
