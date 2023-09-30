
import datetime
import pytz
import pandas as pd
import os

from loaddata import loaddata
from matchMethods import performMatching

#def main():

operatorDF, meterDF_All, sonicDF_All = loaddata()
matchedDF_Bridger, matchedDF_GHGSat, matchedDF_CarbonMapper, matchedDF_MAIR, matchedDF_Satellites, matchedDF_SOOFIE = performMatching(operatorDF, meterDF_All, sonicDF_All)


cwd = os.getcwd()

csvPath = os.path.join(cwd, 'matchedDF_Bridger_23822_300m.csv')
matchedDF_Bridger.to_csv(csvPath)

csvPath = os.path.join(cwd, 'matchedDF_GHGSat_23822_300m.csv')
matchedDF_GHGSat.to_csv(csvPath)

csvPath = os.path.join(cwd, 'matchedDF_CarbonMapper_23822_300m.csv')
matchedDF_CarbonMapper.to_csv(csvPath)

x = 1





#if __name__ == '__main__':
#    main()
