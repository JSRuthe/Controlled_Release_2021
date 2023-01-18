
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







#if __name__ == '__main__':
#    main()
