

from loaddata import loaddata
from matchMethods import performMatching

import os

#cwd = os.getcwd()

#def main():

    # Generate table of Stanford matched controlled releases
    # (1) Load data
operatorDF, meterDF_All, sonicDF_All = loaddata()
    # (2) match bridger data with release data
    
matchedDF_Bridger, matchedDF_GHGSat, matchedDF_CarbonMapper = performMatching(operatorDF, meterDF_All, sonicDF_All)


    # write matched results to csv
#    cwd = os.getcwd()

    #csvPath = os.path.join(cwd, 'MidlandTestAnalysisResults', 'StanfordMatchedPasses.csv')
    #matchedDF.to_csv(csvPath)




#if __name__ == '__main__':
#    main()
