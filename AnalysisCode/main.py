

from loaddata import loaddata
from matchMethods import performMatching

import os

def main():

    # Generate table of Stanford matched controlled releases
    # (1) Load data
    operatorDF, meterDF_All, sonicDF = loaddata()
    # (2) match bridger data with release data
    
    matchedDF = performMatching(operatorDF, meterDF_All, sonicDF,
                                                minPlumeLength=150, cr_averageperiod_sec=cr_averageperiod_sec, CH4_frac=0.962)


    # write matched results to csv
    cwd = os.getcwd()

    csvPath = os.path.join(cwd, 'MidlandTestAnalysisResults', 'StanfordMatchedPasses.csv')
    matchedDF.to_csv(csvPath)




if __name__ == '__main__':
    main()