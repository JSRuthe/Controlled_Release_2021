
from alicatMethods import computeControlledReleases
from loadData import loadData
from matchMethods import performMatching
from loaddata_Stanford import loaddata_Stanford
from matchMethods_Stanford import performMatching_Stanford
from mergeTables import mergeMatchedTables
from plotMethods import plotMain
import os

def main():
    cr_averageperiod_sec = 65

    # Generate table of Stanford matched controlled releases
    # (1) Load data
    Stanford_bridgerDF, Stanford_quadrathermDF, Stanford_sonicDF = loaddata_Stanford(cr_averageperiod_sec=cr_averageperiod_sec)
    # (2) match bridger data with release data
    Stanford_matched = performMatching_Stanford(Stanford_bridgerDF, Stanford_quadrathermDF, Stanford_sonicDF,
                                                minPlumeLength=150, cr_averageperiod_sec=cr_averageperiod_sec, CH4_frac=0.962)

    # Generate table of CSU matched controlled releases
    # (1) Load data
    CSU_flightTrackDF, CSU_bridgerDF, CSU_alicat250DF, CSU_alicat5000DF, CSU_quadrathermDF, CSU_sonicDF, CSU_cupDF = loadData()
    # (2) find start of controlled releases from Alicat data
    CSU_ControlledReleases = computeControlledReleases(CSU_alicat250DF, CSU_alicat5000DF)
    # (3) match bridger data with release data
    CSU_matched = performMatching(CSU_bridgerDF, CSU_ControlledReleases, CSU_sonicDF, CSU_alicat250DF, CSU_alicat5000DF, minPlumeLength=150, cr_averageperiod_sec=cr_averageperiod_sec)

    # write matched results to csv
    cwd = os.getcwd()
    csvPath = os.path.join(cwd, 'MidlandTestAnalysisResults', 'CSUMatchedPasses.csv')
    CSU_matched.to_csv(csvPath)
    csvPath = os.path.join(cwd, 'MidlandTestAnalysisResults', 'StanfordMatchedPasses.csv')
    Stanford_matched.to_csv(csvPath)

    # merge stanford and csu data tables
    matched = mergeMatchedTables(CSU_matched, Stanford_matched)
    csvPath = os.path.join(cwd, 'MidlandTestAnalysisResults', 'AllMatchedPasses.csv')
    matched.to_csv(csvPath)


    # generate plots
    # (broken until I handle new column headers - 2/18/2022)
    #plotMain(matched)


if __name__ == '__main__':
    main()
