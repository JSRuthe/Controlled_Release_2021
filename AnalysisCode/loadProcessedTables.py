import os.path
cwd = os.getcwd()
import pandas as pd

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_Bridger_2291_10kreals.csv')
matchedDF_Bridger = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_GHGSat_2291_10kreals.csv')
matchedDF_GHGSat = pd.read_csv(csvPath)

csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_CarbonMapper_1sigma_2291_10kreals.csv')
matchedDF_CarbonMapper = pd.read_csv(csvPath)

#csvPath = os.path.join(cwd, 'Dataframes for Stanford analysis', 'matchedDF_MAIR_2291_10kreals.csv')
#matchedDF_MAIR = pd.read_csv(csvPath)
