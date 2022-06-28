import os
from Essential import global_params

case = global_params.CASES


def GetThisDir():
    return os.path.dirname( os.path.abspath(__file__) )

def GetDataSource():
    return os.path.abspath(
        os.path.join(GetThisDir(), os.pardir, "Data_source")
    ) 

def GetRawData():
    return os.path.join(GetDataSource(), "Raw")

def GetProcessedData():
    return os.path.join(GetDataSource(), "Processed")

def GetModelsData():
    return os.path.join(GetDataSource(), "Models")

def GetResultsData():
    return os.path.join(GetDataSource(), "Results")

def GetTuningData():
    return os.path.join(GetResultsData(), "Tuning")

def GetModelPerformancesData():
    return os.path.join(GetResultsData(), "Model_performances")

def GetPreprocessedData():
    return os.path.join(GetResultsData(), "Preprocessed")

def GetMachVarModel():
    return os.path.join(GetModelsData(),case[0] )

def GetVFVarModel():
    return os.path.join(GetModelsData(),case[1] )

def InitDataDirectories():
    important_dir = global_params.DIRECTORIES

    for dir in important_dir:
        full_path = os.path.join (GetThisDir(), os.pardir, dir)
        full_path = os.path.abspath (full_path)

        if not os.path.exists(full_path):
            os.makedirs(full_path)