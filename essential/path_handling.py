import os
from TA_LSTMxFlutter.essential import global_params

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