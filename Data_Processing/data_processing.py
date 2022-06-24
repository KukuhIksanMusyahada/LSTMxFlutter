import os
import numpy as np
import pandas as pd

from TA_LSTMxFlutter.essential import global_params as gp
from TA_LSTMxFlutter.essential import path_handling as ph
from TA_LSTMxFlutter.Data_Processing.helping_functions import *




# Importing Dataset
def import_data(no_case= None, no_model= None , vf_case=None, datapath=ph.GetRawData()):
    if no_case == 0:
        data_case = os.path.join(datapath, gp.CASES[no_case])

        if no_model == 1:
            use_cols = gp.CASES_MODELS[no_model]
            dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case, use_cols)
            Labels = labelling(data_case)
            return dataset, Labels
        elif no_model == 2:
            try:
                no_model +=1
                use_cols = gp.CASES_MODELS[no_model]
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case, use_cols)
                Labels = labelling(data_case)
                return dataset, Labels
            except:
                use_cols = gp.CASES_MODELS[no_model]
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case, use_cols)
                Labels = labelling(data_case)
                return dataset, Labels
        else:
            print('no_model is not supported in this model')
        
    elif no_case == 1:
        data_case = os.path.join(datapath, gp.CASES[no_case])

        if no_model == 1:
            use_cols = gp.CASES_MODELS[no_model]
            if vf_case == 0:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[0])
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 1:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[1])
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 2:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[2])
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 3:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[3])
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 4:
                all_data = []
                for i in range(vf_case):
                    data,label = import_data(no_case, no_model, i)
                    all_data.append(data)
                dataset = np.concatenate(all_data, axis=1)
                # Labels = labelling(data_case_VF)
                return dataset
            else:
                print('No Cases Supported!')

        elif no_model == 2:
            use_cols = gp.CASES_MODELS[no_model]
            if vf_case == 0:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[0])
                dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 1:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[1])
                try:
                    no_model +=1
                    use_cols = gp.CASES_MODELS[no_model]
                    dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
                except:
                    use_cols = gp.CASES_MODELS[no_model]
                    dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
            elif vf_case == 2:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[2])
                try:
                    no_model +=1
                    use_cols = gp.CASES_MODELS[no_model]
                    dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
                except:
                    use_cols = gp.CASES_MODELS[no_model]
                    dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
            elif vf_case == 3:
                data_case_VF = os.path.join(data_case, gp.VF_CASE[3])
                try:
                    no_model +=1
                    use_cols = gp.CASES_MODELS[no_model]
                    dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
                except:
                    use_cols = gp.CASES_MODELS[no_model]
                    dataset = create_dataset(gp.NO_ROW, gp.NO_COLUMNS, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
            elif vf_case == 4:
                all_data = []
                all_label=[]
                for i in range(vf_case):
                    data, label = import_data(no_case, no_model, i)
                    all_data.append(data)
                    all_label.append(label)
                
                dataset = np.concatenate(all_data, axis=1)
                Labels = np.concatenate(all_label, axis=1)
                return dataset, Labels
            else:
                print('No Cases Supported!')
        else:
            print('no_model is not supported in this model')
            
    else:
        print(
            "This project only work on 2 cases, \n \t case 0: Mach Variation,\n \t case 1: Velocity Flutter Variation"
        )

# interpolation function

def interpolate(dataset,size_row=gp.SIZE_ROW):
  
    x=np.arange(dataset.shape[0])
    xvals= np.linspace(0,dataset.shape[0],size_row)
    data_interp= []
    for col in range(dataset.shape[1]):
        y = dataset[:,col]
        yinterp= np.interp(xvals,x,y)
        data_interp.append(yinterp)
    data_interp= np.stack(data_interp, axis=1)
    data_interp.shape
    
    return data_interp


#Creating dataframe and save it

def creating_dataframe(names: str, no_case=None, no_model=None, vf_case=None,size_row=gp.SIZE_ROW, predicted= True, datapath= ph.GetProcessedData()):
    
    dataset, Labels= import_data(no_case,no_model,vf_case)
    
    if predicted:
        labels = label_df(Labels[1],no_model)
        df = pd.DataFrame(dataset, columns= labels)
    else:
        data_interp=  interpolate(dataset, size_row)
        labels = label_df(Labels[0],no_model)
        df = pd.DataFrame(data_interp, columns= labels)
    names= names+'.csv'
    df.to_csv(os.path.join(datapath,names), index= False)

max_cases = len(gp.CASES)
max_model_cases = len(gp.CASES_MODELS)
max_vf_cases = len(gp.VF_CASE)

# Data Processing
def data_process(max_cases= max_cases, max_model_cases= max_model_cases, max_vf_cases= max_vf_cases):
    for case in range(max_cases):
        if case ==0:
            for model_case in range(1,max_model_cases):
                data = import_data(case, model_case)
                names= f'{gp.CASES_MODELS[model_case][0]}_{gp.CASES_MODELS[model_case][1]}_{gp.CASES[case]}' 
                creating_dataframe(names, no_case=case, no_model=model_case, predicted= False)
        
        if case ==1:
            for model_case in range(1,max_model_cases):
                for vf_case in range(max_vf_cases):
                    data = import_data(case, model_case, vf_case,)
                    names= f'{gp.CASES_MODELS[model_case][0]}_{gp.CASES_MODELS[model_case][1]}_{gp.CASES[case]}_{gp.VF_CASE[vf_case]}' 
                    creating_dataframe(names, no_case=case, no_model=model_case, vf_case= vf_case, predicted=False)
            
def GetListTrainData(path= ph.GetProcessedData()):
    list_df = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            df= pd.read_csv(os.path.join(path, file))
            list_df.append(df)
    return list_df