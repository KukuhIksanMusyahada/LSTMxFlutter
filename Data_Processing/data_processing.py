import os
import numpy as np
import pandas as pd


# Importing Dataset
def import_data(no_case: int, no_model: int, vf_case: int, datapath=data_raw):
    if no_case == 0:
        data_case = os.path.join(datapath, case[no_case])

        if no_model == 1:
            use_cols = case_model[no_model]
            dataset = create_dataset(no_row, no_column, data_case, use_cols)
            Labels = labelling(data_case)
            return dataset, Labels
        elif no_model == 2:
            try:
                no_model +=1
                use_cols = case_model[no_model]
                dataset = create_dataset(no_row, no_column, data_case, use_cols)
                Labels = labelling(data_case)
                return dataset, Labels
            except:
                use_cols = case_model[no_model]
                dataset = create_dataset(no_row, no_column, data_case, use_cols)
                Labels = labelling(data_case)
                return dataset, Labels
        else:
            print('no_model is not supported in this model')
        
    elif no_case == 1:
        data_case = os.path.join(datapath, case[no_case])

        if no_model == 1:
            use_cols = case_model[no_model]
            if vf_case == 0:
                data_case_VF = os.path.join(data_case, VF_case[0])
                dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 1:
                data_case_VF = os.path.join(data_case, VF_case[1])
                dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 2:
                data_case_VF = os.path.join(data_case, VF_case[2])
                dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 3:
                data_case_VF = os.path.join(data_case, VF_case[3])
                dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
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
            use_cols = case_model[no_model]
            if vf_case == 0:
                data_case_VF = os.path.join(data_case, VF_case[0])
                dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                Labels = labelling(data_case_VF)
                return dataset, Labels
            elif vf_case == 1:
                data_case_VF = os.path.join(data_case, VF_case[1])
                try:
                    no_model +=1
                    use_cols = case_model[no_model]
                    dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
                except:
                    use_cols = case_model[no_model]
                    dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
            elif vf_case == 2:
                data_case_VF = os.path.join(data_case, VF_case[2])
                try:
                    no_model +=1
                    use_cols = case_model[no_model]
                    dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
                except:
                    use_cols = case_model[no_model]
                    dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
            elif vf_case == 3:
                data_case_VF = os.path.join(data_case, VF_case[3])
                try:
                    no_model +=1
                    use_cols = case_model[no_model]
                    dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
                    Labels = labelling(data_case_VF)
                    return dataset, Labels
                except:
                    use_cols = case_model[no_model]
                    dataset = create_dataset(no_row, no_column, data_case_VF, use_cols)
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

#Creating dataframe and save it

def creating_dataframe(no_case: int, no_model: int, vf_case: int,size_row:int,names: str, predicted= True, datapath= data_processed):
    
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
    