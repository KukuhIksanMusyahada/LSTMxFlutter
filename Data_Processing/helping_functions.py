import os
import re

#extracting Mach and Flutter Velocity

def extractMachAndVF(file: str):
    pattern = r"M_([0-9\.]*)_VF_([0-9\.]*).csv"
    result = re.match(pattern, file)

    return result.group(1), result.group(2)

# Creating Numpy Dataset

def create_dataset(no_row, no_column, datapath, use_cols):
    data = []
    count= 0
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(datapath, file), nrows= no_row)
            df_np = df[use_cols].to_numpy()
            data.append(df_np)
            count +=1
    dataset = np.stack(data, axis=1)
    dataset = dataset.reshape(no_row, count*no_column)
    return dataset



# labelling function

def labelling(datapath):
    labels = []
    pred_labels= []
    for file in os.listdir(datapath):
        if file.endswith(".csv"):
            MachAndVF= extractMachAndVF(file)
            label= f'M_{MachAndVF[0]}_VF_{MachAndVF[1]}'
            labels.append(label)
            pred_label= f'Pred M_{MachAndVF[0]}_VF_{MachAndVF[1]}'
            pred_labels.append(pred_label)
    return labels, pred_labels

# dataframe label

def label_df(Labels:list, no_case:int):    
    labels=[]
    cases = case_model[no_case]
    if no_case == 1 or no_case ==2:
        for label in Labels:
            for case in cases:
                label_name= f'{case}_{label}'
                labels.append(label_name)
        return labels
    else:
        print('not supported cases!')