
# Cases
CASES = ["Mach_Variation", "VF_Variation"]
CASES_MODELS = {1: ["CL", "CD"], 2: ["pitch(airfoil)", "plunge(airfoil)"], 3:["pitch_airfoil", "plunge_airfoil"]}
VF_CASE = ["Mach_0.6", "Mach_0.7", "Mach_0.8", "Mach_0.9"]

# Data Operational
NO_ROW = 130
SIZE_ROW = 4000
NO_COLUMNS = len(CASES_MODELS[1])

#Directory Operational

DIRECTORIES = [
        "Data_source",
        "Data_source/Models",
        "Data_source/Raw",
        "Data_source/Raw/Mach_Variation",
        "Data_source/Raw/VF_Variation",
        "Data_source/Results",
        "Data_source/Results/Tuning",
        "Data_source/Results/Tuning/cell_tuning",
        "Data_source/Results/Tuning/layer_tuning",
        "Data_source/Results/Tuning/optimizer_tuning",
        "Data_source/Results/Tuning/learningRate_tuning",
        "Data_source/Results/Preprocessed",
        "Data_source/Results/Model_performances",
        "Data_source/Processed",
    ]