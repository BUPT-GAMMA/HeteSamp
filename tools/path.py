import os

def generate_specific_model_path(model_path, basic_parameters):
    return model_path + "_".join(basic_parameters) + "/"

def create_directory(path_name):
    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print("Complete create directory:\t{}".format(path_name))
    else:
        print("The directory:\t {} has been created".format(path_name))
    return 0