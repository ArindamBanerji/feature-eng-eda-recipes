import os, sys, getopt
import time
import argparse
from pathlib import Path, PureWindowsPath
import unittest
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import importlib, inspect
import uuid
import img2pdf
from pyaml_env import parse_config

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from mlxtend.feature_selection import SequentialFeatureSelector as SFS_mlxtnd
from sklearn.feature_selection import SequentialFeatureSelector as SFS_skl

import sweetviz as sv   #analyzing the dataset

def proj_path_nm() :
    return (os.getenv('BASEPROJDIR') + "\\" + os.getenv('CURPROJ')) 

def yaml_path() :
    return (proj_path_nm() + "\\" + os.getenv('CURYAMLFILE'))

# func 1 - return time 
def get_current_time(time_format):
        current_struct_time = time.localtime(time.time())
        current_time = time.strftime(time_format, current_struct_time)
        return current_time
    
# func 2    
# returns a list of image files in the given directory (at present only png files)
def get_list_of_files (base_dir, extension_type) :

# first juist check what filesare in the given irectory 
    fnm = os.path.join(os.getcwd(), base_dir)
    print ("plots fnm is ", fnm) 
    files = os.listdir(fnm)
 
    ext_type = "*" + extension_type

 # now get a list that hss alll  thepng files    
    images = Path(fnm).glob(ext_type)
    image_strings = [str(p) for p in images]
    for f_img in image_strings :
	    print(f_img)
    
    return image_strings

# func 3 - convert png  to pdf 
def convert_images_to_pdf (base_from_current_dir,  image_type, output_nm) :
    
    print ( "convert - 1",base_from_current_dir,  image_type, output_nm)

# get list of files from given directory    
    imagelist = get_list_of_files(base_from_current_dir, image_type)
    
    with open(output_nm, "wb") as f:
        f.write(img2pdf.convert([i for i in imagelist if i.endswith(".png")]))
  
  
# func4 - gen file name - new         
# Given base name & dir- add etension name + generate new file name
def generate_filename(base_dir, pt_nm, ext):
    
    data = dict()
# data['c_time'] = get_current_time('%Y%m%d_%H%M')
    data['base_dir'] = pt_nm
    data['uid'] = uuid.uuid4().hex
    
    fnm = '_'.join(data.values()) + ext 
    return os.path.join(os.getcwd(), base_dir, fnm)

# func 5 - show the output or save to file 
def output_show (plt_func, mode) : 
    print ("IN  OUTPUTSHOW ")
    if (mode == True) :
        plt_func.show(block=True)
    else :
        fnm_s = generate_filename("plots", "images", "png")
        print (fnm_s)
        plt_func.savefig (fnm_s)
        

# func 6 - remove any previously created files, if needed
def chk_remove_images ( base_dir, ext_type) :
    
    val = input("should previous plots be removed: [y/N] ")
    print(val)

    if ( val == "Y" or val == "y") :
        print ("wil delete prior image outputs")
        image_list = get_list_of_files (base_dir, ext_type)

        for file_img in image_list :
            print(file_img)
            os.remove(file_img)
            
    elif ( val == "N" or val == "n") :
        print ("wil NOT delete prior image outputs")
        
    else : 
        print ("INCORECT INPUT - defaults to NO deletion of prior image outputs")
        
# func 7 - generate data file name
def get_full_datapath_nm (base_dir, data_dir, data_file_nm) :
       
    does_file_exist = False
    incl_path  = base_dir + "\\" + data_dir + "\\" + data_file_nm  


    filename = PureWindowsPath(incl_path)

    # Convert path to the right format for the current operating system
    correct_fnm = Path(filename)
 
    if correct_fnm.is_file():
        print ("Full path NM exists ", correct_fnm)
        does_file_exist = True
    else :
        print ("Full Data path NM does NOT exist ", correct_fnm)
    
    return correct_fnm, does_file_exist
        
# func 8 
def read_df_from_file ( csv_fnm, set_nrows, nrws ) :
    config = read_yaml_conf(yaml_path())

    fnm, exists = get_full_datapath_nm (config['current_proj_dir'], 
                                        config['data_dir_nm'], csv_fnm)
                                        
    print ("full_path nm -from read_df", fnm)

    
    if (exists ==  False) :
        print ("file does not exist", csv_fnm)
        data = None
    # load our first dataset
    elif set_nrows :
            data = pd.read_csv(fnm, nrows=nrws)
    else :
            data = pd.read_csv(fnm)
    
    return data


# Func 9 - save this - important full function
# given a odule name & its path - return a dict with function_nms + package
def check_module_members (module_nm, package_path) :
    pkg = importlib.import_module(module_nm, package=package_path)
    name_func_tuples = inspect.getmembers(pkg, inspect.isfunction)
    functions = dict(name_func_tuples)
    
    return pkg, functions

# func 10 - read yaml file & get the the configuration dict
def read_yaml_conf(path_to_yaml) :
    return parse_config(path_to_yaml)


# Func 11 - generate corr matrix
# remove correlated features to reduce the feature space
def gen_correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

# Func 12 within the SFS we indicate:
# 1) the algorithm we want to create, in this case RandomForests
# (note that I use few trees to speed things up)
# 2) the stopping criteria: want to select 50 features
# 3) wheter to perform step forward or step backward
# 4) the evaluation metric: in this case the roc_auc
# 5) the want cross-validation
# this is going to take a while, do not despair
def  do_bkwd_fwd_selection (estimator, k_features,
                            forward, verbose, scoring, cv, path_to_yaml) :
    
    conf = read_yaml_conf(path_to_yaml=path_to_yaml) 
  
    
    if (conf['project_parms']['use_mlxtnd'] == "True") :
        print ("Calling mlxtnd libs ")
        sfs = SFS_mlxtnd (estimator=estimator,
                    k_features=k_features, # the lower the features we want, the longer this will take
                    forward=forward,
                    floating=False,
                    verbose=verbose,
                    scoring=scoring,
                    cv=cv)
    else :  
        direction = 'backward' if (forward == 'False') else 'forward'
        print ("Calling sklearn libs ")  
        sfs = SFS_skl (estimator=estimator,
                    n_features_to_select=k_features,
                    direction = direction, 
                    scoring=scoring,  
                    cv=cv,  
                    n_jobs=4   )
    
    return sfs


# function to train random forests and evaluate the performance
# Func 13 - do random forest runs 
def run_randomForestClassifier(X_train, X_test, y_train, y_test, path_to_yaml):
    conf = read_yaml_conf(path_to_yaml=path_to_yaml)
    rf = RandomForestClassifier(n_estimators=conf['RandomForestConfig']['n_estimators'],
                                random_state=conf['RandomForestConfig']['rand_state'],
                                max_depth=conf['RandomForestConfig']['max_depth'] )
    
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict_proba(X_train)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    
    print('Test set')
    pred = rf.predict_proba(X_test)
    print('Random Forests roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))
    return pred

# Func 14 - function to train random forests and evaluate the performance
def run_randomForestRegressor(X_train, X_test, y_train, y_test, path_to_yaml):
    conf = read_yaml_conf(path_to_yaml=path_to_yaml)
    rf = RandomForestRegressor(n_estimators=conf['RandomForestConfig']['n_estimators'],
                                random_state=conf['RandomForestConfig']['rand_state'],
                                max_depth=conf['RandomForestConfig']['max_depth'] )
        
    rf.fit(X_train, y_train)

    print('Train set')
    pred = rf.predict(X_train)
    print('Random Forests roc-auc: {}'.format(r2_score(y_train, pred)))
    
    print('Test set')
    pred = rf.predict(X_test)
    print('Random Forests roc-auc: {}'.format(r2_score(y_test, pred)))
    
    return pred


# Func 15 create a function to train a logistic regression 
# and compare its performance in the train and test sets
def run_logistic(X_train, X_test, y_train, y_test, C, max_iter, penalty):
    
    scaler = StandardScaler().fit(X_train)
    
    logit = LogisticRegression(C=C, random_state=10, max_iter=max_iter, penalty=penalty)
    logit.fit(scaler.transform(X_train), y_train)
    
    print('Train set')
    pred = logit.predict_proba(scaler.transform(X_train))
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_train, pred[:,1])))
    
    print('Test set')
    pred = logit.predict_proba(scaler.transform(X_test))
    print('Logistic Regression roc-auc: {}'.format(roc_auc_score(y_test, pred[:,1])))

    return pred

# Func - Remove all duplicates 
def remove_duplicates (X_train, X_test, drop_dup) :
    duplicated_feat = []

    for i in range(0, len(X_train.columns)):
        if i % 10 == 0:  # this helps me understand how the loop is going
            print(i)

        col_1 = X_train.columns[i]

        for col_2 in X_train.columns[i + 1:]:
            if X_train[col_1].equals(X_train[col_2]):
                duplicated_feat.append(col_2)
                
    len(duplicated_feat)
    
    if (drop_dup == True) :
        # remove duplicated features
        X_train.drop(labels=duplicated_feat, axis=1, inplace=True)
        X_test.drop(labels=duplicated_feat, axis=1, inplace=True)

    print (X_train.shape, X_test.shape)
    
    return duplicated_feat

# Func 17 - drop constant features 
def   drop_const_features(X_train, X_test, drop_feat) :
    constant_features = [feat for feat in X_train.columns if X_train[feat].std() == 0]

    if (drop_feat == 'True') :
        X_train.drop(labels=constant_features, axis=1, inplace=True)
        X_test.drop(labels=constant_features, axis=1, inplace=True)

    print (X_train.shape, X_test.shape) 
    
    return constant_features

 
 # Func 18 - drop quasi constant features 
def drop_quasi_const_features (threshold, X_train, X_test, drop_feat) :
    
    sel = VarianceThreshold(threshold=0.01)  # 0.1 indicates 99% of observations approximately
    
    sel.fit(X_train)  # fit finds the features with low variance

    sum(sel.get_support()) # how many not quasi-constant?
    features_to_keep = X_train.columns[sel.get_support()]
    
    if (drop_feat == 'True') :
        X_train = sel.transform(X_train)
        X_test = sel.transform(X_test)

    print (X_train.shape, X_test.shape)
    
    # I transform the NumPy arrays to dataframes
    X_train= pd.DataFrame(X_train)
    X_train.columns = features_to_keep

    X_test= pd.DataFrame(X_test)
    X_test.columns = features_to_keep
    
    return X_train, X_test, features_to_keep

def remove_const_quasi_const (threshold, X_train, X_test, drop_feat) :
    quasi_constant_feat = []

    print ("cont & qusi const", threshold)

    # iterate over every feature
    for feature in X_train.columns:

        # find the predominant value, that is the value that is shared
        # by most observations
        predominant = X_train[feature].value_counts(
            normalize=True).sort_values(ascending=False).values[0]

        # evaluate the predominant feature: do more than 99% of the observations
        # show 1 value?
        if predominant > threshold :

            # if yes, add the variable to the list
            quasi_constant_feat.append(feature)

    
    if (drop_feat == 'True') :
        X_train.drop(labels=quasi_constant_feat, axis=1, inplace=True)
        X_test.drop(labels=quasi_constant_feat, axis=1, inplace=True)

    print (X_train.shape, X_test.shape)

    return X_train, X_test, quasi_constant_feat


# plotfeature importance for xgboost etc.
def plot_feature_importance(importance,names,model_type):

	#Create arrays from feature importance and feature names
	feature_importance = np.array(importance)
	feature_names = np.array(names)

	#Create a DataFrame using a Dictionary
	data={'feature_names':feature_names,'feature_importance':feature_importance}
	fi_df = pd.DataFrame(data)

	#Sort the DataFrame in order decreasing feature importance
	fi_df.sort_values(by=['feature_importance'], ascending=False,inplace=True)

	#Define size of bar plot
	plt.figure(figsize=(10,8))
	#Plot Searborn bar chart
	sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
	#Add chart labels
	plt.title(model_type + 'FEATURE IMPORTANCE')
	plt.xlabel('FEATURE IMPORTANCE')
	plt.ylabel('FEATURE NAMES')


def gen_swviz_eda_from_fnm(filenm, pairwise_analysis ) :
    
    # read in data_set 
    data = read_df_from_file (filenm,
                              set_nrows=False, nrws=0 ) 
    print ("swiz data shape:", data.shape) 
        
    return sv.analyze(data, pairwise_analysis=pairwise_analysis) # create the report


def gen_swviz_eda_from_dataframe( data, pairwise_analysis ) :
    
    layout = 'vertical'
    scale = 1.25
    
    return sv.analyze(data, pairwise_analysis=pairwise_analysis) # create the report


class TestHelperFe(unittest.TestCase):

    def test_pdf_fnm(self):
        #test code 1
# generate an output pdf file name
        output_str = get_current_time('%Y%m%d_%H%M')
        output_nm = "output_" + output_str + ".pdf"
        print ( "output  file nm is ", output_nm)

#test code 2 
    def test_to_pdf() :
        convert_images_to_pdf ("plots", ".png", output_nm)

#test code 3
    def test_remove_images() :
        chk_remove_images ("plots", ".png")


#test code 4
    def test_read() :
        data = read_df_from_file ( "fselect_dataset_2.csv", set_nrows=False, nrws=0 ) 
        print (data.shape)


# test code 5
    def test_member_list() :
        pkg, func = check_module_members ('helper_fe_v2', proj_path_nm())
        print ("func_keys ", func.keys())
        print ("\n\n \n dir  on pkg ", dir(pkg))
        
        
# test code 6 
    def test_yaml_config():
        path_to_yaml = yaml_path()
        print(path_to_yaml)
        conf = read_yaml_conf(path_to_yaml=path_to_yaml) 
        print (conf)



if __name__ == '__main__':
    TestHelperFe.test_yaml_config()
