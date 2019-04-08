import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from itertools import combinations 
import json
import pydot
from sklearn import tree
import graphviz
from graphviz import Source

from threading import Thread
import time
from collections import Counter
import multiprocessing as mp
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
import pickle
from sklearn.neighbors import KNeighborsClassifier

from multiprocessing import Manager
import multiprocessing.sharedctypes as sharedctypes
import ctypes

import multiprocessing

## from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB


def assign_to_dict(dic, key_list, value):
    current = dic
    try:
        for (index, key) in enumerate(key_list):
            if index == len(key_list) - 1:
                current[key] = value
                return
            elif current.get(key) is None:
                current[key] = {}
            current = current[key]
    except:
        import pdb; pdb.set_trace()

def get_classifier(name):
    if name == "dt":
        return DecisionTreeClassifier()
    if name == "dt3":
        return DecisionTreeClassifier(max_depth=3)
    if name == "dt5":
        return DecisionTreeClassifier(max_depth=5)
    # if name == "xgboost":
    #     return xgb.XGBClassifier(objective="binary:logistic")
    # if name == "xgboost3":
    #     return xgb.XGBClassifier(objective="binary:logistic", max_depth=3)
    # if name == "xgboost5":
        # return xgb.XGBClassifier(objective="binary:logistic", max_depth=5)
    if name == "rf":
        return RandomForestClassifier(n_estimators=10, max_features=1)
    if name == "rf3":
        return RandomForestClassifier(max_depth=3, n_estimators=10, max_features=1)
    if name == "rf5":
        return RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
    # if name == "svm":
    #     return SVC(kernel="linear", C=0.025)
    # if name == "kn":
    #     return KNeighborsClassifier(3)
    return "uname"

def train_and_inference(params):
    classifier_name = params["classifier_name"]
    can = params["can"]
    df_test_list = params["df_test_list"]
    test_csv_list = params["test_csv_list"]
    train_csv = params["train_csv"]
    df_train = params["df_train"]
    
    result = {
        "training": {
            "time": {

            }
        },
        "inference": {
            "time": {

            }, 
            "precision": {
            
            },
            "recall": {
                
            }
        }
    }

    #-----------------------------Training-------------------------------------------#


    classifier = get_classifier(classifier_name) 
    
    df_train_current = df_train[can]
    candidate_str = "+".join(can)
    start = time.time()
    classifier.fit(df_train_current.values, df_train["IsBot"])
    end = time.time()
    
    assign_to_dict(result, ["training", "time", \
        classifier_name, train_csv, candidate_str], end - start)
    print ("{0} Training of data is {1}s".format(classifier_name, str(end-start)) )
    with open("./model/"+classifier_name+"_"+str(hash(candidate_str))+"_"+str(hash(train_csv))+"cross-check.pkl", "wb") as fp:
        pickle.dump(classifier, fp)

    #-----------------------------Inference-------------------------------------------#


    for (index, df_test) in enumerate(df_test_list):
        df_test_current = df_test[can]
        
        with open("./model/"+classifier_name+"_"+str(hash(candidate_str))+"_"+str(hash(train_csv))+"cross-check.pkl", "rb") as fp:
            classifier = pickle.load(fp)
    
        start = time.time()
        predicted = classifier.predict(df_test_current.values)
        end = time.time()
        assign_to_dict(result, ["inference", "time", classifier_name, \
                train_csv, candidate_str, test_csv_list[index]], end - start)
        print (" {0} Inference of data {1} is {2}s".format(classifier_name, index, str(end-start)) )

        tn, fp, fn, tp = confusion_matrix(df_test["IsBot"], predicted).ravel()
        tn = int(tn)
        fp = int(fp)
        fn = int(fn)
        tp = int(tp)
        assign_to_dict(result, ["inference", "precision", classifier_name, \
                train_csv, candidate_str, test_csv_list[index]], tp / (tp + fp))

        assign_to_dict(result, ["inference", "recall", classifier_name, \
                train_csv, candidate_str, test_csv_list[index]], tp / (tp + fn))
        
    print ("--------------------------------------------")
    print ()

    return ("success", result)



def merge_encoding(df_list, key):
    encoder = LabelEncoder()
    column_list = list(map(lambda s: s[key], df_list))
    column = pd.concat(column_list, axis=0)
    encoder.fit(column.values)

    return \
        list(map(lambda item: encoder.transform(item.values), column_list))
    

def func_feature_combination(train_csv, test_csv, result_filename):

    format_column = [
        'IsBot', 'requestUri', 'UserAgent',\
        'SessionDuration', 'Vertical', 'ClientIP',\
        'PageName', 'UserIsNew', 'SuccessClickCount',\
        'IsRewardUser', 'DwellTime', 'PageClickCount'
    ]

    to_int_list = ['IsBot', 'UserIsNew','IsRewardUser']
    to_str_list = ["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]
    
    candidates = [
        "PageName", 
        "Vertical", 
        "ClientIP", 
        "UserIsNew", 
        "IsRewardUser", 
        "requestUri", 
        "UserAgent", 
        "SuccessClickCount", 
        "PageClickCount", 
        "SessionDuration", 
        "DwellTime"
    ]

    df_train = pd.read_csv(train_csv, sep='\t', header=0)
    df_test = pd.read_csv(test_csv, sep='\t', header=0)
    
    df_train.columns = format_column
    df_test.columns = format_column
    
    df_train[to_int_list] = \
        df_train[to_int_list].astype(int)
    df_train[to_str_list] = \
        df_train[to_str_list].astype(str)
    
    df_test[to_int_list] = \
        df_test[to_int_list].astype(int)
    df_test[to_str_list] = \
        df_test[to_str_list].astype(str)

    y_train = df_train['IsBot']
    y_test = df_test['IsBot']

    # for key in ["requestUri", \
    #                 "UserAgent", "Vertical", "PageName", "ClientIP"]:
    #     try:
    #         df_train[key], df_test[key] = \
    #             merge_encoding([df_train, df_test], key)
    #     except:
    #         import pdb; pdb.set_trace()

    result = {}
    train_length = len(df_train)
    for i in range(len(candidates)):
        for candidate in combinations(candidates, i+1):
            candidate = list(candidate)
            key = "+".join(candidate)
            df_train_current = df_train[candidate][0:int(train_length*0.02)]
            y_train_current = y_train[0:int(train_length*0.02)]

            df_test_current = df_test[candidate][0:int(train_length*1)]
            y_test_current = y_test[0:int(train_length*1)]

            dt = DecisionTreeClassifier()

            dt.fit(df_train_current, y_train_current)

            predicted = dt.predict(df_test_current)
            tn, fp, fn, tp = confusion_matrix(y_test_current, predicted).ravel()
            tn = int(tn)
            fp = int(fp)
            fn = int(fn)
            tp = int(tp)
            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                result[key] = {
                    "precision": precision,
                    "recall": recall
                }
            except:
                result[key] = {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp
                }
            print (key)
            print (tn, fp, fn, tp)
            print ()
        with open(result_filename, 'w') as fp:
            fp.write(json.dumps(result, indent=4, separators=(',', ': ')))
        print (i)
    
def func_join(train_csv, test_csv):
    df_train = pd.read_csv(train_csv, sep='\t', header=0)
    df_test = pd.read_csv(test_csv, sep='\t', header=0)
    df_train.columns = ['TimeStamp', 'requestUri', 'UserAgent', "SessionDuration", 'Vertical', 'ClientIP',
       'ClientId', 'PageName', 'UserIsNew', 'SuccessClickCount', 'IsRewardUser', 'DwellTime', 'PageClickCount',
       'QueryIsNormal', 'IsBot', 'UriCount', 'ClientIp1', 'EncodingClientIp', 'UserAgent1', 'EncodingUserAgent']
    df_test.columns = ['TimeStamp', 'requestUri', 'UserAgent', "SessionDuration", 'Vertical', 'ClientIP',
       'ClientId', 'PageName', 'UserIsNew', 'SuccessClickCount', 'IsRewardUser', 'DwellTime', 'PageClickCount',
       'QueryIsNormal', 'IsBot', 'UriCount', 'ClientIp1', 'EncodingClientIp', 'UserAgent1', 'EncodingUserAgent']
    

    df_train[['IsBot', 'UserIsNew','IsRewardUser','QueryIsNormal']] = \
        df_train[['IsBot', 'UserIsNew','IsRewardUser','QueryIsNormal']].astype(int)
    df_test[['IsBot', 'UserIsNew','IsRewardUser','QueryIsNormal']] = \
        df_test[['IsBot', 'UserIsNew','IsRewardUser','QueryIsNormal']].astype(int)


    df_train[["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]] = \
        df_train[["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]].astype(str)
    df_test[["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]] = \
        df_test[["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]].astype(str)



    for key in ["requestUri", \
                    "UserAgent", "Vertical", "PageName", "ClientIP"]:
        try:
           df_train[key] = df_train[key].apply(hash)
           df_test[key] = df_test[key].apply(hash)
        except:
            import pdb; pdb.set_trace()
    
    import pdb; pdb.set_trace()
    candidates = [
        "PageName", 
        "Vertical", 
        "ClientIP", 
        "UserIsNew", 
        "IsRewardUser", 
        "requestUri", 
        "UserAgent", 
        "SuccessClickCount", 
        "PageClickCount", 
        "SessionDuration", 
        "DwellTime"
    ]

    y_train = df_train['IsBot']
    y_test = df_test['IsBot']

    result = {}
    
    for i in range(len(candidates)):
        for candidate in combinations(candidates, i+1):
            candidate = list(candidate)
            key = "+".join(candidate)
            df_train_current = df_train[candidate]
            df_test_current = df_test[candidate]

            dt = DecisionTreeClassifier()

            dt.fit(df_train_current, y_train)

            predicted = dt.predict(df_test_current)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
            tn = int(tn)
            fp = int(fp)
            fn = int(fn)
            tp = int(tp)
            try:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                result[key] = {
                    "precision": precision,
                    "recall": recall
                }
            except:
                result[key] = {
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    "tp": tp
                }
            print (key)
            print (tn, fp, fn, tp)
            print ()
        with open("result_hash.json", 'w') as fp:
            fp.write(json.dumps(result))
        print (i)    

def sort():
    with open("result_hash.json", "r") as fp:
        dic = json.loads(fp.read())

    result = []
    for key in dic:
        try:
            result.append((dic[key]["precision"], dic[key]["recall"], key))
        except:
            pass
        
    result = sorted(result, key=lambda student : student[0], reverse=True) 
    with open("sorted_hash.json", "w") as fp:
        fp.write(json.dumps(result))

def analysis_data_distribution(train_csv, test_csv):
    df_train = pd.read_csv(train_csv, header=0)
    df_test = pd.read_csv(test_csv, header=0)
    
    
    ret = {}
    for key in df_train.columns:
        ret[key] = {"train": Counter(), "test": Counter()}
        for (index, item_train) in df_train[key].iteritems():
            ret[key]["train"][item_train] += 1
        for (index, item_test) in df_test[key].iteritems():
            ret[key]["test"][item_test] += 1
        
        ret[key]["train"] = ret[key]["train"].most_common()
        ret[key]["test"] = ret[key]["test"].most_common()
        print (key)
    
    # with open("counter.json", "w") as fp:
    #     fp.write(json.dumps(ret))
    import pdb; pdb.set_trace()

def filter_based_on_probability(train_csv, test_csv, de):
    df_train = pd.read_csv(train_csv, header=0)
    df_test = pd.read_csv(test_csv, header=0)

    filt = {
        "UserAgent": {},
        "requestUri": {}
    }
    for (index, item_train) in df_train.iterrows():
        for key in ["UserAgent", "requestUri"]:
            if filt[key].get(item_train[key]) is None:
                
                filt[key][item_train[key]] = [0, 0]
                
            filt[key][item_train[key]][int(item_train["IsBot"])] += 1 

    
    expect = list(df_test["IsBot"])
    print("HHH")
    for key in ["UserAgent", "requestUri"]:
        predict = []
        for (index, item_test) in df_test.iterrows():    
            if filt[key].get(item_test[key]) is None:
                predict.append(de)
            else:
                f, t = filt[key].get(item_test[key])
                predict.append(False if f > t else True)


        import pdb; pdb.set_trace()
        tn, fp, fn, tp = confusion_matrix(expect, predict).ravel()


def func_hash(train_csv, test_csv_list):
    
    format_column = [
        'IsBot', 'requestUri', 'UserAgent',\
        'SessionDuration', 'Vertical', 'ClientIP',\
        'PageName', 'UserIsNew', 'SuccessClickCount',\
        'IsRewardUser', 'DwellTime', 'PageClickCount', 'UriCount'
    ]

    to_int_list = ['IsBot', 'UserIsNew','IsRewardUser']
    to_str_list = ["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]
    
    candidates = [
        "PageName", 
        "Vertical", 
        "ClientIP", 
        "UserIsNew", 
        "IsRewardUser", 
        "requestUri", 
        "UserAgent", 
        "SuccessClickCount", 
        "PageClickCount", 
        "SessionDuration", 
        "DwellTime"
    ]

    start = time.time()
    df_train = pd.read_csv(train_csv, sep='\t', header=0, memory_map=True)
    print ("---------------------Training data loaded----------------------------")
    print ("Spend {0}".format(str(time.time() - start)))

    df_train.columns = format_column
    df_train[to_int_list] = \
        df_train[to_int_list].astype(int)
    df_train[to_str_list] = \
        df_train[to_str_list].astype(str)
    
    y_train = df_train['IsBot']
    df_train = df_train[candidates]

    ## dt = DecisionTreeClassifier()
    ## dt.fit(df_train, y_train)
    import pdb; pdb.set_trace()
    
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
    xgb_model.fit(df_train, y_train)
    
    # y_pred = xgb_model.predict()
    
    print ("---------------------DT Training finished--------------------------")
    print ("Spend {0}".format(str(time.time() - start)))


    class ThreadWithReturnValue(Thread):
        def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
            Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

            self._return = None

        def run(self):
            if self._target is not None:
                self._return = self._target(*self._args, **self._kwargs)

        def join(self):
            Thread.join(self)
            return self._return

    def read_test_file(file_name):
        return pd.read_csv(test_csv, sep='\t', header=0)

    thread_pool = []
    for test_csv in test_csv_list:
        th = ThreadWithReturnValue(target=read_test_file, args=(test_csv,))
        th.start()
        thread_pool.append(th)
        print ("Load Test file asynchrously")


    df_test_list = []
    for thread in thread_pool:
        df_test_list.append(thread.join())

    print ("---------------------Data loaded----------------------------")
    
    
    # for test_csv in test_csv_list:
    #     df_test_list.append(pd.read_csv(test_csv, sep='\t', header=0))
    #     print ("---------------------Testing data loaded----------------------------")


    for df_test in df_test_list:
        df_test.columns = format_column
        df_test[to_int_list] = \
            df_test[to_int_list].astype(int)
        df_test[to_str_list] = \
            df_test[to_str_list].astype(str)
        y_test = df_test['IsBot']
        df_test = df_test[candidates]

        predicted = dt.predict(df_test)
        tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
        tn = int(tn)
        fp = int(fp)
        fn = int(fn)
        tp = int(tp)
        ## precision = tp / (tp + fp)
        ## recall = tp / (tp + fn)
        import pdb; pdb.set_trace()



def func_partial(train_csv, partial_list, test_csv_list, parrallel, filename):
    format_column = [
        'IsBot', 'requestUri', 'UserAgent',\
        'SessionDuration', 'Vertical', 'ClientIP',\
        'PageName', 'UserIsNew', 'SuccessClickCount',\
        'IsRewardUser', 'DwellTime', 'PageClickCount', 'UriCount'
    ]

    to_int_list = ['IsBot', 'UserIsNew','IsRewardUser']
    to_str_list = ["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]
    
    candidates = [
        "PageName", 
        "Vertical", 
        "ClientIP", 
        "UserIsNew", 
        "IsRewardUser", 
        "requestUri", 
        "UserAgent", 
        "SuccessClickCount", 
        "PageClickCount", 
        "SessionDuration", 
        "DwellTime"
    ]


    df_test_list = []
    df_train = None
    y_train = None
    start_time = time.time()
    if parrallel:
        class ThreadWithReturnValue(Thread):
            def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
                Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

                self._return = None

            def run(self):
                if self._target is not None:
                    self._return = self._target(*self._args, **self._kwargs)

            def join(self):
                Thread.join(self)
                return self._return

        def read_csv_file(file_name):
            return pd.read_csv(file_name, sep='\t', header=0, memory_map=True)
        
        thread_pool = []

        th = ThreadWithReturnValue(target=read_csv_file, args=(train_csv,))
        th.start()
        thread_pool.append(th)
        print ("Load Train file asynchrously")

        for test_csv in test_csv_list:
            th = ThreadWithReturnValue(target=read_csv_file, args=(test_csv,))
            th.start()
            thread_pool.append(th)
            print ("Load Test file asynchrously")

        
        for (index, thread) in enumerate(thread_pool):
            if index == 0:
                df_train = thread.join()
                df_train.columns = format_column
                df_train[to_int_list] = \
                    df_train[to_int_list].astype(int)
                df_train[to_str_list] = \
                    df_train[to_str_list].astype(str)
                
                y_train = df_train['IsBot']
                df_train = df_train[candidates]
            else:
                df_test = thread.join()
                df_test.columns = format_column
                df_test[to_int_list] = \
                    df_test[to_int_list].astype(int)
                df_test[to_str_list] = \
                    df_test[to_str_list].astype(str)
                y_test = df_test['IsBot']
                df_test = df_test[candidates]
                df_test_list.append((df_test, y_test))

    else:
        df_train = pd.read_csv(train_csv, sep='\t', header=0)
        df_train.columns = format_column
        df_train[to_int_list] = \
            df_train[to_int_list].astype(int)
        df_train[to_str_list] = \
            df_train[to_str_list].astype(str)
        
        y_train = df_train['IsBot']
        df_train = df_train[candidates]
        for (index, test_csv) in enumerate(test_csv_list):
            start = time.time()
            df_test = pd.read_csv(test_csv, sep='\t', header=0)
            df_test.columns = format_column
            df_test[to_int_list] = \
                df_test[to_int_list].astype(int)
            df_test[to_str_list] = \
                df_test[to_str_list].astype(str)
            y_test = df_test['IsBot']
            df_test = df_test[candidates]
            df_test_list.append((df_test, y_test))

            end = time.time()
            print ("Spend {0}s to load {1} testing dataset".format(str(end-start), str(index)))

    print ("---------------------All data prepared----------------------------")
    print ("Spend {0} minutes to load all data".format(str(time.time() - start_time)))
    import pdb; pdb.set_trace()


    result = {
        "training": {
            "time": {

            }
        },
        "inference": {
            "time": {

            }, 
            "precision": {
            
            },
            "recall": {
                
            }
        }
    }
    classifier_list = ["dt", "xgboost"]

    train_length = len(df_train)
    for partial in partial_list:
        df_train_current = df_train[0:int(train_length*partial)]
        y_train_current = y_train[0:int(train_length*partial)]
        
        
        for classifier_name in classifier_list:
            classifier = get_classifier(classifier_name) 

            start = time.time()
            classifier.fit(df_train_current.values, y_train_current)
            end = time.time()
            if result["training"]["time"].get(classifier_name) is None: 
                result["training"]["time"][classifier_name] = {}
            result["training"]["time"][classifier_name][partial] = end - start
            print ("{0} Training with {1} of data is {2}s".format(classifier_name, str(partial), str(end-start)) )
            with open("./model/"+classifier_name+"-"+str(partial)+".pkl", "wb") as fp:
                pickle.dump(classifier, fp)

        
        
        for (index, (df_test, y_test)) in enumerate(df_test_list):
            
            for classifier_name in classifier_list:
                with open("./model/"+classifier_name+"-"+str(partial)+".pkl", "rb") as fp:
                    classifier = pickle.load(fp)
            
                start = time.time()
                predicted = classifier.predict(df_test.values)
                end = time.time()
                if result["inference"]["time"].get(classifier_name) is None:
                    result["inference"]["time"][classifier_name] = {}
                result["inference"]["time"][classifier_name][partial] = end - start
                print (" {0} Inference with {1} of data is {2}s".format(classifier_name, str(partial), str(end-start)) )

                tn, fp, fn, tp = confusion_matrix(y_test, predicted).ravel()
                tn = int(tn)
                fp = int(fp)
                fn = int(fn)
                tp = int(tp)

                if result["inference"]["precision"].get(classifier_name) is None:
                    result["inference"]["precision"][classifier_name] = {}
                if result["inference"]["precision"][classifier_name].get(partial) is None:
                    result["inference"]["precision"][classifier_name][partial] = {}
                result["inference"]["precision"][classifier_name][partial][index] = tp / (tp + fp)

                if result["inference"]["recall"].get(classifier_name) is None:
                    result["inference"]["recall"][classifier_name] = {}
                if result["inference"]["recall"][classifier_name].get(partial) is None:
                    result["inference"]["recall"][classifier_name][partial] = {}
                result["inference"]["recall"][classifier_name][partial][index] = tp / (tp + fn)

                with open(filename, "w") as fp:
                    fp.write(json.dumps(result, indent=4, separators=(',', ': ')))
            
        print ("--------------------------------------------")
        print()


format_column = [
    'IsBot', 'CountryIso', 'State', 'City', 'Region', \
    'PostalCode', 'Lat', 'Long', 'TimeZone', 'ClientIp',\
    'UserAgent', 'RequestUrl', 'StatusCode', 'Request_Bytes',\
    'Request_DataCenter', 'Request_Browser', 'FdPartnerName',\
    'Ports', 'User_AcceptLanguage', 'Request_Referrer',\
    'Market', 'HasCookie', 'HasMsIp', 'AppNS', 'Page_UiLanguage'
]

to_int_list = ['HasCookie', 'HasMsIp']

to_int32_list = ['CountryIso', 'State', 'City', 'Region', \
    'PostalCode', 'Lat', 'Long', 'TimeZone', 'ClientIp',\
    'UserAgent', 'RequestUrl', 'Request_Bytes',\
    'Request_DataCenter', 'Request_Browser', 'FdPartnerName',\
    'Ports', 'User_AcceptLanguage', 'Request_Referrer',\
    'Market', 'AppNS', 'Page_UiLanguage']

candidates = [
    'CountryIso', 'State', 'City', 'Region',\
    'PostalCode', 'Lat', 'Long', 'TimeZone', 'ClientIp',\
    'UserAgent', 'RequestUrl', 'StatusCode', 'Request_Bytes',\
    'Request_DataCenter', 'Request_Browser', 'FdPartnerName',\
    'Ports', 'User_AcceptLanguage', 'Request_Referrer',\
    'Market', 'HasCookie', 'HasMsIp', 'AppNS', 'Page_UiLanguage'
]

var_candidates = [
    'CountryIso', 'State', 'City', \
    'PostalCode', 'Lat', 'Long', 'TimeZone', 'ClientIp',\
    'UserAgent', 'RequestUrl', 'StatusCode', 'Request_Bytes',\
    'Request_DataCenter', 'Request_Browser', 'FdPartnerName',\
    'User_AcceptLanguage', 'Request_Referrer',\
    'Market', 'AppNS', 'Page_UiLanguage'
]

# geo_candidates = [
#     'CountryIso', 'State', 'City', 'Region',
#     'PostalCode', 'Lat', 'Long', 'TimeZone'
# ]



candidates_list = {
    "candidates": [\
        'CountryIso', 'State', 'City', 'Region',\
        'PostalCode', 'Lat', 'Long', 'TimeZone', 'ClientIp',\
        'UserAgent', 'RequestUrl', 'StatusCode', 'Request_Bytes',\
        'Request_DataCenter', 'Request_Browser', 'FdPartnerName',\
        'Ports', 'User_AcceptLanguage', 'Request_Referrer',\
        'Market', 'HasCookie', 'HasMsIp', 'AppNS', 'Page_UiLanguage'
    ],

    "var_candidates": [
        'CountryIso', 'State', 'City', \
        'PostalCode', 'Lat', 'Long', 'TimeZone', 'ClientIp',\
        'UserAgent', 'RequestUrl', 'StatusCode', 'Request_Bytes',\
        'Request_DataCenter', 'Request_Browser', 'FdPartnerName',\
        'User_AcceptLanguage', 'Request_Referrer',\
        'Market', 'AppNS', 'Page_UiLanguage'
    ],

    # "geo_candidates": [\
    #     'CountryIso', 'State', 'City', 'Region',\
    #     'PostalCode', 'Lat', 'Long', 'TimeZone'
    # ]
}

classifier_list = ["dt", "xgboost", "rf5", "rf3", "kn"]


class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return

def read_csv_file(file_name):
    return pd.read_csv(file_name, sep='\t', header=0, memory_map=True, low_memory=False)

def shrink(x):
    return x & ((1<<30)-1)

def train_some(args):
    print("---------------------------------------------------")
    classifier_name = args["classifier_name"]
    can_name = args["can_name"]
    train_csv = args["train_csv"]
    index = args["index"]
    df_train = args["namespace"].df[index]

    print (classifier_name+can_name+train_csv+"Begin............")

    classifier = get_classifier(classifier_name) 
    can = candidates_list[can_name]
    df_train_current = df_train[can]
    start = time.time()
    classifier.fit(df_train_current.values, df_train["IsBot"])
    end = time.time()

    print ("{0}{1}{2} Training of data \
        consume {3}s......".format(classifier_name, can_name, train_csv, str(end-start)) )
    train_name = train_csv.split("/")[-1].split(".")[0]
    with open("./model_new/"+classifier_name+"_"+train_name+"_"+can_name+"cross-check.pkl", "wb") as fp:
        pickle.dump(classifier, fp)

    return [(classifier_name, train_csv, can_name), end - start]

def only_train(train_csv_list, candidates_list, classifier_list, outputfile):
    df_train_list = []
    df_train = None
        
    train_thread_pool = []
    for train_csv in train_csv_list:
        th = ThreadWithReturnValue(target=read_csv_file, args=(train_csv,))
        th.start()
        train_thread_pool.append(th)
        print ("Load Train file asynchrously")
    


    ## Sync the I/O threads
    for (index, thread) in enumerate(train_thread_pool):
        df_train = thread.join()
        df_train.columns = format_column
        df_train_list.append(df_train)

    for (index, df_train) in enumerate(df_train_list):
        df_train_list[index].dropna(how='any', inplace=True)
        df_train_list[index][to_int_list] = \
                        df_train[to_int_list].astype(int)
        df_train_list[index][to_int32_list] = \
                        df_train[to_int32_list].astype(np.int64).apply(shrink)
        print("finish {0} training".format(str(index)))

    
    for (filename, df_train) in zip(train_csv_list, df_train_list):
        train_name = filename.split("/")[-1].split(".")[0]
        counter = {}
        bot_count = sum(df_train["IsBot"])
        assign_to_dict(counter, [filename, "total"], len(df_train) )
        assign_to_dict(counter, [filename, "bot"], bot_count)
        assign_to_dict(counter, [filename, "nonbot"], len(df_train)-bot_count) 

        with open("./result_new/cross_check_counter_" + train_name + ".json", "w") as fp:
            fp.write(json.dumps(counter, indent=4, separators=(',', ': ')))

    ret = {
        
    }
    
    # mgr = Manager()
    # ns = mgr.Namespace()
    # ns.df = df_train_list

    ## Syncronous Code

    for (df_train, train_csv) in zip(df_train_list, train_csv_list):
        for can_name in candidates_list:    
            for classifier_name in classifier_list:
                classifier = get_classifier(classifier_name) 
                can = candidates_list[can_name]
                df_train_current = df_train[can]
                print ("Begin {0} Training of data {1} {2}".format(classifier_name, can_name, train_csv) )
                start = time.time()
                classifier.fit(df_train_current.values, df_train["IsBot"])
                end = time.time()
                
                assign_to_dict(ret, ["training", "time", \
                    classifier_name, train_csv, can_name], end - start)

                print ("{0} Training of data is {1}s".format(classifier_name, str(end-start)) )
                train_name = train_csv.split("/")[-1].split(".")[0]
                with open("./model_new/"+classifier_name+"_"+train_name+"_"+can_name+"cross-check.pkl", "wb") as fp:
                    pickle.dump(classifier, fp)


    # args = []
    # for (index, train_csv) in enumerate(train_csv_list):
    #     for can_name in candidates_list:    
    #         for classifier_name in classifier_list:
    #             args.append({
    #                 "classifier_name": classifier_name,
    #                 "can_name": can_name,
    #                 "train_csv": train_csv,
    #                 "namespace": ns,
    #                 "index":index
    #            })

    # pool = mp.Pool(processes=25)    
    # results = pool.map(train_some, args)
    # for result in results:
    #     classifier_name = result[0][0]
    #     train_csv = result[0][1]
    #     can_name = result[0][2]
    #     time = result[1]
    #     assign_to_dict(ret, ["training", \
    #                 classifier_name, can_name, train_csv, "time"], time)

    train_name = train_csv_list[0].split("/")[-1].split(".")[0]
    with open("./result_new/crosscheck_"+classifier_name+"_"+train_name+"_"+can_name+".json", "w") as fp:
        fp.write(json.dumps(ret, indent=4, separators=(',', ': ')))     

def inference_some(args):
    classifier_name = args["classifier_name"]
    can_name = args["can_name"]
    train_csv = args["train_csv"]
    index = args["index"]
    df_test_list = args["namespace"].df
    df_test = df_test_list[index]
    can = candidates_list[can_name]

    df_test_current = df_test[can]
    train_name = train_csv.split("/")[-1].split(".")[0]
    with open("./model_new/"+classifier_name+"_"+train_name+"_"+can_name+"cross-check.pkl", "rb") as fp:
        classifier = pickle.load(fp)
    start = time.time()
    predicted = classifier.predict(df_test_current.values)
    end = time.time()
    print (" {0} Inference of data {1} is {2}s".format(classifier_name, index, str(end-start)) )

    tn, fp, fn, tp = confusion_matrix(df_test["IsBot"], predicted).ravel()
    tn = int(tn)
    fp = int(fp)
    fn = int(fn)
    tp = int(tp)

    return [classifier_name, train_csv, can_name, (tp / (tp + fp), tp / (tp + fn)), end-start, index]

def only_inference(train_csv_list, test_csv_list, candidates_list, classifier_list, outputfile):
    test_thread_pool = []
    df_test_list =[]
    for test_csv in test_csv_list:
        th = ThreadWithReturnValue(target=read_csv_file, args=(test_csv,))
        th.start()
        test_thread_pool.append(th)
        print ("Load Test file asynchrously")

    for (index, thread) in enumerate(test_thread_pool):    
        df_test = thread.join()
        df_test.columns = format_column
        df_test_list.append(df_test)

    for (index, df_test) in enumerate(df_test_list):
        df_test_list[index].dropna(how='any', inplace=True)
        df_test_list[index][to_int_list] = \
                    df_test[to_int_list].astype(int)
        df_test_list[index][to_int32_list] = \
                        df_test[to_int32_list].astype(np.int64).apply(shrink)
        print("finish {0} test".format(str(index)))

    counter = {}
    for (filename, df_test) in zip(test_csv_list, df_test_list):
        bot_count = sum(df_test["IsBot"])
        assign_to_dict(counter, [filename, "total"], len(df_test) )
        assign_to_dict(counter, [filename, "bot"], bot_count)
        assign_to_dict(counter, [filename, "nonbot"], len(df_test)-bot_count)

    with open("./result/cross_check_counter_test.json", "w") as fp:
        fp.write(json.dumps(counter, indent=4, separators=(',', ': ')))

    ret = {

    }

    mgr = Manager()
    ns = mgr.Namespace()
    ns.df = df_test_list
    args = []
    for train_csv in train_csv_list:
        for can_name in candidates_list:    
            for classifier_name in classifier_list:
                for (index, df_test) in enumerate(df_test_list):
                    args.append({
                        "classifier_name": classifier_name,
                        "can_name": can_name,
                        "train_csv": train_csv,
                        "index": index,
                        "namespace": ns
                    })
                    
                    

    pool = mp.Pool(processes=20)    
    results = pool.map(inference_some, args)
    
    for result in results:
        classifier_name = result[0]
        train_csv = result[1]
        can_name = result[2]
        precision = result[3][0]
        recall = result[3][1]
        time = result[4]
        index = result[5]
        assign_to_dict(ret, ["inference", classifier_name, \
                            train_csv, can_name, test_csv_list[index], "time"], time)
        assign_to_dict(ret, ["inference", classifier_name, \
                            train_csv, can_name, test_csv_list[index], "precision"], precision)
        assign_to_dict(ret, ["inference", classifier_name, \
                            train_csv, can_name, test_csv_list[index], "recall"], recall)


    with open(outputfile, "w") as fp:
        fp.write(json.dumps(ret, indent=4, separators=(',', ': ')))

def func_cross_check_v2(train_csv_list, candidates_list, classifier_list, test_csv_list):
    only_train(train_csv_list, candidates_list, \
            classifier_list, "./result/cross_check_train.json")
    only_inference(train_csv_list, test_csv_list, \
        candidates_list, classifier_list, "./result/cross_check_test.json")



def func_cross_check(train_csv_list, test_csv_list, parrallel, file):

    # tree_candidates = [
    #     'CountryIso', 'AppNS'
    # ]

    df_test_list = []
    df_train_list = []
    df_train = None
    y_train = None
    start_time = time.time()
    if parrallel:
        train_thread_pool = []
        test_thread_pool = []
        for train_csv in train_csv_list:
            th = ThreadWithReturnValue(target=read_csv_file, args=(train_csv,))
            th.start()
            train_thread_pool.append(th)
            print ("Load Train file asynchrously")

        for test_csv in test_csv_list:
            th = ThreadWithReturnValue(target=read_csv_file, args=(test_csv,))
            th.start()
            test_thread_pool.append(th)
            print ("Load Test file asynchrously")

        
        for (index, thread) in enumerate(train_thread_pool):
            df_train = thread.join()
            df_train.columns = format_column
            df_train_list.append(df_train)
            
        for (index, thread) in enumerate(test_thread_pool):    
            df_test = thread.join()
            df_test.columns = format_column
            df_test_list.append(df_test)

    else:
        for (index, train_csv) in enumerate(train_csv_list):
            start = time.time()
            df_train = pd.read_csv(train_csv, sep='\t', header=0)
            df_train.columns = format_column
            df_train_list.append(df_train)

        for (index, test_csv) in enumerate(test_csv_list):
            start = time.time()
            df_test = pd.read_csv(test_csv, sep='\t', header=0)
            df_test.columns = format_column
            df_test_list.append(df_test)

    print ("---------------------All data prepared----------------------------")
    print ("Spend {0} minutes to load all data".format(str(time.time() - start_time)))
    
    for (index, df_train) in enumerate(df_train_list):
        df_train_list[index].dropna(how='any', inplace=True)

        df_train_list[index][to_int_list] = \
                        df_train[to_int_list].astype(int)
        df_train_list[index][to_int32_list] = \
                        df_train[to_int32_list].astype(np.int64).apply(shrink)
        print("finish {0} training".format(str(index)))


    for (index, df_test) in enumerate(df_test_list):
        df_train_list[index].dropna(how='any', inplace=True)
        df_test_list[index][to_int_list] = \
                    df_test[to_int_list].astype(int)
        df_test_list[index][to_int32_list] = \
                    df_test[to_int32_list].astype(np.int64).apply(shrink)
        print("finish {0} test".format(str(index)))
    import pdb; pdb.set_trace()
    
    
    ## import pdb; pdb.set_trace()
    counter = {}
    for (filename, df_train) in zip(train_csv_list, df_train_list):
        bot_count = sum(df_train["IsBot"])
        assign_to_dict(counter, [filename, "total"], len(df_train) )
        assign_to_dict(counter, [filename, "bot"], bot_count)
        assign_to_dict(counter, [filename, "nonbot"], len(df_train)-bot_count) 
        

    for (filename, df_test) in zip(test_csv_list, df_test_list):
        bot_count = sum(df_test["IsBot"])
        assign_to_dict(counter, [filename, "total"], len(df_test) )
        assign_to_dict(counter, [filename, "bot"], bot_count)
        assign_to_dict(counter, [filename, "nonbot"], len(df_test)-bot_count) 

    with open("./result/cross_check_counter.json", "w") as fp:
        fp.write(json.dumps(counter))


    print("finish counter")
    # sel = VarianceThreshold(threshold=(.1 * (1 - .1)))
    # var_train = sel.fit_transform(df_train)


    # # lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(df_train, y_train)    
    # # model = SelectFromModel(lsvc, prefit=True)
    # # svc_train = model.transform(df_train)


    # clf = ExtraTreesClassifier(n_estimators=50)
    # clf = clf.fit(df_train, y_train)
    # model = SelectFromModel(clf, prefit=True)
    # tree_train = model.transform(df_train)

    for i in range(3, 24):
        X_new = SelectKBest(chi2, k=i).fit_transform(df_train_list[0][candidates].values, df_train_list[0]["IsBot"])
        import pdb; pdb.set_trace()
    # import pdb; pdb.set_trace()
    
    results = []
    for (df_train, train_csv) in zip(df_train_list, train_csv_list):
        for can in [candidates, var_candidates]:    
            for classifier_name in ["dt", "xgboost", "rf5", "rf3", "kn"]:
                # th = ThreadWithReturnValue(target=train_and_inference, args=(classifier_name, can,))
                # th.start()
                # train_inference_pool.append(th)
                # args.append({
                #     "classifier_name": classifier_name,
                #     "can": can,
                #     "df_test_list": df_test_list,
                #     "test_csv_list": test_csv_list,
                #     "df_train": df_train,
                #     "y_train": y_train
                # })
                arg = {
                    "classifier_name": classifier_name,
                    "can": can,
                    "df_test_list": df_test_list,
                    "test_csv_list": test_csv_list,
                    "df_train": df_train,
                    "train_csv": train_csv
                }
                results.append(train_and_inference(arg))


    def merge_right_dict(original_dict, new_dict):
        if not isinstance(new_dict, dict):
            return new_dict

        for k in new_dict:
            if original_dict.get(k) is None:
                original_dict[k] = {}
            original_dict[k] = merge_right_dict(original_dict[k], new_dict[k])
                
        return original_dict


    all_result = {}
    # pool = mp.Pool(processes=20)    
    # results = pool.map(train_and_inference, args)
    # status, result = item.join()
    for result in results:
        all_result = merge_right_dict(all_result, result[1])
    with open(file, "w") as fp:
        fp.write(json.dumps(all_result, indent=4, separators=(',', ': ')))





        

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn')
    # func1("./data/3_8_8_half_min.tsv", "./data/3_20_8_half_min.tsv")
    # func_hash("./data/3_8_8_half_min.tsv", "./data/3_20_8_half_min.tsv")
    # sort()
    # analysis("./data/logs_0201_refined", "./data/logs_0202_refined")
    #filter("./data/logs_0201_refined", "./data/logs_0202_refined", False)
    

    ## func_hash("./data/3_6_20_1h.tsv", ["./data/3_9_20_1h.tsv", "./data/3_13_20_1h.tsv"])
    # func_partial("./data/3_6_20_1h.tsv", [1], \
    #     ["./data/3_9_20_1h.tsv", "./data/3_13_20_1h.tsv"], True, "./result/1hour_result.json")

    # func_cross_check_v2(["./data/3_6_20_geo_1s.tsv", "./data/3_6_4_geo_1s.tsv"], \
    #     candidates_list, classifier_list, ["./data/3_6_20_geo_1s.tsv", "./data/3_9_20_geo_1s.tsv"])
    


    # func_cross_check_v2(["./data/3_6_20_geo_1h.tsv"], \
    #     candidates_list, classifier_list, \
    #         ["./data/3_9_20_geo_1h.tsv", "./data/3_13_20_geo_1h.tsv", "./data/3_9_4_geo_1h.tsv", "./data/3_13_4_geo_1h.tsv"])


    # func_cross_check(["./data/3_6_20_geo_1h.tsv"], \
    #     [], True, "result/cross_check.json")

    # func_cross_check(["./data/3_6_20_geo_1h.tsv"], \
    #     ["./data/3_9_20_geo_1h.tsv", "./data/3_13_20_geo_1h.tsv", \
    #         "./data/3_9_4_geo_1h.tsv", "./data/3_13_4_geo_1h.tsv"], True, "result/cross_check.json")


    # func_feature_combination("./data/3_6_20_20min.tsv", \
    #     "./data/3_9_20_20min.tsv", "./result/20min_combination.json")


    ## classifier_list = ["dt", "xgboost", "rf5", "rf3", "rf7", "kn"]






    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--clas', '-c', required = True, type = str)
    parser.add_argument('--data', '-d', required = True, type = int) # 20 or 4
    ar = parser.parse_args()
    train_file = "./data/3_6_" + str(ar.data) + "_geo_1h.tsv"
    only_train([train_file], candidates_list, \
            [ar.clas], "./result_new/cross_check_train_" + ar.clas + "_" + train_file + ".json")