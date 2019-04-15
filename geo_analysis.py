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
import math

from ua_parser import user_agent_parser
import dask.dataframe as dd

from pandas_multiprocess import multi_process

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

def shrink_hash(x):
    return hash(x) & ((1<<30)-1)


def ua_parsing(ua_csv):
    format_column = [
        'IsBot', 'CountryIso', 'State',\
        'City', '_Region', 'PostalCode',\
        'Lat', 'Long', 'TimeZone',\
        'User_Ip', 'Request_UserAgent', 'Request_Url', \
        'StatusCode', 'Request_Bytes', 'Request_DataCenter',\
        'Request_Browser', 'Request_FdPartnerName', 'User_AcceptLanguage',\
        'Request_Referrer', 'Market', 'Admin_User_HasCookies', 'User_HasMsIp', \
        'Request_Domain', 'Request_ResponseBytes', 'Request_PartnerApplicationId',\
        'ClientId', 'AppNS', 'Page_UiLanguage', 'Request_FirstEventTimestamp'
    ]

    # to_int_list = ['IsBot', 'UserIsNew','IsRewardUser']
    # to_str_list = ["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]

    
    
    df_ua = pd.read_csv(ua_csv, sep='\t', header=0)
    df_ua.columns = format_column

    ua_dic = {}
    columns=["device_brand", "device_family", "device_model", \
                "os_family", "os_major", "os_minor", "os_patch", \
                "user_agent_family", "user_agent_major", "user_agent_minor", "user_agent_patch"]
    
    for column in columns:
        ua_dic[column] = []
    
    ua_dic["IsBot"] = []


    def apply_ua(row):
        try:
            ua = user_agent_parser.Parse(row["Request_UserAgent"])
        except:
            return
        
        for outer_key in ua:
            for inner_key in ua[outer_key]:
                key = outer_key+"_"+inner_key
                if ua_dic.get(key) is None:
                    continue
                ua_dic[key].append(shrink_hash(ua[outer_key][inner_key]))
        ua_dic["IsBot"].append(row["IsBot"])

    print (len(df_ua))
    df_ua.apply(apply_ua, axis=1)

    df_ua = pd.DataFrame(ua_dic)
    df_ua.to_csv("data/ua_" + ua_csv, index=False)


def main(geo_csv):
    format_column = [
        'IsBot', 'CountryIso', 'State',\
        'City', '_Region', 'PostalCode',\
        'Lat', 'Long', 'TimeZone',\
        'User_Ip', 'Request_UserAgent', 'Request_Url', \
        'StatusCode', 'Request_Bytes', 'Request_DataCenter',\
        'Request_Browser', 'Request_FdPartnerName', 'User_AcceptLanguage',\
        'Request_Referrer', 'Market', 'Admin_User_HasCookies', 'User_HasMsIp', \
        'Request_Domain', 'Request_ResponseBytes', 'Request_PartnerApplicationId',\
        'ClientId', 'AppNS', 'Page_UiLanguage', 'Request_FirstEventTimestamp'
    ]

    # to_int_list = ['IsBot', 'UserIsNew','IsRewardUser']
    # to_str_list = ["requestUri", "UserAgent", "Vertical", "PageName", "ClientIP"]

    

    # df_geo = pd.read_csv(geo_csv, sep='\t', header=0)
    df_geo = pd.read_csv(geo_csv, sep='\t', header=0)
    df_geo.columns = format_column



    loc_dic = {}
    geo_dic = {}
    df_ua = pd.DataFrame(columns=["os", "device", "ua"])


    for lat in range(-90, 90):
        geo_dic[lat] = {}
        for lon in range(-180, 180):
            geo_dic[lat][lon] = {
                "bot": 0,
                "nonbot": 0
            }

    def apply_geo(row):
        label = "bot" if row["IsBot"] else "nonbot"
        if pd.isna(row["Lat"]) or pd.isna(row["Long"]):
            return 
        
        lat = math.floor(int(row["Lat"]))
        lon = math.floor(int(row["Long"]))
        geo_dic[lat][lon][label] += 1
        

        if pd.isna(row["CountryIso"]):
            if loc_dic.get("unknown_nation") is None:
                loc_dic["unknown_nation"] = {
                    "bot": 0,
                    "nonbot": 0
                }
            loc_dic["unknown_nation"][label] += 1
            return 

        if loc_dic.get(row["CountryIso"]) is None:
            loc_dic[row["CountryIso"]] = {}

        
        if pd.isna(row["State"]):
            if loc_dic[row["CountryIso"]].get("unknown_state") is None:
                loc_dic[row["CountryIso"]]["unknown_state"] = {
                    "bot": 0,
                    "nonbot": 0
                }
            loc_dic[row["CountryIso"]]["unknown_state"][label] += 1
            return

        if loc_dic[row["CountryIso"]].get(row["State"]) is None:
            loc_dic[row["CountryIso"]][row["State"]] = {}



        if pd.isna(row["City"]):
            if loc_dic[row["CountryIso"]][row["State"]].get("unknown_city") is None:
                loc_dic[row["CountryIso"]][row["State"]]["unknown_city"] = {
                    "bot": 0,
                    "nonbot": 0
                }
            loc_dic[row["CountryIso"]][row["State"]]["unknown_city"][label] += 1
            return

        if loc_dic[row["CountryIso"]][row["State"]].get(row["City"]) is None:
            loc_dic[row["CountryIso"]][row["State"]][row["City"]] = {
                "bot": 0,
                "nonbot": 0   
            }
        
        loc_dic[row["CountryIso"]][row["State"]][row["City"]][label] += 1


    print (len(df_geo))
    import pdb; pdb.set_trace()
    df_geo.apply(apply_geo, axis=1)


    import pdb; pdb.set_trace()

    
        

    with open("result/loc_dict2.json", 'w') as fp:
        fp.write(json.dumps(loc_dic))

    with open("result/geo_dict2.json", 'w') as fp:
        fp.write(json.dumps(geo_dic))

def dump_3d_map_csv(filepath):
    dic = {
        "name": [],
        "latitude": [],
        "longitude": [],
        "count": []
    }
    
    with open("result/geo_dict.json", 'r') as fp:
        ll = json.loads(fp.read())
    
    for lat in ll:
        for lon in ll[lat]:
            if ll[lat][lon]["bot"] == 0:
                continue
            dic["count"].append(ll[lat][lon]["bot"])
            dic["latitude"].append(lat)
            dic["longitude"].append(lon)
            dic["name"].append("bot")
    
    df = pd.DataFrame(dic)
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    # ua_parsing("./data/3_9_20_geo_no_hash_1h.tsv")

    dump_3d_map_csv("./bot.csv")


