#!/usr/bin/env python3

import math
import sys
import time
from datetime import date

import numpy as np
from sklearn.externals import joblib

import geohash
import argparse

g = 9 #geohash length, a 1.2km x 609.4m square area
b = 12 # number of time bins per day

def main(argv):
    

    parser = argparse.ArgumentParser()
    parser.add_argument("model",  action="store", choices=["forest", "knn"], help = "Predictive model to use, can be either forest or knn")
    parser.add_argument("time", action="store", help = 'Start trip time, with the format "yyyy-MM-dd HH:mm:ss". It must be between quotation marks. For instance, you coud use: "2017-05-24 12:26:37"')
    parser.add_argument("latitude",action="store", type=float, help = "Latitude of the trip start position. For instance, you could use: 47.409291")
    parser.add_argument("longitude",action="store", type=float, help = "Longitude of the trip end position. For instance, you could use: 8.546942")
    args = parser.parse_args()
    model = args.model
    startTime = args.time
    startLat = args.latitude
    startLon = args.longitude
    model_file = None

    if (model == "forest"):
        print("Model used: Random Forest")
        model_file = 'random_forest_model.pkl'
    elif (model == "knn"):
        print("Model used: K-Nearest Neighbor")
        model_file = 'k_nearest_model.pkl'

    # Note: b must evenly divide 60
    minutes_per_bin = int((24 / float(b)) * 60)

    regressor = joblib.load(model_file)
    #regressor.set_params(verbose=False)

    # Regressor needs the follwing parameters: time_num, time_sin, time_cos, day_num, start_lat, start_lon

    startLat = float(startLat)
    startLon = float(startLon)
    x_start = math.cos(startLat) * math.cos(startLon)
    y_start = math.cos(startLat) * math.sin(startLon) 
    z_start = math.sin(startLat) 

    zippedFeatures = date_extractor(startTime, b, minutes_per_bin)
    parameters = np.array((zippedFeatures[8], x_start, y_start, z_start)).reshape(1, -1)
    prediction = regressor.predict(parameters)
    print("\nPrediction: {0}, {1}".format(prediction[0][0], prediction[0][1]))

def bucketed_location(lat, lon):
    location = geohash.encode(float(lat), float(lon), g)
    return geohash.decode(location)

def date_extractor(date_str,b,minutes_per_bin):
    # Takes a datetime object as a parameter
    # and extracts and returns a tuple of the form: (as per the data specification)
    # (time_cat, time_num, time_cos, time_sin, day_cat, day_num, day_cos, day_sin, weekend)
    # Split date string into list of date, time
    
    d = date_str.split()
    
    #safety check
    if len(d) != 2:
        return tuple([None,])
    
    # TIME (eg. for 16:56:20 and 15 mins per bin)
    #list of hour,min,sec (e.g. [16,56,20])
    time_list = [int(t) for t in d[1].split(':')]
    
    #safety check
    if len(time_list) != 3:
        return tuple([None,])
    
    # calculate number of minute into the day (eg. 1016)
    num_minutes = time_list[0] * 60 + time_list[1]
    
    # Time of the start of the bin
    time_bin = num_minutes / minutes_per_bin     # eg. 1005
    hour_bin = num_minutes / 60                  # eg. 16
    min_bin = (time_bin * minutes_per_bin) % 60  # eg. 45
    
    #get time_cat
    hour_str = str(hour_bin) if hour_bin / 10 > 0 else "0" + str(hour_bin)  # eg. "16"
    min_str = str(min_bin) if min_bin / 10 > 0 else "0" + str(min_bin)      # eg. "45"
    time_cat = hour_str + ":" + min_str                                     # eg. "16:45"
    
    # Get a floating point representation of the center of the time bin
    time_num = (hour_bin*60 + min_bin + minutes_per_bin / 2.0)/(60*24)      # eg. 0.7065972222222222
    
    time_cos = math.cos(time_num * 2 * math.pi)
    time_sin = math.sin(time_num * 2 * math.pi)
    
    # DATE
    # Parse year, month, day
    date_list = d[0].split('-')
    d_obj = date(int(date_list[0]),int(date_list[1]),int(date_list[2]))
    day_to_str = {0: "Monday",
                  1: "Tuesday",
                  2: "Wednesday",
                  3: "Thursday",
                  4: "Friday",
                  5: "Saturday",
                  6: "Sunday"}
    day_of_week = d_obj.weekday()
    day_cat = day_to_str[day_of_week]
    day_num = (day_of_week + time_num)/7.0
    day_cos = math.cos(day_num * 2 * math.pi)
    day_sin = math.sin(day_num * 2 * math.pi)
    
    year = d_obj.year
    month = d_obj.month
    day = d_obj.day
    
    weekend = 0
    #check if it is the weekend
    if day_of_week in [5,6]:
        weekend = 1
       
    return (year, month, day, time_cat, time_num, time_cos, time_sin, day_cat, day_num, day_cos, day_sin, weekend)



if __name__ == "__main__":
   main(sys.argv[1:])
