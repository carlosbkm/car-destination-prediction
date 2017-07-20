# Car destination prediction model
The goal of this project is to create a predictive model to predict the position destination of a car based on a date and a starting position.

## System requirements
The has been done using Python and Spark as the main technologies. To be able to run the notebooks you need to have installed the following:
* Spark 2.2
* Python 3.6. Also, these packages are required:
  * notebook
  * findspark
  * numpy
  * pandas
  * scikit-learn
  * python-geohash
  * matplotlib
  * gmaps
  
If you have Anaconda installed in your computer, you can easily get your Python environment ready by loading [python-environment.yml](python-environment.yml), which contains all the dependencies. You can do it by simply running:
```
conda env create -f python-environment.yml
```

Although is not necessary to perform the data processing and running the model, you will need a *Google Maps Javascript API Key* to visualize maps with gmaps in some notebook. After you activate it in the Google Developers Console, you must add it to your environment by:
```
export GOOGLE_API_KEY=[Your fantastic API KEY goes here]
```
## Project structure
The project contains the following type of files:
* __Jupyter notebooks__. They contain the code for the project implementation. You will better understand the project by following in this order:
   * _data-cleansing.ipynb_: Contains the code for read and explore the raw dataset, make some data cleanup transformations and visualization (maps). It produces as a result the file [processed-dataset.csv](processed-dataset.csv)
   * _features-preparation_: Normalize the data, expands dimensionality, and in general compute new features which could be useful depending on the model that choose later. It produces [featured-dataset.csv](featured-dataset.csv)
   * _random-forest-model.ipynb_: Implements Random Forest Prediction Model.
   * _k-nearest-model.ipynb_: Implements K-Nearest Neighbor Prediction Model.
 * __Python script__.
   * _predict-destination.py_: This script runs the models generated in the notebooks to predict the heading of a vehicle based on its starting position and time.
 * __Models__. The trained models are stored in the following files:
   * _random_forest_model.pkl_
   * _k_nearest_model.pkl_
 * __Analysis Documentation__. There is a PDF file which details all the analysis, decision making, and discuss the code of the implementation: [predictive-analytics-connected-car.pdf](predictive-analytics-connected-car.pdf)

## Running the models
To ease the evaluation of the model, I've created a simple script in Python so that you can play with different values and see the prediction.

To run the script, Spark is not needed, and only _numpy_, _scikit-learn_ and _geohash_ Python packages are required. However, if you loaded the environment which I provided with the project, you'll have everything you need to go. 

From a command line if you type:
```
./predict-destination.py -h
```
You will get help on how to use it:
```
usage: predict-destination.py [-h] {forest,knn} time latitude longitude

positional arguments:
  {forest,knn}  Predictive model to use, can be either forest or knn
  time          Start trip time, with the format "yyyy-MM-dd HH:mm:ss". It
                must be between quotation marks. For instance, you coud use:
                "2017-05-24 12:26:37"
  latitude      Latitude of the trip start position. For instance, you could
                use: 47.409291
  longitude     Longitude of the trip end position. For instance, you could
                use: 8.546942

optional arguments:
  -h, --help    show this help message and exit

```
For example, if you wanted to make a prediction using the K-Nearest Neighbor Model:
```
./predict-destination.py knn "2017-05-29 18:23:27" 32.989318 -97.263840
```
***
#### And that's all. Enjoy the code! Feedback is welcome ;-)
