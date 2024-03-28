Getting Started
===============

Setting Up A Directory File
---------------------------

create a directory that will hold your drone imagery place the config file in the project root directory. 

| data_directory
|   └──{fieldTripId}_config.yml
|       └── {fieldTirpId}
|           ├── raw/
|           │   ├── {downLoadCode}
|           │   │   ├── {SdCardDataID}
|           │   │   │    └── DCIM
|           │   │   └── ...
|           │   │
|           │   ├── process/
|           │   │   ├── surveyareas.csv
|           │   │   ├── surveys.csv
|           │   │   ├── surveyswitharea.csv
|           │   │   ├── imagedata.csv
|           │   │   ├── {countryCode}_{surveyCode}_{dateTime}_survey.csv
|           │   │   ├── {countryCode}_{surveyCode}_{dateTime}_survey_area.csv
|           │   │   └── {countryCode}_{surveyCode}_{dateTime}_survey_area_data.csv
|           │   └── gnss
|           │
|           └── surveys/
|               ├── {countryCode}
|               │   └── {siteID}
|               │       └── {surveyID}
|               │           └── {droneID}_{imageType}_{countryCode}_{siteID}_{dateTime}_{imageCount}.JPG
|               ├── reports
|               └── train


.. code-block:: console
    $python ./turtledrone/exifdata.py {fieldTripId}_config.yml

In the folder structure above:

- ``{fieldTripId}`` create this folder with the name of the field trip eg. TR2022-01 
- ``raw`` is the directory where the SD cards are copied to
- ``process`` keeps files that span across the raw input files
- ``gnss`` data directory for any additional base station files used in RTK processing
- ``reports`` stores reports from the processing
- ``train``  after processing with labelme a training set will be build

.. note::
    You can set up a config file to output data into a survey folder that contains many field trips

Setting Up A Configuration File
-------------------------------

The configuration of the project is held in config.yml

.. literalinclude:: ../../examples/example_config.yml


