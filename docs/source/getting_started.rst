Getting Started
===============

Setting Up A Configuration File
-------------------------------

create a directory that will hold your drone imagery place the config file in the project root directory

::
project_root/
│
├── config.yml/
├── survey_area/
│   ├── {shortName}_{longName}
│   │   ├── {shortName}_{longName}_SurveyArea.shp
│   │   └── {shortName}_{longName}_Survey_AOI.shp
│
├── raw/
│   ├── {downLoadCode}
│   │   ├── {SdCardDataID}
│   │   │    └── DCIM
│   │   └── ...
│   │
│   ├── processing/
│   │   ├── surveyareas.csv
│   │   ├── surveys.csv
│   │   ├── surveyswitharea.csv
│   │   ├── imagedata.csv
│   │   ├── {countryCode}_{surveyCode}_{dateTime}_survey.csv
│   │   ├── {countryCode}_{surveyCode}_{dateTime}_survey_area.csv
│   │   └── {countryCode}_{surveyCode}_{dateTime}_survey_area_data.csv
│   └── gnss
│
└── surveys/
    ├── {countryCode}
    ├── reports
    └── train
::



.. note::
    You can set up a config file to output data into a survey folder that contains many field trips