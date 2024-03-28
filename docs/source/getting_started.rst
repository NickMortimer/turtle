Getting Started
===============

Setting Up A Configuration File
-------------------------------

create a directory that will hold your drone imagery

```
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
```


The simplest way to install the package is using pip:

    ``pip install turtledrone`` (not yet implemented!)

Install from the repository
---------------------------

If you want to get a bleeding edge version, you can download the package form our github homepage.

First of all download the software from github:

`Download zip package <https://github.com/NickMortimer/turtle/archive/refs/heads/master.zip>`_

or clone with Git:

    ``git clone https://github.com/NickMortimer/turtle.git``

Then open the installed folder and execute the following command in a command line:

    ``python setup.py install``

.. note::
    If you plan to update regularly, e.g. you have cloned repository, you can instead used ``python setup.py develop``
    that will not copy the the package to the python directory, but will use the files in place. This means that you don't
    have to install the package again if you update the code.