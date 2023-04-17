#!/bin/bash

doit -f initalise.py  config=$1
doit -f exifdata.py  config=$1
doit -f DJIP4Pro.py  config=$1
doit -f setsurveyarea.py  config=$1
doit -f surveys.py config=$1
doit -f reports.py   config=$1


