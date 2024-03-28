#!/bin/bash

doit -f ../turtledrone/process/initalise.py  config=$1
doit -f exifdata.py  config=$1
#doit -r getbase.py config=$1
doit -f DJIP4RTK.py  config=$1
doit -f setsurveyarea.py  config=$1
doit -f surveys.py config=$1
doit -f reports.py   config=$1


