#!/bin/bash

doit clean -f initalise.py  config=$1
doit  -f exifdata.py  config=$1
#doit clean -f getbase.py config=$1
doit clean -f DJIP4Pro.py  config=$1
doit clean -f setsurveyarea.py  config=$1
doit clean -f surveys.py config=$1
doit clean -f reports.py   config=$1
doit -f initalise.py  config=$1
doit  -f exifdata.py  config=$1
#doit clean -f getbase.py config=$1
doit -f DJIP4Pro.py  config=$1
doit -f setsurveyarea.py  config=$1
doit -f surveys.py config=$1
doit -f reports.py   config=$1


