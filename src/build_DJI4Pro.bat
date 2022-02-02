set config=%1
doit -f initalise.py  config=%config%
doit -f exifdata.py  config=%config%
doit -f surveys.py config=%config%
doit -f DJIP4Pro.py  config=%config%
doit -f reports.py  config=%config%


