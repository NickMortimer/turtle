set config=%1
doit -f initalise.py  config=%config%
doit -f exifdata.py  config=%config%
doit -f DJIP4RTK.py  config=%config%bu  
doit -f setsurveyarea.py  config=%config%
doit -f surveys.py config=%config%
doit -f reports.py  config=%config%


