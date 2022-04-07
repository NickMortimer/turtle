import sys, os, datetime, json
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.pdfmetrics import registerFont
from rlextra.rml2pdf import rml2pdf
from rlextra.radxml.html_cleaner import cleanBlocks
from rlextra.radxml.xhtml2rml import xhtml2rml
import preppy
import pandas as pd



orderdata = [['plate',20.,18.,22.,12.,1],['fork',20.,18.,22.,12.,10],['spoon',20.,18.,22.,12.,15]]
orders = pd.DataFrame(orderdata,columns=['description','price','net','vat','gross','quantity'])
data = dict(orders=orders.to_dict('recprds'))
ns = dict(data=data, format="long" )

#we usually put some standard things in the preppy namespace
ns['DATE_GENERATED'] = datetime.date.today()
ns['showBoundary'] = "1" 

#let it know where it is running; trivial in a script, confusing inside
#a big web framework, may be used to compute other paths.  In Django
#this might be relative to your project path,
ns['RML_DIR'] = os.getcwd()     #os.path.join(settings.PROJECT_DIR, appname, 'rml')

#we tend to keep fonts in a subdirectory.  If there won't be too many,
#you could skip this and put them alongside the RML
FONT_DIR = ns['FONT_DIR'] = os.path.join(ns['RML_DIR'], 'fonts')


#directory for images, PDF backgrounds, logos etc relating to the PDF
ns['RSRC_DIR'] = os.path.join(ns['RML_DIR'], 'resources')

#We tell our template to use Preppy's standard quoting mechanism.
#This means any XML characters (&, <, >) will be automatically
#escaped within the prep file.
template = preppy.getModule('rml/report.prep')
rmlText = template.getOutput(ns, quoteFunc=preppy.stdQuote)
rml2pdf.go(rmlText, outputFileName='test.pdf')

