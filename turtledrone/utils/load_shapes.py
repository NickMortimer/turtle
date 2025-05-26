import json
import pandas as pd

def loadshapes(file):
    print(file)
    lines =[]
    with open(file, "r") as read_file:
        data = json.load(read_file)

    #data = json.loads(''.join(lines).replace("\n", "").replace("'", '"').replace('u"', '"'))
    data =pd.DataFrame(data['shapes'])
    data['FilePath'] =file             
    return(data)