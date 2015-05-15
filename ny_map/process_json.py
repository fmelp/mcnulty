import json
import pandas as pd
import re
import numpy as np

bd = pd.read_csv("~/Desktop/metis/mcnulty_project/borgo_data/Ospiti.csv", sep=';')
bd['CAP'] = bd['CAP'].dropna().apply(lambda x: re.sub("[^0-9]", "", x)).apply(lambda x: x if len(x) == 5 else np.nan)
bd = bd[bd['CAP'] != np.nan]
bd = bd[~(bd['CAP'].isnull())]

zip_list = bd['CAP'].tolist()
zip_dict = dict( [ (i, zip_list.count(i)) for i in set(zip_list) ] )
print zip_dict

with open("nyc_open_data.json") as json_file:
    json_data = json.load(json_file)

print json_data['features'][55]['properties']

for i in range(len(json_data['features'])):
    if json_data['features'][i]['properties']['postalCode'] in zip_dict.keys():
        print i
        json_data['features'][i]['properties']['num_guests'] = zip_dict[json_data['features'][i]['properties']['postalCode']]
    else:
        json_data['features'][i]['properties']['num_guests'] = 0


with open("fixed.json", "w") as f:
    json.dump(json_data, f)