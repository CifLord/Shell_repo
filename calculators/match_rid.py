from pathlib import Path
from ocpmodels.datasets import LmdbDataset
import json

#Need to change to your LMDB file path
#---------------------------------------------------------------------------------
ppath=Path('/shareddata/shell/Shell_repo/calculators/prediction/') #--------------
#---------------------------------------------------------------------------------

sid_E100s={}
db_paths = sorted(ppath.glob("*ads.lmdb"))
for db_path in db_paths:
    datas=LmdbDataset({"src":db_path})    
    for i in range(len(datas)): 
        if datas.preds=='Failed':       
            sid_E100s[round(datas[i].y,2)]=datas[i].rid
        else: 
            pass

# with open("./prediction/E100_rid.json", "w") as outfile:
#     json.dump(sid_E100s, outfile)
ppath=Path('/project/grabow/rtran25/Shell_predictiosn/prediction/continue_result/')
E100_id={}
s100_paths = sorted(ppath.glob("*.json"))
for db_path in s100_paths:
    try:
        with open(db_path) as f:
            data=json.load(f)
            E100_id[round(data["unrelaxed_energy"],2)]=db_path.split('+')[-1].replace('.traj','')
    except:
        pass
# with open("./prediction/E100_id.json", "w") as outfile:
#     json.dump(E100_id, outfile)

sid_id = {}
for key in sid_E100s:
    if key in E100_id:
        sid_id[sid_E100s[key]] = E100_id[key]
with open("./prediction/sid_id.json", "w") as outfile:
    json.dump(sid_id, outfile)