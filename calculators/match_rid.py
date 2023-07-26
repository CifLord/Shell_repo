from pathlib import Path
from ocpmodels.datasets import LmdbDataset
import json, argparse, os


def read_options():

    parser = argparse.ArgumentParser()

    parser.add_argument("-l", "--lmdb", dest="lmdb_path", type=str,
                        help="Location of lmdbs")
    parser.add_argument("-j", "--json", dest="json_predictions_path", type=str,
                        help="Location of json files")
    parser.add_argument("-o", "--output", dest="output_path", type=str,
                        help="Location of output files")

    args = parser.parse_args()
    return args
    
if __name__=="__main__":

    args = read_options()
    ppath=Path(args.lmdb_path)

    sid_E100s={}
    db_paths = sorted(ppath.glob("*ads.lmdb"))
    for db_path in db_paths:
        datas=LmdbDataset({"src":db_path})
        for i, data in enumerate(datas):
            print(data)
            if data.preds=='Failed':
                sid_E100s[round(data.y,2)]=data.rid
            else:
                pass

    # with open("./prediction/E100_rid.json", "w") as outfile:
    #     json.dump(sid_E100s, outfile)
    ppath=Path(args.json_predictions_path)
    E100_id={}
    s100_paths = sorted(ppath.glob("*.json"))
    for db_path in s100_paths:
        try:
            with open(db_path) as f:
                data=json.load(f)
                E100_id[round(data["unrelaxed_energy"],2)]=db_path.split('+')[-1].replace('.traj','')
        except:
            pass

    sid_id = {}
    for key in sid_E100s:
        if key in E100_id:
            sid_id[sid_E100s[key]] = E100_id[key]
    with open(os.path.join(args.output_path, "sid_id.json"), "w") as outfile:
        json.dump(sid_id, outfile)
