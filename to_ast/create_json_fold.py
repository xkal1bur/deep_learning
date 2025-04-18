import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold


df = pd.read_csv("train-audios.csv")
tmp = df.drop("filename", axis=1)

a = [(df.loc[row[0]]["filename"], ','.join((np.where(row[1] == 1)[0]+1).astype(str))) for row in tmp.iterrows()]
adf=pd.DataFrame(a, columns=["filename", "labels"])

adf["labels"] = adf["labels"].replace("", "0")

k_folds = 5
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

adf["fold"] = -1
for fold, (_, val_index) in enumerate(kf.split(df), start=1):
    adf.loc[val_index, 'fold'] = fold

for fold in [1, 2, 3, 4, 5]:
    base_path = f"./data/audios_train_16k/"
    train_wav_list = []
    eval_wav_list = []
    for i in range(len(adf)):
        cur_path = adf.loc[i, "filename"]
        cur_labels = adf.loc[i, "labels"].split(",")
        labels = ','.join(['/m/07rwj'+lab.zfill(2) for lab in cur_labels])
        cur_dict = {"wav": base_path + cur_path, "labels": labels}
        if adf.loc[i, "fold"] == fold:
            eval_wav_list.append(cur_dict)
        else:
            train_wav_list.append(cur_dict)
    
    print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))

    with open('./data/datafiles/animals_sounds_train_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': train_wav_list}, f, indent=1)

    with open('./data/datafiles/animal_sounds_eval_data_'+ str(fold) +'.json', 'w') as f:
        json.dump({'data': eval_wav_list}, f, indent=1)
    
