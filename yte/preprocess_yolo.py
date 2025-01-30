import pandas as pd
import os, glob
from sklearn.model_selection import train_test_split
import tqdm
import random
import pydicom, shutil
from utils import parallel_iterate
random.seed(42)


os.makedirs("/home/ubuntu/ductq/data_yolo_c1/images/train", exist_ok=True)
os.makedirs("/home/ubuntu/ductq/data_yolo_c1/images/val", exist_ok=True)
os.makedirs("/home/ubuntu/ductq/data_yolo_c1/labels/train", exist_ok=True)
os.makedirs("/home/ubuntu/ductq/data_yolo_c1/labels/val", exist_ok=True)

df = pd.read_csv("/home/ubuntu/ductq/csvs/train_new.csv")

c0 = {"Pleural thickening":0,"Pulmonary fibrosis":1}
class0 = df[df.class_id.isin([11,13])].reset_index(drop=True)
class0["new_class"] = class0["class_name"].apply(lambda x: c0[x])
class0["image_path"] = class0["image_id"].apply(lambda x: f"/home/ubuntu/duynd/AIyte/secondnd/input/train/{x}.dicom")

class0_train, class0_val = train_test_split(class0, test_size=0.15)
class0_train = class0_train.reset_index(drop=True).sort_values("image_id").reset_index(drop=True)
class0_val = class0_val.reset_index(drop=True).sort_values("image_id").reset_index(drop=True)

class0_train.to_csv("/home/ubuntu/ductq/csvs/yolo_csv/class1_train.csv",index=False)
class0_val.to_csv("/home/ubuntu/ductq/csvs/yolo_csv/class1_val.csv",index=False)



class0_train_ = class0_train.drop_duplicates("image_id").reset_index(drop=True)
id_train = list(class0_train_.image_id)


for id in tqdm.tqdm(id_train):
    file_path = f"/home/ubuntu/ductq/data_yolo_c1/labels/train/{id}_train.txt"
    with open(file_path, 'w') as file:
        pass


class0_val_ = class0_val.drop_duplicates("image_id").reset_index(drop=True)
id_val = list(class0_val_.image_id)

for id in tqdm.tqdm(id_val):
    file_path = f"/home/ubuntu/ductq/data_yolo_c1/labels/val/{id}_val.txt"
    with open(file_path, 'w') as file:
        pass


def get_shape(path):
    data = pydicom.dcmread(path).pixel_array
    
    return data.shape[1], data.shape[0]


def txt_write(id, df, mode):
    # if i > 2: continue
    info = df.iloc[id]
    box = eval(info.box)

    file = f"/home/ubuntu/ductq/data_yolo_c1/labels/{mode}/{info['image_id']}_{mode}.txt"
    x_shape, y_shape = get_shape(info.image_path)

    cls = info.new_class
    x_center = (box[0] + box[2])/(2*x_shape)
    y_center = (box[1] + box[3])/(2*y_shape)
    w = (box[2] - box[0])/x_shape
    h = (box[3] - box[1])/y_shape
    
    data = str(cls) + " " + str(x_center) + " " + str(y_center) + " " + str(w) + " " + str(h) + "\n"
    
    with open(file,'a+') as f:
        f.write(data)



parallel_iterate(range(len(class0_train)), txt_write, df=class0_train, mode="train", workers=12)
parallel_iterate(range(len(class0_val)), txt_write, df=class0_val, mode="val", workers=12)

print("-"*100)

train_id = list(class0_train_.image_id)
val_id = list(class0_val_.image_id)


for id in tqdm.tqdm(train_id):
    shutil.copy2(f"/home/ubuntu/ductq/data/{id}.jpg","/home/ubuntu/ductq/data_yolo_c1/images/train")


file = list(glob.glob("/home/ubuntu/ductq/data_yolo_c1/images/train/*.jpg"))
for f in tqdm.tqdm(file):

    os.rename(f, f"/home/ubuntu/ductq/data_yolo_c1/images/train/{f.split('/')[-1][0:-4]}_train.jpg")



for id in tqdm.tqdm(val_id):
    shutil.copy2(f"/home/ubuntu/ductq/data/{id}.jpg","/home/ubuntu/ductq/data_yolo_c1/images/val")


file = list(glob.glob("/home/ubuntu/ductq/data_yolo_c1/images/val/*.jpg"))
for f in tqdm.tqdm(file):

    os.rename(f, f"/home/ubuntu/ductq/data_yolo_c1/images/val/{f.split('/')[-1][0:-4]}_val.jpg")