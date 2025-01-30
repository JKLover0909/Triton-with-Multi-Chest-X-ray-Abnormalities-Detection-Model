import pandas as pd
import os, glob
from sklearn.model_selection import train_test_split
import tqdm
import random
import pydicom, shutil
from utils import parallel_iterate
random.seed(42)


os.makedirs("/home/ubuntu/ductq/yte2/data_yolo/images/train", exist_ok=True)
os.makedirs("/home/ubuntu/ductq/yte2/data_yolo/images/val", exist_ok=True)
os.makedirs("/home/ubuntu/ductq/yte2/data_yolo/labels/train", exist_ok=True)
os.makedirs("/home/ubuntu/ductq/yte2/data_yolo/labels/val", exist_ok=True)



train_case = pd.read_csv("/home/ubuntu/ductq/yte2/csv/yolo_csv/train_case.csv")
id_train = list(train_case.image_id)

train_box = pd.read_csv("/home/ubuntu/ductq/yte2/csv/yolo_csv/train_box.csv")


for id in tqdm.tqdm(id_train):
    file_path = f"/home/ubuntu/ductq/yte2/data_yolo/labels/train/{id}_train.txt"
    with open(file_path, 'w') as file:
        pass

val_case = pd.read_csv("/home/ubuntu/ductq/yte2/csv/yolo_csv/val_case.csv")
id_val = list(val_case.image_id)

val_box = pd.read_csv("/home/ubuntu/ductq/yte2/csv/yolo_csv/val_box.csv")

for id in tqdm.tqdm(id_val):
    file_path = f"/home/ubuntu/ductq/yte2/data_yolo/labels/val/{id}_val.txt"
    with open(file_path, 'w') as file:
        pass


def get_shape(path):
    data = pydicom.dcmread(path).pixel_array
    
    return data.shape[1], data.shape[0]


def txt_write(id, df, mode):
    # if i > 2: continue
    info = df.iloc[id]
    box = eval(info.box)

    file = f"/home/ubuntu/ductq/yte2/data_yolo/labels/{mode}/{info['image_id']}_{mode}.txt"
    x_shape, y_shape = get_shape(info.image_path)

    x_center = (box[0] + box[2])/(2*x_shape)
    y_center = (box[1] + box[3])/(2*y_shape)
    w = (box[2] - box[0])/x_shape
    h = (box[3] - box[1])/y_shape
    
    data = str(0) + " " + str(x_center) + " " + str(y_center) + " " + str(w) + " " + str(h) + "\n"
    
    with open(file,'a+') as f:
        f.write(data)



parallel_iterate(range(len(train_box)), txt_write, df=train_box, mode="train", workers=12)
print("-"*100)

parallel_iterate(range(len(val_box)), txt_write, df=val_box, mode="val", workers=12)
print("-"*100)





def copy_train(id, ids):
    idx = ids[id]
    shutil.copy2(f"/home/ubuntu/ductq/yte/data/{idx}.jpg","/home/ubuntu/ductq/yte2/data_yolo/images/train")

parallel_iterate(range(len(id_train)), copy_train, ids=id_train, workers=12)
# for id in tqdm.tqdm(train_id):
    
print("-"*100)

file = list(glob.glob("/home/ubuntu/ductq/yte2/data_yolo/images/train/*.jpg"))

for f in tqdm.tqdm(file):

    os.rename(f, f"/home/ubuntu/ductq/yte2/data_yolo/images/train/{f.split('/')[-1][0:-4]}_train.jpg")

print("-"*100)

# for id in tqdm.tqdm(train_id):
#     shutil.copy2(f"/home/ubuntu/ductq/yte/data/{id}.jpg","/home/ubuntu/ductq/yte2/data_yolo/images/val")

def copy_val(id, ids):
    idx = ids[id]
    shutil.copy2(f"/home/ubuntu/ductq/yte/data/{idx}.jpg","/home/ubuntu/ductq/yte2/data_yolo/images/val")

parallel_iterate(range(len(id_val)), copy_val, ids=id_val, workers=12)

print("-"*100)

file = list(glob.glob("/home/ubuntu/ductq/yte2/data_yolo/images/val/*.jpg"))

for f in tqdm.tqdm(file):

    os.rename(f, f"/home/ubuntu/ductq/yte2/data_yolo/images/val/{f.split('/')[-1][0:-4]}_val.jpg")