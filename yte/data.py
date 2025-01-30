import cv2
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class XrayDataset():
    def __init__(self, dataframe, train_transform=None):
        self.dataframe = dataframe
        self.train_transform = train_transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_path = self.dataframe.iloc[idx].path
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label = torch.as_tensor(self.dataframe.iloc[idx].label)

        if self.train_transform is not None:
            image = self.train_transform(image=image)["image"]

        else:
            image = A.Compose([A.Normalize(),ToTensorV2()])(image=image)["image"]

        return {"image":image, "label":label}