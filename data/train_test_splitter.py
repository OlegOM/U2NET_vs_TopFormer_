import pandas as pd
from sklearn.model_selection import train_test_split
import shutil
import os
from os.path import join

DATASET_PATH = "/media/andriy/1TB/Datasets"

def copy_image(row: pd.DataFrame):
    filepath = row["ImageId"]
    src = join(DATASET_PATH, f"train/{filepath}")
    dst = join(DATASET_PATH, f"test/{filepath}")
    shutil.copyfile(src, dst)

def main():
    print("Loading dataset...")
    dataset = pd.read_csv(join(DATASET_PATH, "train.csv"))
    print(f"Dataset loaded! Dataset shape is {dataset.shape}")
    images = list(set(dataset.ImageId))
    print(f"Total number of images: {len(images)}")
    train, test = train_test_split(images, test_size=0.1) # 90/10 split
    print(len(train))
    print(len(test))
    train_dataset = dataset.loc[dataset.ImageId.isin(train)]
    test_dataset = dataset.loc[dataset.ImageId.isin(test)]
    print(f"Train shape: {train_dataset.shape}")
    print(f"Test shape: {test_dataset.shape}")
    print("Writing splits...")
    train_dataset.to_csv(join(DATASET_PATH, "train_train.csv"), index=False)
    test_dataset.to_csv(join(DATASET_PATH, "train_test.csv"), index=False)
    print("Splits are written!")
    print("Copying images from test split to input_images folder...")
    shutil.rmtree(join(DATASET_PATH, "input_images"))	
    os.mkdir(join(DATASET_PATH,"input_images"))
    test_dataset.apply(copy_image, axis=1)
    print("All images copied!")

def eda():
    print("Loading dataset...")
    dataset = pd.read_csv(join(DATASET_PATH, "train.csv"))
    # dataset = pd.read_csv(join(DATASET_PATH, "train_test.csv"))
    print(f"Dataset loaded! Dataset shape is {dataset.shape}")
    print(dataset.ImageId.value_counts())

if __name__ == "__main__":
    eda()
