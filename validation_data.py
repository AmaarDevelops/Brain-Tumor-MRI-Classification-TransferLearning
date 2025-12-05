import os
import shutil
from sklearn.model_selection import train_test_split

# paths
dataset_dir = "./Training"   # MODIFY to your actual path
val_ratio = 0.2

# target folders
train_out = "train"
val_out = "val"

os.makedirs(train_out, exist_ok=True)
os.makedirs(val_out,  exist_ok=True)

classes = os.listdir(dataset_dir)

for cls in classes:
    cls_path = os.path.join(dataset_dir, cls)
    images = os.listdir(cls_path)

    train_imgs, val_imgs = train_test_split(images, test_size=val_ratio, random_state=42)

    os.makedirs(os.path.join(train_out, cls), exist_ok=True)
    os.makedirs(os.path.join(val_out, cls), exist_ok=True)

    for img in train_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(train_out, cls, img))

    for img in val_imgs:
        shutil.copy(os.path.join(cls_path, img), os.path.join(val_out, cls, img))

print("Train/Validation split finished!")
