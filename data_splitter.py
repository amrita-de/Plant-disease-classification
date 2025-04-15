import os
import shutil
from sklearn.model_selection import train_test_split

def split_data(data_dir, train_dir, val_dir, val_size=0.2):
    """
    Splits a dataset into training and validation sets.

    Args:
        data_dir (str): Path to the directory containing the dataset.
        train_dir (str): Path to the directory where the training set will be created.
        val_dir (str): Path to the directory where the validation set will be created.
        val_size (float): The proportion of the dataset to include in the validation split.
    """

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    class_names = os.listdir(data_dir)

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            train_images, val_images = train_test_split(images, test_size=val_size, random_state=42)

            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

            for img in train_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(train_dir, class_name, img))

            for img in val_images:
                shutil.copy(os.path.join(class_dir, img), os.path.join(val_dir, class_name, img))

# Example usage:
data_directory = "dataset" # The dataset is inside the current directory.
train_directory = "train"
validation_directory = "valid"
split_data(data_directory, train_directory, validation_directory)