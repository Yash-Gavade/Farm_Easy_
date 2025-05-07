import os
import shutil

# Paths to your dataset
dataset_path = "Dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Create the new directories for YOLO format
new_dataset_path = "yolo_dataset"
images_path = os.path.join(new_dataset_path, "images")
labels_path = os.path.join(new_dataset_path, "labels")
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)

# Define class labels (index starts from 0)
classes = {
    "freshbanana": 0,
    "rottenbanana": 1
}

# Helper function to copy images and create labels
def copy_images_and_create_labels(source_path, split):
    for fruit_class in os.listdir(source_path):
        class_path = os.path.join(source_path, fruit_class)
        if os.path.isdir(class_path):
            class_index = classes[fruit_class]
            for image_name in os.listdir(class_path):
                image_path = os.path.join(class_path, image_name)
                if image_name.endswith(".jpg") or image_name.endswith(".png"):
                    # Move image to YOLO images directory
                    new_image_path = os.path.join(images_path, f"{split}_{fruit_class}_{image_name}")
                    shutil.copy(image_path, new_image_path)

                    # Create label for the image
                    label_file = os.path.join(labels_path, f"{split}_{fruit_class}_{image_name.replace(image_name.split('.')[-1], 'txt')}")
                    with open(label_file, 'w') as f:
                        # Format: class x_center y_center width height (normalized)
                        # For simplicity, you can assume the bounding box is the entire image
                        # You will need to adjust bounding boxes based on actual data in your case
                        width, height = 1, 1  # Placeholder, you will need actual bbox data here
                        x_center, y_center = 0.5, 0.5  # Placeholder, you will need actual bbox data here
                        f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

# Process the train and test datasets
copy_images_and_create_labels(train_path, "train")
copy_images_and_create_labels(test_path, "test")

# Create the data.yaml file
data_yaml = """
path: ./yolo_dataset
train: images
val: images

names:
  0: freshbanana
  1: rottenbanana
"""

with open("yolo_dataset/data.yaml", "w") as yaml_file:
    yaml_file.write(data_yaml)

print("Dataset is organized in YOLO format.")
