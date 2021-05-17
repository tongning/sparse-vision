import os
import csv
import shutil

class_codes = os.listdir('train')
data_folder = 'tiny-imagenet-200'

if not os.path.exists(os.path.join(data_folder, 'val_organized')):
    os.mkdir(os.path.join(data_folder, 'val_organized'))
if not os.path.exists(os.path.join(data_folder, 'test_organized')):
    os.mkdir(os.path.join(data_folder, 'test_organized'))

for class_code in class_codes:
    class_val_folder_path = os.path.join(data_folder, 'val_organized', class_code)
    class_test_folder_path = os.path.join(data_folder, 'test_organized', class_code)
    if not os.path.exists(class_val_folder_path):
        os.mkdir(class_val_folder_path)
    if not os.path.exists(class_test_folder_path):
        os.mkdir(class_test_folder_path)

val_images_folder = os.path.join(data_folder, 'val', 'images')
val_images_names = os.listdir(val_images_folder)
val_annotations_file = os.path.join(data_folder, 'val', 'val_annotations.txt')

with open(val_annotations_file) as tsv:
    for line in csv.reader(tsv, dialect='excel-tab'):
        filename = line[0]
        class_code = line[1]
        source_path = os.path.join(val_images_folder, filename)
        destination_path = os.path.join(data_folder, 'val_organized', class_code, filename)
        shutil.copy(source_path, destination_path)
        print("Copying")

exit(0)