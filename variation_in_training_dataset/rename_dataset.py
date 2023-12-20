import os
from os.path import splitext

def set_paths(drive_path) :

    image_paths = sorted(glob.glob(drive_path + "data/*.tiff"))
    label_paths = sorted(glob.glob(drive_path + "labels/*.tif"))
    dir_img = drive_path + 'data/'
    dir_mask = drive_path + 'labels/'
    dir_checkpoint = Path('./checkpoints/')
    print("Total Observations:\t", 'images', len(image_paths), ', labels', len(label_paths))

    return image_paths, label_paths, dir_img, dir_mask, dir_checkpoint

# Images
def rename_images(image_paths,drive_path ):
    for index, path in enumerate(image_paths):
        id = splitext(path)[0][-3:]
        print(splitext(path)[0][-3:])
        print(drive_path + "data/" + id + ".tiff")
        os.rename(path, drive_path + "data/" + id + ".tiff")

# Labels
def rename_labels(label_paths,drive_path ):
    for index, path in enumerate(label_paths):
        id = splitext(path)[0][-3:]
        print(splitext(path)[0][-3:])
        print(drive_path + "labels/" + id + ".tif")
        os.rename(path, drive_path + "labels/" + id + ".tif")

drive_path = 'training_dataset/'
image_paths, label_paths, dir_img, dir_mask, dir_checkpoint = set_paths(drive_path=drive_path)

rename_images(image_paths = image_paths,drive_path = drive_path)
rename_labels(label_paths = label_paths,drive_path =drive_path)