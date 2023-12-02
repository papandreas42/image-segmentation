import glob
import os


from os.path import splitext, isfile, join


drive_path = 'training_dataset/'


image_paths = sorted(glob.glob(drive_path + "data/*.tiff"))
label_paths = sorted(glob.glob(drive_path + "labels/*.tif"))

for index, path in enumerate(image_paths):
  id = splitext(path)[0][-3:]
  print(splitext(path)[0][-3:])
  print(drive_path + "data/" + id + ".tiff")
  os.rename(path, drive_path + "data/" + id + ".tiff")

for index, path in enumerate(label_paths):
  id = splitext(path)[0][-3:]
  print(splitext(path)[0][-3:])
  print(drive_path + "labels/" + id + ".tif")
  os.rename(path, drive_path + "labels/" + id + ".tif")
