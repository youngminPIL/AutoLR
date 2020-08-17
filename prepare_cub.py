import os
import shutil

# You only need to change this line to your dataset download path
download_path = '/home/ro/FG/CUB_200_2011'

save_path = download_path + '/pytorch'
if not os.path.isdir(save_path):
    os.mkdir(save_path)
#-----------------------------------------
#test

images_path = download_path + '/images'

train_save_path = download_path + '/pytorch/train'
test_save_path = download_path + '/pytorch/test'
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)
if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)



dir_names = os.listdir(images_path)

for i in range(len(dir_names)):
    if int(dir_names[i][0:3]) <= 100:
        shutil.move(images_path +'/'+ dir_names[i], train_save_path)
    else:
        shutil.move(images_path +'/'+ dir_names[i], test_save_path)



