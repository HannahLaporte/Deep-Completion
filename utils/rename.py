import os

folder_path = '/root/autodl-tmp/VA-DepthNet/Dataset/save/1200/2011_09_26_drive_0020_sync'
files = os.listdir(folder_path)

n = 349 
# 150 215 349 

for file in files:
    new_name = f"{n:05d}{os.path.splitext(file)[-1]}"
    new_path = os.path.join(folder_path, new_name)
    print(new_path)
    os.rename(os.path.join(folder_path, file), new_path)

    n += 1