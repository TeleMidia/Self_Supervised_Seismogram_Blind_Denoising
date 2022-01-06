import glob
from tqdm import tqdm
from PIL import Image

files = glob.glob("exported/*/*.tiff")
folder_dict = {}


for file in tqdm(files):
    folder = file.split('/')[1]

    if folder not in folder_dict:
        folder_dict[folder] = []
    
    img = Image.open(file)
    if img.size not in folder_dict[folder]:
        folder_dict[folder].append(img.size)
   


for folder in folder_dict:
    print("\n", folder, folder_dict[folder])
    #for data in folder_dict[folder]:
    #    print(data[0], end=',')
