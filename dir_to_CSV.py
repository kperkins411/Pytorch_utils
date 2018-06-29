from glob import glob
import pandas as pd
import os
from utils import CSV_Dataset_FullyQualifiedFilenames

def directories_to_CSV(csv_file:str,
                       root_dir:str,
                       *,
                       df_column_names:list=["file", "label"],
                       subdirs:list = [None],
                       use_full_path = True):
    '''
    Generate a csv file from a directory based collection of training images which is
    of the form:
    root_dir
        dogs
          dog1.png
          dog2.png
          :
        cats
          cat1.png
          cat2.png
          :
    Args:
        csv_file (string): CSV file to create
        root_dir (string): fully qualified path of root_directory
        df_column_names (list):name of the file, qualified from root_dir
        subdirs (list): directories of interest
        use_full_path (boolean): record the full path of the filename or just the part withoutout the root_dir
    '''
    #take care of forgotton /
    # if not root_dir.endswith('/'):
    #     root_dir +='/'

    df = pd.DataFrame(columns=df_column_names)

    for dir in subdirs:
        dir = os.path.join(root_dir, dir) if dir is not None else root_dir
        dir = dir + "/**/*.png"
        for image in glob(dir, recursive = True):   #get all the png files
            image_path = image if use_full_path == True else\
                image.replace((root_dir + dir + '/'),'')                    #strip root dir
            image_class =image_path.split('/')[-2]                          #get directory name (the class)

            #save to list
            df = df.append({
                df_column_names[0]: image_path,
                df_column_names[1]: image_class
                }, ignore_index=True)

    df.to_csv(csv_file, index=False)

if __name__=='__main__':
    root_dir = "/home/keith/data/plant_seedlings"
    directories_to_CSV(csv_file = "tmp.csv",root_dir=root_dir, df_column_names=["file","label"],  use_full_path = True)

    ds = CSV_Dataset_FullyQualifiedFilenames("tmp.csv")
    ds.__getitem__(0)