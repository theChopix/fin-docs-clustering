import kagglehub
import shutil
import os

def download_dataset(target_dir: str = 'data') -> None:
    # download dataset
    path = kagglehub.dataset_download("mehaksingal/personal-financial-dataset-for-india")

    os.makedirs(target_dir, exist_ok=True)

    # structure of the path is path/<doc_type>/*.jpg, 
    #  move all files to target_dir
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            for file in os.listdir(dir_path):
                shutil.move(os.path.join(dir_path, file), os.path.join(target_dir, f'{dir}_{file}'))
    

if __name__ == "__main__":
    download_dataset()