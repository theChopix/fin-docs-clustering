import kagglehub
import shutil
import os

def download_dataset(target_dir: str = 'data') -> None:
    # download dataset
    path = kagglehub.dataset_download("mehaksingal/personal-financial-dataset-for-india")

    os.makedirs(target_dir, exist_ok=True)

    for file in os.listdir(path):
        shutil.move(os.path.join(path, file), target_dir)
    
    print("Dataset moved to:", target_dir)


if __name__ == "__main__":
    download_dataset()