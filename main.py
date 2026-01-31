import click
import json
import os

from download_dataset import download_dataset
from clustering_pipeline.feature_extraction import feature_extraction

@click.command()
@click.option('--input_data_path', default='data', help='Path to the input data directory.')
@click.option('--output_data_path', default='results', help='Path to the experiment results directory.')
def run(input_data_path: str, output_data_path: str):

    # if the input data path does not exist or is empty, download the dataset
    if not os.path.exists(input_data_path) or not os.listdir(input_data_path):
        download_dataset(target_dir=input_data_path)
    
    # extraction step, process all images in the input data path into the output data path
    pipeline_step = 'feature_extraction'
    for type_doc_dir in os.listdir(input_data_path):
        os.makedirs(os.path.join(output_data_path, pipeline_step, type_doc_dir), exist_ok=True)
        for filename in os.listdir(os.path.join(input_data_path, type_doc_dir)):
            if filename.endswith('.jpg'):
                image_path = os.path.join(input_data_path, type_doc_dir, filename)
                features = feature_extraction.extract_features_from_image(image_path)
                
                output_file_path = os.path.join(output_data_path, pipeline_step, type_doc_dir, f"{os.path.splitext(filename)[0]}_features.json")
                with open(output_file_path, 'w') as outfile:
                    json.dump(features, outfile)


if __name__ == '__main__':
    run()

    


    
