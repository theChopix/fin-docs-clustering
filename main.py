import click
import json
import os

from clustering_pipeline.download_dataset import download_dataset
from clustering_pipeline.feature_extraction import feature_extraction
from clustering_pipeline.feature_engineering.feature_engineering import process_images_features
import yaml

@click.command()
@click.option('--input_data_path', default='data', help='Path to the input data directory.')
@click.option('--config', default='configs/config.yaml', help='Path to the experiment configuration file (with feature weights).')
@click.option('--output_data_path', default='results', help='Path to the experiment results directory.')
def run(input_data_path: str, config: str, output_data_path: str):

    # load feature weights from config file
    with open(config, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
        feature_weights = config_data.get('feature_weights', {})

    # if the input data path does not exist or is empty, download the dataset
    if not os.path.exists(input_data_path) or not os.listdir(input_data_path):
        download_dataset(target_dir=input_data_path)
    
    # extraction step, process all images in the input data path into the output data path
    pipeline_step = 'feature_extraction'
    if not os.path.exists(output_data_path) or not os.listdir(os.path.join(output_data_path, pipeline_step)):
    
        for type_doc_dir in os.listdir(input_data_path):
            click.echo(f"[feature_extraction] Processing documents of type '{type_doc_dir}'...")

            os.makedirs(os.path.join(output_data_path, pipeline_step, type_doc_dir), exist_ok=True)
            for filename in os.listdir(os.path.join(input_data_path, type_doc_dir)):
                if filename.endswith('.jpg'):
                    image_path = os.path.join(input_data_path, type_doc_dir, filename)
                    features = feature_extraction.extract_features_from_image(image_path)
                    
                    output_file_path = os.path.join(output_data_path, pipeline_step, type_doc_dir, f"{os.path.splitext(filename)[0]}_features.json")
                    with open(output_file_path, 'w') as outfile:
                        json.dump(features, outfile)

            click.echo(f"[feature_extraction] Processed documents of type '{type_doc_dir}'.")

    # feature engineering step: process extracted features into fused feature vectors
    pipeline_step = 'feature_engineering'
    if not os.path.exists(output_data_path) or not os.path.exists(os.path.join(output_data_path, pipeline_step)):
        images_features = {}
        for type_doc_dir in os.listdir(os.path.join(output_data_path, 'feature_extraction')):
            for filename in os.listdir(os.path.join(output_data_path, 'feature_extraction', type_doc_dir)):
                if filename.endswith('_features.json'):
                    image_id = f"{type_doc_dir}/{os.path.splitext(filename)[0].replace('_features', '')}"
                    with open(os.path.join(output_data_path, 'feature_extraction', type_doc_dir, filename), 'r') as infile:
                        features = json.load(infile)
                        images_features[image_id] = features

        click.echo("[feature_engineering] Processing features for clustering...")
        fused_features = process_images_features(
            images_features=images_features,
            feature_weights=feature_weights
        )

        os.makedirs(os.path.join(output_data_path, pipeline_step), exist_ok=True)
        with open(os.path.join(output_data_path, pipeline_step, 'fused_features.json'), 'w') as outfile:
            json.dump(fused_features, outfile)

        click.echo("[feature_engineering] Processed features for clustering.")


if __name__ == '__main__':
    run()

    


    
