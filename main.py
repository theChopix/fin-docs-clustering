import click
import json
import os
import yaml

from clustering_pipeline.download_dataset import download_dataset
from clustering_pipeline.feature_extraction import feature_extraction
from clustering_pipeline.feature_engineering.feature_engineering import process_images_features
from clustering_pipeline.docs_clustering import docs_clustering


@click.command()
@click.option('--input_data_path', default='data', help='Path to the input data directory.')
@click.option('--config', default='configs/config.yaml', help='Path to the experiment configuration file (with feature weights).')
@click.option('--output_data_path', default='results', help='Path to the experiment results directory.')
def run(input_data_path: str, config: str, output_data_path: str):

    # load feature weights from config file
    with open(config, 'r') as config_file:
        config_data = yaml.safe_load(config_file)
        feature_weights = config_data.get('feature_weights', {})

    # -- data download step --
    # if the input data path does not exist or is empty, download the dataset
    if not os.path.exists(input_data_path) or not os.listdir(input_data_path):
        click.echo("[data_download] Downloading dataset...")
        download_dataset(target_dir=input_data_path)
        click.echo("[data_download] Dataset downloaded.")
    
    # -- feature extraction step --
    pipeline_step = 'feature_extraction'
    # if the output data path for feature extraction does not exist or is empty, run feature extraction
    if not os.path.exists(output_data_path) or not os.path.exists(os.path.join(output_data_path, pipeline_step)):
        click.echo(f"[{pipeline_step}] Extracting features from images...")
        os.makedirs(os.path.join(output_data_path, pipeline_step), exist_ok=True)
        for file in os.listdir(input_data_path):
            features = feature_extraction.extract_features_from_image(
                image_path=os.path.join(input_data_path, file)
            )
            output_file_path = os.path.join(output_data_path, pipeline_step, f"{os.path.splitext(file)[0]}_features.json")
            with open(output_file_path, 'w') as outfile:
                json.dump(features, outfile)
        click.echo(f"[{pipeline_step}] Extracted features from images.")

    # -- feature engineering step --
    pipeline_step = 'feature_engineering'
    # if the output data path for feature engineering does not exist or is empty, run feature engineering
    if not os.path.exists(output_data_path) or not os.path.exists(os.path.join(output_data_path, pipeline_step)):
        click.echo(f"[{pipeline_step}] Processing features for clustering...")
        images_features = {}
        for filename in os.listdir(os.path.join(output_data_path, 'feature_extraction')):
            if filename.endswith('_features.json'):
                image_id = os.path.splitext(filename)[0].replace('_features', '')
                with open(os.path.join(output_data_path, 'feature_extraction', filename), 'r') as infile:
                    features = json.load(infile)
                    images_features[image_id] = features

        fused_features = process_images_features(
            images_features=images_features,
            feature_weights=feature_weights
        )
        # backup processed features
        os.makedirs(os.path.join(output_data_path, pipeline_step), exist_ok=True)
        for image_id, features in fused_features.items():
            output_file_path = os.path.join(output_data_path, pipeline_step, f"{image_id}_processed_features.json")
            with open(output_file_path, 'w') as outfile:
                json.dump(features, outfile)

        click.echo(f"[{pipeline_step}] Processed features for clustering.")

    # -- document clustering step --
    pipeline_step = 'document_clustering'
    # if the output data path for document clustering does not exist or is empty, run document clustering
    if not os.path.exists(output_data_path) or not os.path.exists(os.path.join(output_data_path, pipeline_step)):
        click.echo(f"[{pipeline_step}] Clustering documents...")

        # load processed features from feature_engineering step
        fused_features = {}
        feature_engineering_dir = os.path.join(output_data_path, 'feature_engineering')
        for filename in os.listdir(feature_engineering_dir):
            if filename.endswith('_processed_features.json'):
                image_id = os.path.splitext(filename)[0].replace('_processed_features', '')
                with open(os.path.join(feature_engineering_dir, filename), 'r') as infile:
                    features = json.load(infile)
                    fused_features[image_id] = features

        clusters = docs_clustering.cluster_documents(fused_features)

        # backup clustering results
        os.makedirs(os.path.join(output_data_path, pipeline_step), exist_ok=True)
        for doc_name, cluster_info in clusters.items():
            output_file_path = os.path.join(output_data_path, pipeline_step, f"{doc_name}_cluster.json")
            with open(output_file_path, 'w') as outfile:
                json.dump(cluster_info, outfile)

        click.echo(f"[{pipeline_step}] Documents clustered.")


if __name__ == '__main__':
    run()

    


    
