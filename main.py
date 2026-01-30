import os

from clustering_pipeline.feature_extraction import feature_extraction

# paths to images
image_paths = [os.path.join('data/Check', filename) 
               for filename in os.listdir('data/Check') 
               if filename.endswith('.jpg')]

# first step: extract features from images
img_features = {image_path: feature_extraction.extract_features_from_image(image_path) for image_path in image_paths}

# print extracted features for verification
for img, features in img_features.items():
    print(f"Extracted features for {img}: {features}")