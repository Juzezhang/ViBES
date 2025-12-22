#!/bin/bash
urle () { [[ "${1}" ]] || return 1; local LANG=C i x; for (( i = 0; i < ${#1}; i++ )); do x="${1:i:1}"; [[ "${x}" == [a-zA-Z0-9.~-] ]] && echo -n "${x}" || printf '%%%02X' "'${x}"; done; echo; }

# Fetch SMPLX data
echo -e "\nBefore you continue, you must register at https://smpl-x.is.tue.mpg.de/ and agree to the SMPLX license terms."
read -p "Username (SMPLX):" username
read -p "Password (SMPLX):" password
username=$(urle $username)
password=$(urle $password)

# Create directories for models
mkdir -p model_files
mkdir -p model_files/smplx_models
mkdir -p model_files/smplx_models/smplx
mkdir -p model_files/pretrained_cpt/face

echo -e "\nDownloading SMPLX 2020 model files..."
# Download the neutral SMPLX model
echo "Downloading SMPLX2020 neutral model..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=SMPLX_NEUTRAL_2020.npz' -O './model_files/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz' --no-check-certificate --continue

# Download the SMPLX lockedhead model
echo "Downloading SMPLX lockedhead model..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=smplx&sfile=smplx_lockedhead_20230207.zip' -O './model_files/smplx_models/smplx_lockedhead_20230207.zip' --no-check-certificate --continue

echo "Downloading FLAME 2020 model..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=flame&resume=1&sfile=FLAME2020.zip' -O './model_files/FLAME2020.zip' --no-check-certificate --continue

echo "Downloading region files..."
wget 'https://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_masks.zip' -O './model_files/FLAME_masks.zip' --no-check-certificate --continue

echo "Downloading FLAME static embedding file..."
wget 'https://huggingface.co/camenduru/show/resolve/main/data/flame/flame_static_embedding.pkl' -O './model_files/flame_static_embedding.pkl' --no-check-certificate --continue


# Extract lockedhead model to a temporary directory
echo "Extracting SMPLX lockedhead model..."
mkdir -p ./model_files/temp_smplx
mkdir -p ./model_files/temp_flame
unzip -o './model_files/smplx_models/smplx_lockedhead_20230207.zip' -d './model_files/temp_smplx/'
unzip -o './model_files/FLAME2020.zip' -d './model_files/temp_flame/'
unzip -o './model_files/FLAME_masks.zip' -d './model_files/FLAME2020/'

# Move FLAME files to correct location (remove extra FLAME2020 subdirectory)
echo "Moving FLAME model files to correct location..."
mv ./model_files/temp_flame/FLAME2020/* ./model_files/FLAME2020/

# Find and move all NPZ files to the target directory
echo "Moving model files to the target directory..."
find ./model_files/temp_smplx -name "*.npz" -exec mv {} ./model_files/smplx_models/smplx/ \;

# Check specifically for gendered models
if [ -f "./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_FEMALE.npz" ]; then
    mv ./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_FEMALE.npz ./model_files/smplx_models/smplx/
fi

if [ -f "./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_MALE.npz" ]; then
    mv ./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_MALE.npz ./model_files/smplx_models/smplx/
fi

if [ -f "./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_NEUTRAL.npz" ]; then
    mv ./model_files/temp_smplx/models_lockedhead/smplx/SMPLX_NEUTRAL.npz ./model_files/smplx_models/smplx/
fi

if [ -f "./model_files/FLAME2020/generic_model.pkl" ]; then
    cp ./model_files/FLAME2020/generic_model.pkl ./model_files/FLAME2020/FLAME_NEUTRAL.pkl
fi

if [ -f "./model_files/flame_static_embedding.pkl" ]; then
    cp ./model_files/flame_static_embedding.pkl ./model_files/FLAME2020/
fi

# Clean up
echo "Cleaning up..."
rm -rf './model_files/temp_smplx'
rm -rf './model_files/temp_flame'
rm -rf './model_files/smplx_models/smplx_lockedhead_20230207.zip'
rm -rf './model_files/FLAME2020.zip'

echo -e "\nSMPLX 2020 model setup completed successfully!"
echo "Models are available in: model_files/smplx_models/"

echo -e "\nDownloading face model checkpoint..."

echo "Downloading face model to model_files/pretrained_cpt/face..."
gdown "https://drive.google.com/file/d/14EDHEWdzfOGaae5SyqAXZ5-kGzdyimAQ/view?usp=sharing" --fuzzy -O "./model_files/pretrained_cpt/face/face.ckpt"

echo "Face model download completed!"
echo "Model is available in: model_files/pretrained_cpt/face/"

echo "Download completed! All models are stored in the model_files directory."