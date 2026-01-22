#!/bin/bash
urle () { 
    [ "${1}" ] || return 1
    local LANG=C i=0 x
    while [ $i -lt ${#1} ]; do
        x=$(printf '%s' "${1}" | cut -c$((i+1)))
        case "${x}" in
            [a-zA-Z0-9.~-]) printf '%s' "${x}" ;;
            *) printf '%%%02X' "'${x}" ;;
        esac
        i=$((i+1))
    done
    printf '\n'
}

# Fetch SMPLH data
echo -e "\nBefore you continue, you must register at https://smpl-x.is.tue.mpg.de/ and agree to the SMPLX license terms."
read -p "Username (SMPLH):" username
read -p "Password (SMPLH):" password
username=$(urle $username)
password=$(urle $password)

# Create directories for amass


echo -e "\n Please enter the path to the AMASS dataset."
read -p "AMASS path:" amass_path
mkdir -p $amass_path/AMASS_original_smplh

echo "Downloading amass smplh version..."
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/ACCAD.tar.bz2' -O "$amass_path/AMASS_original_smplh/ACCAD.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/BMLhandball.tar.bz2' -O "$amass_path/AMASS_original_smplh/BMLhandball.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/BMLmovi.tar.bz2' -O "$amass_path/AMASS_original_smplh/BMLmovi.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/BMLrub.tar.bz2' -O "$amass_path/AMASS_original_smplh/BMLrub.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/CMU.tar.bz2' -O "$amass_path/AMASS_original_smplh/CMU.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/DanceDB.tar.bz2' -O "$amass_path/AMASS_original_smplh/DanceDB.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/DFaust.tar.bz2' -O "$amass_path/AMASS_original_smplh/DFaust.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/EKUT.tar.bz2' -O "$amass_path/AMASS_original_smplh/EKUT.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/EyesJapanDataset.tar.bz2' -O "$amass_path/AMASS_original_smplh/EyesJapanDataset.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/GRAB.tar.bz2' -O "$amass_path/AMASS_original_smplh/GRAB.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/HDM05.tar.bz2' -O "$amass_path/AMASS_original_smplh/HDM05.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/HUMAN4D.tar.bz2' -O "$amass_path/AMASS_original_smplh/HUMAN4D.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/HumanEva.tar.bz2' -O "$amass_path/AMASS_original_smplh/HumanEva.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/KIT.tar.bz2' -O "$amass_path/AMASS_original_smplh/KIT.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/MoSh.tar.bz2' -O "$amass_path/AMASS_original_smplh/MoSh.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/PosePrior.tar.bz2' -O "$amass_path/AMASS_original_smplh/PosePrior.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/SFU.tar.bz2' -O "$amass_path/AMASS_original_smplh/SFU.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/SOMA.tar.bz2' -O "$amass_path/AMASS_original_smplh/SOMA.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/SSM.tar.bz2' -O "$amass_path/AMASS_original_smplh/SSM.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/TCDHands.tar.bz2' -O "$amass_path/AMASS_original_smplh/TCDHands.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/TotalCapture.tar.bz2' -O "$amass_path/AMASS_original_smplh/TotalCapture.tar.bz2" --no-check-certificate --continue
wget --post-data "username=$username&password=$password" 'https://download.is.tue.mpg.de/download.php?domain=amass&resume=1&sfile=amass_per_dataset/smplh/gender_specific/mosh_results/Transitions.tar.bz2' -O "$amass_path/AMASS_original_smplh/Transitions.tar.bz2" --no-check-certificate --continue

echo "Successfully downloaded amass dataset!"


echo "Extracting all downloaded datasets..."
cd "$amass_path/AMASS_original_smplh"

# Extract all tar.bz2 files
for file in *.tar.bz2; do
    if [ -f "$file" ]; then
        echo "Extracting $file..."
        tar -xjf "$file"
    fi
done

echo "Extraction completed!"
echo "Cleaning up compressed files..."

# Optional: Remove the compressed files after extraction
read -p "Do you want to remove the compressed .tar.bz2 files? (y/n): " remove_compressed
if [[ $remove_compressed == "y" || $remove_compressed == "Y" ]]; then
    rm *.tar.bz2
    echo "Compressed files removed."
else
    echo "Compressed files kept."
fi

cd - > /dev/null  # Return to original directory