#!/bin/bash
# Download cosyvoice and speech_tokenizer from GLM-4-Voice project to speech_related_test directory
# This script uses git sparse-checkout to download only the required directories

# cd "$(dirname "$0")" || exit 1

# Create test directory if it doesn't exist
mkdir -p speech_related
cd speech_related || exit 1

# Use sparse-checkout to download only the required directories
git clone --no-checkout https://github.com/zai-org/GLM-4-Voice.git glm4voice_temp
cd glm4voice_temp
git sparse-checkout init --cone
git sparse-checkout set cosyvoice speech_tokenizer
git checkout main
cd ..

# Move directories to target location
mv glm4voice_temp/cosyvoice .
mv glm4voice_temp/speech_tokenizer .
rm -rf glm4voice_temp

# Download glm-4-voice-decoder from Hugging Face
# Note: This requires git-lfs to be installed
git lfs install
git clone https://huggingface.co/THUDM/glm-4-voice-decoder

echo "✅ Successfully downloaded cosyvoice, speech_tokenizer, and glm-4-voice-decoder to speech_related_test/ directory"
