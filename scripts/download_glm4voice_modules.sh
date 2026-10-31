#!/bin/bash
# Download cosyvoice and speech_tokenizer from the GLM-4-Voice project, and the
# glm-4-voice-decoder weights, into speech_related/.
# Run from the repo root. Safe to re-run.
set -euo pipefail

mkdir -p speech_related
cd speech_related

# Sparse-checkout only the two directories we need
rm -rf glm4voice_temp
git clone --no-checkout https://github.com/zai-org/GLM-4-Voice.git glm4voice_temp
git -C glm4voice_temp sparse-checkout init --cone
git -C glm4voice_temp sparse-checkout set cosyvoice speech_tokenizer
git -C glm4voice_temp checkout main

# speech_related/cosyvoice/ already exists when the repo was cloned with
# --recurse-submodules (it holds the third_party/Matcha-TTS submodule), so merge
# into it instead of moving the directory over it
mkdir -p cosyvoice speech_tokenizer
cp -r glm4voice_temp/cosyvoice/. cosyvoice/
cp -r glm4voice_temp/speech_tokenizer/. speech_tokenizer/
rm -rf glm4voice_temp

# glm-4-voice-decoder from Hugging Face (requires git-lfs)
git lfs install
if [ ! -f glm-4-voice-decoder/flow.pt ]; then
    rm -rf glm-4-voice-decoder
    git clone https://huggingface.co/THUDM/glm-4-voice-decoder
fi

for f in cosyvoice/cli cosyvoice/flow cosyvoice/hifigan speech_tokenizer/modeling_whisper.py \
         glm-4-voice-decoder/flow.pt glm-4-voice-decoder/hift.pt; do
    [ -e "$f" ] || { echo "Missing speech_related/$f" >&2; exit 1; }
done
echo "✅ Downloaded cosyvoice, speech_tokenizer and glm-4-voice-decoder to speech_related/"
