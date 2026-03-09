import os
from pathlib import Path
from google.cloud import storage

BUCKET = "data-storage-0"

def _client_bucket():
    client = storage.Client()
    return client, client.bucket(BUCKET)

def upload_file(local_path: str, gcs_path: str):
    _, bucket = _client_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_filename(local_path)
    print(f"Uploaded: {local_path} -> gs://{BUCKET}/{gcs_path}")

def download_file(gcs_path: str, local_path: str):
    _, bucket = _client_bucket()
    blob = bucket.blob(gcs_path)
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    blob.download_to_filename(local_path)
    print(f"Downloaded: gs://{BUCKET}/{gcs_path} -> {local_path}")

def upload_dir(local_dir: str, gcs_prefix: str, skip_hidden: bool = True):
    """
    Upload a local directory recursively to gs://BUCKET/gcs_prefix/

    Example:
      upload_dir("data", "datasets/mydata")
      -> data/a.txt -> gs://data-storage-0/datasets/mydata/a.txt
    """
    local_dir = os.path.abspath(local_dir)
    gcs_prefix = gcs_prefix.strip("/")

    _, bucket = _client_bucket()

    for root, dirs, files in os.walk(local_dir):
        # optionally skip hidden dirs
        if skip_hidden:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            files = [f for f in files if not f.startswith(".")]

        for fname in files:
            local_path = os.path.join(root, fname)
            rel_path = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            gcs_path = f"{gcs_prefix}/{rel_path}" if gcs_prefix else rel_path

            blob = bucket.blob(gcs_path)
            blob.upload_from_filename(local_path)
            print(f"Uploaded: {local_path} -> gs://{BUCKET}/{gcs_path}")

def download_dir(gcs_prefix: str, local_dir: str):
    """
    Download all objects under gs://BUCKET/gcs_prefix/ into local_dir.

    Example:
      download_dir("datasets/mydata", "./mydata_local")
    """
    gcs_prefix = gcs_prefix.strip("/")
    local_dir = os.path.abspath(local_dir)

    client, bucket = _client_bucket()

    # List all objects under prefix
    blobs = client.list_blobs(BUCKET, prefix=(gcs_prefix + "/") if gcs_prefix else "")

    for blob in blobs:
        # Skip "directory marker" objects if any
        if blob.name.endswith("/"):
            continue

        # remove prefix from blob.name to get relative path
        if gcs_prefix:
            rel_path = blob.name[len(gcs_prefix) + 1 :]  # +1 for '/'
        else:
            rel_path = blob.name

        local_path = os.path.join(local_dir, rel_path.replace("/", os.sep))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        blob.download_to_filename(local_path)
        print(f"Downloaded: gs://{BUCKET}/{blob.name} -> {local_path}")

if __name__ == "__main__":
    # 上传整个目录
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25", "BEAT2/beat_english_v2.0.0/smplxflame_25")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/smplxflame_25_mirror", "BEAT2/beat_english_v2.0.0/smplxflame_25_mirror")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/TOKENS_AGENT_25", "BEAT2/beat_english_v2.0.0/TOKENS_AGENT_25")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/textgrid", "BEAT2/beat_english_v2.0.0/textgrid")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/wave16k", "BEAT2/beat_english_v2.0.0/wave16k")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/weights", "BEAT2/beat_english_v2.0.0/weights")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/train_test_split.csv", "BEAT2/beat_english_v2.0.0")
    # upload_dir("/simurgh2/datasets/BEAT2/beat_english_v2.0.0/sem", "BEAT2/beat_english_v2.0.0/sem")
    # upload_dir("/scr/juze/datasets/BEAT2/beat_english_v2.0.0/processed_beat2_dataset_tokenized_mot_encode_position_body_v3_train", "BEAT2/beat_english_v2.0.0/processed_beat2_dataset_tokenized_mot_encode_position_body_v3_train")
    upload_dir("/simurgh2/datasets/AMASS/amass_parts_25", "AMASS/amass_parts_25")

    # # 下载整个目录
    # download_dir("datasets/juze/data", "./data_downloaded")
