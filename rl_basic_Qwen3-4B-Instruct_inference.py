"""
This script downloads the weights of the trained Qwen3-4B-Instruct-2507 model
after running the rl_basic_Qwen3-4B-Instruct-2507.py script from Tinker.
"""
import tinker
import requests

# Your saved sampler path from the training output
sampler_path = "tinker://f0297fe1-2fac-5f60-b2ed-8f3ff82c5dc4:train:0/sampler_weights/final"

# Create service client
service_client = tinker.ServiceClient()

# Download the checkpoint weights to a file
rest_client = service_client.create_rest_client()
# get_checkpoint_archive_url_from_tinker_path returns a CheckpointArchiveUrlResponse with a .url property
future = rest_client.get_checkpoint_archive_url_from_tinker_path(sampler_path)
url_response = future.result()

# Download from the URL
print(f"Downloading checkpoint from: {url_response.url}")
response = requests.get(url_response.url, stream=True)
response.raise_for_status()

with open("model-checkpoint.tar.gz", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)

print("Checkpoint downloaded to model-checkpoint.tar.gz")
