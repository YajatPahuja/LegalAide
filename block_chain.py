################ LOAD THE MODEL ##############
import pickle
import numpy as np
import os
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()


# Load your trained model
with open("compliance_model.pkl", "rb") as f:
    model = pickle.load(f)

def check_compliance(features):
    """Check if a document is compliant."""
    prediction = model.predict([features])
    return prediction[0]  # 1 for compliant, 0 for non-compliant


###########UPLOAD TO IPFS##################33
import requests

# Replace with your Pinata API credentials
PINATA_API_KEY = os.getenv('PINATA_API_KEY')
PINATA_SECRET_API_KEY = os.getenv('PINATA_SECRET_API_KEY')

def upload_to_pinata(file_path, api_key, api_secret):
    """
    Uploads a file to Pinata (IPFS) and returns the IPFS hash.

    :param file_path: Path to the file to upload.
    :param api_key: Pinata API Key.
    :param api_secret: Pinata Secret API Key.
    :return: IPFS hash if successful, None otherwise.
    """
    url = "https://api.pinata.cloud/pinning/pinFileToIPFS"
    headers = {
        "pinata_api_key": api_key,
        "pinata_secret_api_key": api_secret
    }

    try:
        with open(file_path, "rb") as f:
            response = requests.post(url, files={"file": f}, headers=headers)
        
        if response.status_code == 200:
            ipfs_hash = response.json()["IpfsHash"]
            print(f"✅ File uploaded: https://gateway.pinata.cloud/ipfs/{ipfs_hash}")
            return ipfs_hash
        else:
            print("❌ Upload failed:", response.text)
            return None
    except Exception as e:
        print(f"⚠ Error: {e}")
        return None
    




