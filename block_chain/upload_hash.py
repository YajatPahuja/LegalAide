import json
from web3 import Web3
import os
from dotenv import load_dotenv
import requests

load_dotenv()

w3 = Web3(Web3.HTTPProvider(os.getenv('GANACHE_URL')))

with open('contract/contract_abi.json', 'r') as f:
    contract_abi = json.load(f)

contract_address = os.getenv('CONTRACT_ADDRESS')


contract = w3.eth.contract(address=contract_address, abi=contract_abi)

private_key = os.getenv('PRIVATE_KEY')
account = w3.eth.account.from_key(private_key)

PINATA_URL = "https://api.pinata.cloud/pinning/pinFileToIPFS"
PINATA_API_KEY = os.getenv('PINATA_API_KEY')
PINATA_API_SECRET = os.getenv('PINATA_SECRET_API_KEY')

print(f"API Key: {os.getenv('PINATA_API_KEY')}")
print(f"API Secret: {os.getenv('PINATA_SECRET_API_KEY')}")


headers = {
    'pinata_api_key': PINATA_API_KEY,
    'pinata_secret_api_key': PINATA_API_SECRET
}

def upload_file_to_ipfs(file_path):
    """
    Uploads a file to IPFS using Pinata.
    Returns the IPFS URL for the uploaded file.
    """
    with open(file_path, 'rb') as file:
        # Upload the file to IPFS
        response = requests.post(PINATA_URL, headers=headers, files={'file': file})

    if response.status_code == 200:
        ipfs_hash = response.json()['IpfsHash']
        ipfs_link = f"https://gateway.pinata.cloud/ipfs/{ipfs_hash}"
        print(f"✅ File uploaded to IPFS. IPFS Link: {ipfs_link}")
        return ipfs_link
    else:
        print(f"❌ Error uploading file to IPFS: {response.text}")
        return None

def store_contract_data(contract_hash, is_compliant, ipfs_link):
    """
    Stores contract details (hash, compliance status, IPFS link) on the blockchain.
    """
    nonce = w3.eth.get_transaction_count(account.address)

    # Create transaction
    txn = contract.functions.storeContract(contract_hash, is_compliant, ipfs_link).build_transaction({
        "from": account.address,
        "gas": 200000,
        "gasPrice": w3.to_wei("10", "gwei"),
        "nonce": nonce
    })

    # Sign and send transaction
    signed_txn = w3.eth.account.sign_transaction(txn, private_key)
    tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

    print(f"✅ Transaction sent! Tx Hash: {tx_hash.hex()}")
    return tx_hash.hex()

# Example usage
file_path = "data/mydocument.pdf"  # Replace with the path to your contract file
contract_hash = "Yajat"  # Replace with actual contract hash
is_compliant = True  # Compliance result from your ML model

# Upload file to IPFS
ipfs_link = upload_file_to_ipfs(file_path)

if ipfs_link:
    store_contract_data(contract_hash, is_compliant, ipfs_link)


