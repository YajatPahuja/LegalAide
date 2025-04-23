import json
from web3 import Web3
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connect to Ganache (update if using another provider)
w3 = Web3(Web3.HTTPProvider(os.getenv('GANACHE_URL')))

# Load contract ABI
with open('contract/contract_abi.json', 'r') as f:
    contract_abi = json.load(f)

# Get contract address from environment
contract_address = os.getenv('CONTRACT_ADDRESS')

# Load contract
contract = w3.eth.contract(address=contract_address, abi=contract_abi)

# Wallet details from environment
private_key = os.getenv('PRIVATE_KEY')
account = w3.eth.account.from_key(private_key)

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
contract_hash = "YajatPahuja"  # Replace with actual contract hash
is_compliant = True  # Compliance result from your ML model
ipfs_link = "https://gateway.pinata.cloud/ipfs/QmSi7k2gb8eB25yLgUgDx1LfTFGYq7ZwsBrabjt1MHDhKf"  # Replace with actual IPFS link

store_contract_data(contract_hash, is_compliant, ipfs_link)
