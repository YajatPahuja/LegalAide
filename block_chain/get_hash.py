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

def get_contract_data(contract_hash):
    """
    Retrieves stored contract details (hash, compliance, IPFS link) from blockchain.
    """
    try:
        contract_data = contract.functions.getContract(contract_hash).call()
        print(f"📄 Contract Hash: {contract_data[0]}")
        print(f"✔ Compliant: {contract_data[1]}")
        print(f"🔗 IPFS Link: {contract_data[2]}")
        return contract_data
    except Exception as e:
        print(f"⚠ Error: {e}")
        return None

# Example usage
retrieved_data = get_contract_data("Yajat")
