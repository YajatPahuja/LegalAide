from web3 import Web3
import json

# Connect to local Ganache blockchain
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
assert w3.is_connected(), "Not connected to blockchain"

# Load ABI and Bytecode
with open("contract_abi.json", "r") as f:
    abi = json.load(f)

with open("contract_bytecode.txt", "r") as f:
    bytecode = f.read()

# Your test account from Ganache
acct = w3.eth.account.from_key("0x1382e4d6d8fdff658d7ebdb906a98bdcb66dfdbe8921856daa28c2514bd95683")  # Replace this!
w3.eth.default_account = acct.address

# Deploy contract
Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
tx_hash = Contract.constructor().transact()
tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

print("✅ Contract deployed at:", tx_receipt.contractAddress)

# Save deployed address
with open("contract_address.txt", "w") as f:
    f.write(tx_receipt.contractAddress)
