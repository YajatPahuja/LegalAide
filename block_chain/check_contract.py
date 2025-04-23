from web3 import Web3

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))  # Ensure it matches your Ganache RPC URL
tx_hash = "4846723314fef7c33f93eb67cc18cefe017cbd1532a37e22ca897b689d301806"

# Get transaction receipt
receipt = w3.eth.get_transaction_receipt(tx_hash)

if receipt:
    print("✅ Transaction is mined!")
else:
    print("⏳ Transaction is still pending...")


contract_address = "0x1968D16fD0D858d919d14D0d120B81D60fec0B50"
code = w3.eth.get_code(contract_address)
print(f"Contract code at {contract_address}: {code.hex()}")  # Should not return 0x

