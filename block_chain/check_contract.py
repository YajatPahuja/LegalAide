from web3 import Web3

w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))  # Ensure it matches your Ganache RPC URL
tx_hash = "aa55bc17436d8852566448838582f2808c34015ba39b1b2218e0b538f25a937c"

# Get transaction receipt
receipt = w3.eth.get_transaction_receipt(tx_hash)

if receipt:
    print("✅ Transaction is mined!")
else:
    print("⏳ Transaction is still pending...")


contract_address = "0x424E55EccD2F9A2321430B4FbAeC60A1c912Ac6D"
code = w3.eth.get_code(contract_address)
print(code)  # Should not return 0x

