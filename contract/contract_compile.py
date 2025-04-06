from solcx import compile_source, install_solc
install_solc("0.8.20")  # Install compiler version

with open("legal_contract.sol", "r") as file:
    source_code = file.read()

compiled = compile_source(
    source_code,
    output_values=["abi", "bin"],
    solc_version="0.8.20"
)

contract_id, contract_interface = compiled.popitem()
abi = contract_interface["abi"]
bytecode = contract_interface["bin"]

# Save ABI and Bytecode
import json
with open("contract_abi.json", "w") as f:
    json.dump(abi, f)

with open("contract_bytecode.txt", "w") as f:
    f.write(bytecode)

print("✅ Contract compiled!")
