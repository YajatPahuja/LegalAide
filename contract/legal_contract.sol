// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract LegalContractRegistry {
    struct ContractRecord {
        string contractHash;
        bool isCompliant;
        string ipfsLink;
    }

    mapping(string => ContractRecord) public contracts;

    event ContractStored(string contractHash, bool isCompliant, string ipfsLink);

    function storeContract(string memory _hash, bool _isCompliant, string memory _ipfsLink) public {
        require(bytes(contracts[_hash].contractHash).length == 0, "Already exists");
        contracts[_hash] = ContractRecord(_hash, _isCompliant, _ipfsLink);
        emit ContractStored(_hash, _isCompliant, _ipfsLink);
    }

    function getContract(string memory _hash) public view returns (string memory, bool, string memory) {
    require(bytes(contracts[_hash].contractHash).length != 0, "Not found");
    ContractRecord memory contractRecord = contracts[_hash];
    return (contractRecord.contractHash, contractRecord.isCompliant, contractRecord.ipfsLink);
}
}
