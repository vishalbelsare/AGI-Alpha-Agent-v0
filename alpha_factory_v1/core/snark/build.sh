#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Build the Circom circuit for score aggregation.
# Requires circom and snarkjs installed locally.
set -euo pipefail

CIRCUIT_DIR="$(dirname "$0")"
cd "$CIRCUIT_DIR"

circom score.circom --r1cs --wasm
snarkjs groth16 setup score.r1cs pot12_final.ptau score_0000.zkey
snarkjs zkey contribute score_0000.zkey score_final.zkey --name "Alpha" -v -e="random"
snarkjs zkey export verificationkey score_final.zkey verification_key.json
