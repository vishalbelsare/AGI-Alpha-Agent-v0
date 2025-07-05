// SPDX-License-Identifier: Apache-2.0
pragma circom 2.0.0;

// Simple circuit that hashes a pair of scores.
// This placeholder circuit demonstrates how aggregated
// proofs could be built for evaluation transcripts.

include "circomlib/sha256.circom";

template ScoreHash() {
    signal input scores[2];
    signal output hash[2];

    component hasher = Sha256(64);
    for (var i = 0; i < 2; i++) {
        hasher.in[i] <== scores[i];
    }
    for (var i = 2; i < 64; i++) {
        hasher.in[i] <== 0;
    }
    hash[0] <== hasher.out[0];
    hash[1] <== hasher.out[1];
}

component main = ScoreHash();
