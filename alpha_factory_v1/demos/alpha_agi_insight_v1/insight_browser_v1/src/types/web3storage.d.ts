// SPDX-License-Identifier: Apache-2.0
declare module '*bundle.esm.min.js' {
  export class Web3Storage {
    constructor(opts: { token: string });
    put(files: File[]): Promise<string>;
  }
}
