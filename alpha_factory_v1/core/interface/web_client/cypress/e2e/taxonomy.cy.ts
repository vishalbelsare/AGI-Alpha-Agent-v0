// SPDX-License-Identifier: Apache-2.0

describe('taxonomy persistence', () => {
  it('restores taxonomy tree after reload', () => {
    cy.visit('/', {
      onBeforeLoad(win) {
        const req = win.indexedDB.open('sectorTaxonomy', 1);
        req.onupgradeneeded = () => {
          req.result.createObjectStore('nodes');
        };
        req.onsuccess = () => {
          const tx = req.result.transaction('nodes', 'readwrite');
          tx.objectStore('nodes').put({ id: 'foo', parent: null }, 'foo');
          tx.oncomplete = () => {};
        };
      },
    });
    cy.get('#taxonomy-tree button').contains('foo');
    cy.reload();
    cy.get('#taxonomy-tree button').contains('foo');
  });
});
