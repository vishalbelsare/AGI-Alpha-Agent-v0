// SPDX-License-Identifier: Apache-2.0
describe('adaptive toggle', () => {
  it('toggles body attribute', () => {
    cy.visit('/');
    cy.get('#adaptive').check();
    cy.get('body').should('have.attr', 'data-adaptive', 'true');
    cy.get('#adaptive').uncheck();
    cy.get('body').should('not.have.attr', 'data-adaptive');
  });
});
