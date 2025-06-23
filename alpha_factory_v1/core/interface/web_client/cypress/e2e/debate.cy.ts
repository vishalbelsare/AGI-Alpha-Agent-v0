// SPDX-License-Identifier: Apache-2.0
describe('debate arena', () => {
  it('runs debate and updates ranking', () => {
    cy.visit('/');
    cy.get('#start-debate').click();
    cy.get('#debate-panel li').should('have.length.at.least', 4);
    cy.get('#ranking li').should('have.length.at.least', 1);
  });
});
