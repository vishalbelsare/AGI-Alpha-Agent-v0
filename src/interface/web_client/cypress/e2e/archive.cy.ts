// SPDX-License-Identifier: Apache-2.0
describe('archive page', () => {
  it('renders diff when selecting an agent', () => {
    cy.visit('/archive');
    cy.get('.agent-row button').first().click();
    cy.get('pre.diff').should('be.visible');
  });

  it('shows backlink to parent', () => {
    cy.visit('/archive');
    cy.get('.agent-row button').first().click();
    cy.get('a.parent-link').should('have.attr', 'href');
  });
});
