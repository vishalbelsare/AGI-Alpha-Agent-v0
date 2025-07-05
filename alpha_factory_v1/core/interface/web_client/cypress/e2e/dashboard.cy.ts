// SPDX-License-Identifier: Apache-2.0
describe('dashboard', () => {
  it('loads lineage tree', () => {
    cy.on('window:before:load', (win) => {
      cy.spy(win.console, 'error').as('consoleError');
    });
    cy.visit('/');
    cy.get('#lineage-tree g.slice').should('have.length.gte', 3);
    cy.get('@consoleError').should('not.be.called');
  });

  it('shows annotation on hover', () => {
    cy.visit('/');
    cy.get('#lineage-tree g.slice').first().trigger('mouseover');
    cy.get('#lineage-tree .hovertext').should('be.visible');
  });
});
