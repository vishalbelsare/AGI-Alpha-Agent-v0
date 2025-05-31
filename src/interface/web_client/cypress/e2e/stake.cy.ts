// SPDX-License-Identifier: Apache-2.0
describe('staking flow', () => {
  it('stakes tokens and retrieves proof', () => {
    cy.intercept('POST', '/stake', { statusCode: 200 }).as('stake');
    cy.intercept('POST', '/dispatch', { statusCode: 200 }).as('dispatch');
    cy.intercept('GET', /\/proof\/.+/, { cid: 'Qm123', proof: 'proof' }).as('proof');
    cy.visit('/staking/index.html');
    cy.get('input').type('1');
    cy.contains('Stake').click();
    cy.wait('@stake');
    cy.wait('@dispatch');
    cy.contains('Load Proof').click();
    cy.wait('@proof');
    cy.contains('Qm123');
    cy.contains('proof');
  });
});
