// SPDX-License-Identifier: Apache-2.0
import React from 'react';

interface Props {
  open: boolean;
  onClose: () => void;
  docUrl: string;
}

export default function RationaleModal({ open, onClose, docUrl }: Props) {
  if (!open) return null;
  return (
    <div
      className="modal-overlay"
      style={{ position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)' }}
    >
      <div
        className="modal-content"
        style={{ background: '#fff', margin: '10% auto', padding: 20, maxWidth: 400 }}
      >
        <p>
          See <a href={docUrl} target="_blank" rel="noopener noreferrer">documentation</a>{' '}
          for the rationale behind these scores.
        </p>
        <button type="button" onClick={onClose}>Close</button>
      </div>
    </div>
  );
}
