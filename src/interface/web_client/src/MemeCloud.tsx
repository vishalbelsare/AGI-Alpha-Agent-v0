// SPDX-License-Identifier: Apache-2.0
import React from 'react';

interface Props {
  counts: Record<string, number>;
}

export default function MemeCloud({ counts }: Props) {
  const entries = Object.entries(counts);
  if (!entries.length) return <div id="meme-cloud" />;
  const max = Math.max(...entries.map(([, c]) => c));
  return (
    <div id="meme-cloud" style={{ lineHeight: '2em' }}>
      {entries
        .sort((a, b) => b[1] - a[1])
        .map(([m, c]) => {
          const size = 0.5 + (c / max) * 1.5;
          return (
            <span key={m} style={{ fontSize: `${size}em`, marginRight: '0.5em' }}>
              {m}
            </span>
          );
        })}
    </div>
  );
}
