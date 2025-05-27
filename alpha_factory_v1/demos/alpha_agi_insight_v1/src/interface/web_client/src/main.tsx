// SPDX-License-Identifier: Apache-2.0
import React, { useEffect } from 'react';
import ReactDOM from 'react-dom/client';

function App() {
  useEffect(() => {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/progress`);
    ws.onmessage = (ev) => console.log(ev.data);
    return () => ws.close();
  }, []);
  return <div>α‑AGI Insight Demo</div>;
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
