// SPDX-License-Identifier: Apache-2.0
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Archive from './pages/Archive';
import { IntlProvider, useI18n } from './IntlContext';

function Nav() {
  const { lang, setLang, t } = useI18n();
  return (
    <nav>
      <Link to="/">{t('nav.dashboard')}</Link> |{' '}
      <Link to="/archive">{t('nav.archive')}</Link>{' '}
      <select
        aria-label={t('aria.language')}
        value={lang}
        onChange={(e) => setLang(e.target.value)}
      >
        <option value="en">English</option>
        <option value="fr">Français</option>
      </select>
    </nav>
  );
}

function App() {
  return (
    <BrowserRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/archive" element={<Archive />} />
      </Routes>
    </BrowserRouter>
  );
}

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <IntlProvider>
    <App />
  </IntlProvider>,
);

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/sw.js').catch((err) => {
      console.error('SW registration failed', err);
    });
  });
}

window.addEventListener('beforeinstallprompt', (e) => {
  e.preventDefault();
  const promptEvent = e as any;
  if (window.confirm('Add to Home Screen?')) {
    promptEvent.prompt();
  }
});
