// SPDX-License-Identifier: Apache-2.0
import React, { createContext, ReactNode, useState, useEffect, useContext } from 'react';
import en from '../i18n/en.json';
import fr from '../i18n/fr.json';
import es from '../i18n/es.json';

export type Messages = Record<string, string>;

const dictionaries: Record<string, Messages> = { en, fr, es };

interface IntlValue {
  lang: string;
  messages: Messages;
  setLang: (lang: string) => void;
}

const fallback = 'en';

export const IntlContext = createContext<IntlValue>({
  lang: fallback,
  messages: dictionaries[fallback],
  setLang: () => {},
});

export function IntlProvider({ children }: { children: ReactNode }) {
  const browserLang = typeof navigator !== 'undefined' ? navigator.language : fallback;
  const lower = browserLang.toLowerCase();
  const defaultLang = lower.startsWith('fr') ? 'fr' : lower.startsWith('es') ? 'es' : 'en';
  const [lang, setLang] = useState<string>(defaultLang);
  const [messages, setMessages] = useState<Messages>(dictionaries[defaultLang]);

  useEffect(() => {
    setMessages(dictionaries[lang] || dictionaries[fallback]);
  }, [lang]);

  return (
    <IntlContext.Provider value={{ lang, messages, setLang }}>
      {children}
    </IntlContext.Provider>
  );
}

export function useI18n() {
  const { lang, messages, setLang } = useContext(IntlContext);
  const t = (key: string) => messages[key] ?? key;
  return { lang, setLang, t };
}
