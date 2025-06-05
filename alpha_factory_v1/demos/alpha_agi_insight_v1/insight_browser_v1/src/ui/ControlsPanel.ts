// @ts-nocheck
// SPDX-License-Identifier: Apache-2.0
import {t,setLanguage,currentLanguage} from './i18n.ts';
const MAX_VAL = 500;
export function initControls(
  params: any,
  onChange: (p: any, info: any) => void,
): { setValues: (p: any) => void; pauseBtn: HTMLButtonElement; exportBtn: HTMLButtonElement; dropZone: HTMLElement } {
  const root=document.getElementById('controls');
  root.replaceChildren();
  function addLabel(text: string, input: HTMLElement): void {
    const label = document.createElement('label');
    label.appendChild(document.createTextNode(text + ' '));
    label.appendChild(input);
    root.appendChild(label);
  }

  const seedInput = document.createElement('input');
  seedInput.id = 'seed';
  seedInput.type = 'number';
  seedInput.min = '0';
  seedInput.setAttribute('aria-label', t('seed'));
  seedInput.tabIndex = 1;
  addLabel(t('seed'), seedInput);

  const popInput = document.createElement('input');
  popInput.id = 'pop';
  popInput.type = 'number';
  popInput.min = '1';
  popInput.max = String(MAX_VAL);
  popInput.setAttribute('aria-label', t('population'));
  popInput.tabIndex = 2;
  addLabel(t('population'), popInput);

  const genInput = document.createElement('input');
  genInput.id = 'gen';
  genInput.type = 'number';
  genInput.min = '1';
  genInput.max = String(MAX_VAL);
  genInput.setAttribute('aria-label', t('generations'));
  genInput.tabIndex = 3;
  addLabel(t('generations'), genInput);

  function addCheck(id: string, text: string, tab: number): HTMLInputElement {
    const c = document.createElement('input');
    c.id = id;
    c.type = 'checkbox';
    c.setAttribute('aria-label', text);
    c.tabIndex = tab;
    addLabel(text, c);
    return c;
  }

  const gauss = addCheck('gaussian', t('gaussian'), 4);
  const swap = addCheck('swap', t('swap'), 5);
  const jump = addCheck('jump', t('jump'), 6);
  const scramble = addCheck('scramble', t('scramble'), 7);
  const adaptive = addCheck('adaptive', t('adaptive'), 8);

  const pause = document.createElement('button');
  pause.id = 'pause';
  pause.setAttribute('role', 'button');
  pause.setAttribute('aria-label', t('pause'));
  pause.tabIndex = 9;
  pause.textContent = t('pause');
  root.appendChild(pause);

  const exportBtn = document.createElement('button');
  exportBtn.id = 'export';
  exportBtn.setAttribute('role', 'button');
  exportBtn.setAttribute('aria-label', t('export'));
  exportBtn.tabIndex = 10;
  exportBtn.textContent = t('export');
  root.appendChild(exportBtn);

  const drop = document.createElement('div');
  drop.id = 'drop';
  drop.setAttribute('role', 'button');
  drop.setAttribute('aria-label', t('drop'));
  drop.tabIndex = 10;
  drop.textContent = t('drop');
  root.appendChild(drop);

  const langSel = document.createElement('select');
  langSel.id = 'lang';
  langSel.tabIndex = 11;
  const en = document.createElement('option');
  en.value = 'en';
  en.textContent = 'English';
  const fr = document.createElement('option');
  fr.value = 'fr';
  fr.textContent = 'Français';
  const es = document.createElement('option');
  es.value = 'es';
  es.textContent = 'Español';
  langSel.append(en, fr, es);
  root.appendChild(langSel);
  const seed=root.querySelector('#seed'),
        pop=root.querySelector('#pop'),
        gen=root.querySelector('#gen'),
        gauss=root.querySelector('#gaussian'),
        swap=root.querySelector('#swap'),
        jump=root.querySelector('#jump'),
        scramble=root.querySelector('#scramble'),
        adaptive=root.querySelector('#adaptive');
  function update(p){
    seed.value=p.seed;
    pop.value=Math.min(p.pop,MAX_VAL);
    gen.value=Math.min(p.gen,MAX_VAL);
    const set=new Set(p.mutations||[]);
    gauss.checked=set.has('gaussian');
    swap.checked=set.has('swap');
    jump.checked=set.has('jump');
    scramble.checked=set.has('scramble');
    adaptive.checked=p.adaptive||false;
  }
  update(params);
  function emit(){
    const muts=[gauss,swap,jump,scramble].filter(c=>c.checked).map(c=>c.id);
    let popVal=Math.min(+pop.value,MAX_VAL);
    let genVal=Math.min(+gen.value,MAX_VAL);
    const info={popClamped:popVal!==+pop.value,genClamped:genVal!==+gen.value};
    pop.value=popVal;
    gen.value=genVal;
    const p={seed:+seed.value,pop:popVal,gen:genVal,mutations:muts,adaptive:adaptive.checked};
    try{localStorage.setItem('insightParams',JSON.stringify(p));}catch{}
    onChange(p,info);
  }
  seed.addEventListener('change',emit);
  pop.addEventListener('change',emit);
  gen.addEventListener('change',emit);
  gauss.addEventListener('change',emit);
  swap.addEventListener('change',emit);
    jump.addEventListener('change',emit);
    scramble.addEventListener('change',emit);
    adaptive.addEventListener('change',emit);
  const pause=root.querySelector('#pause'),
        exportBtn=root.querySelector('#export'),
        drop=root.querySelector('#drop'),
        langSel=root.querySelector('#lang');
  langSel.value=currentLanguage;
  langSel.addEventListener('change',async()=>{await setLanguage(langSel.value);location.reload();});
  return{setValues:update,pauseBtn:pause,exportBtn,dropZone:drop};
}
