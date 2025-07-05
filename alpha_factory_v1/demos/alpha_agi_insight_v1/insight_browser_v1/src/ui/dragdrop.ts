// SPDX-License-Identifier: Apache-2.0
export function initDragDrop(el: HTMLElement, onDrop: (data: string | ArrayBuffer | null) => void): void {
  function over(ev: DragEvent): void {
    ev.preventDefault();
    el.classList.add('drag');
  }
  function leave(): void {
    el.classList.remove('drag');
  }
  function drop(ev: DragEvent): void {
    ev.preventDefault();
    el.classList.remove('drag');
    const dt = ev.dataTransfer;
    const file = dt && dt.files && dt.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => onDrop(reader.result);
    reader.readAsText(file);
  }
  el.addEventListener('dragover', over);
  el.addEventListener('dragleave', leave);
  el.addEventListener('drop', drop);
}
