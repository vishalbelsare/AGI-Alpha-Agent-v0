
async function refresh(){
  const res=await fetch('/api/logs?limit=50');const data=await res.json();
  const tbody=document.querySelector('#tbl tbody');tbody.innerHTML='';
  data.forEach(r=>{
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${r.ts}</td><td>${r.agent}</td><td>${r.kind}</td><td><pre>${JSON.stringify(r.data,null,2)}</pre></td>`;
    tbody.appendChild(tr);
  });
}
setInterval(refresh,5000);refresh();
