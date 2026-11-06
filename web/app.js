const $ = (id) => document.getElementById(id);
let history = [], records = [], busy = false, historyTurns = 3;
const modeNames = {demo: '离线证据摘录 · 未调用大模型', ollama: 'Ollama / Qwen 生成'};
const rewriteNames = {none:'仅当前问题', history_concat:'历史直接拼接', no_history:'无历史，保留原问题', heuristic_demo:'离线启发式规则', heuristic_fallback:'LLM失败后的启发式回退', llm:'大模型重写', llm_structured:'结构化重写'};

async function api(path, body) {
  const response = await fetch(path, body === undefined ? {} : {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}
function element(tag, className, text) {
  const node = document.createElement(tag); if (className) node.className=className; if (text !== undefined) node.textContent=text; return node;
}
function updateMode() {
  const demo = $('mode').value === 'demo';
  $('badge').textContent = demo ? '离线演示' : '本地大模型';
  $('modeHint').textContent = demo ? '使用 BM25 离线展示检索证据和流程。' : '混合检索需要 Ollama 中的 Qwen 和 bge-m3。';
  $('notice').textContent = demo ? '当前是离线演示：回答为资料原文摘录，Query 重写为启发式规则。生成式问答与正式重写实验请选择 Qwen 模式。' : '当前使用本地 Ollama 生成回答与重写 Query；来源编号可追溯到下方证据，请结合原文检查回答。';
}
function showIndex(index, config) {
  const passages=index.corpus_format==='passages_jsonl';
  $('indexInfo').textContent=`${index.documents} 份来源 · ${index.chunks} 个检索单元\n后端：${(config?.retriever_backend || index.backend).toUpperCase()} / ${index.bm25_mode || 'scan'}\n${passages ? '使用原始完整 passage，不再次切块' : `分块：${index.chunk_size} 字符 / 重叠 ${index.overlap} 字符`}`;
}
function clearChat() {history=[]; records=[]; $('conversation').replaceChildren(); const node=element('p','hint','对话已清空，可以开始一个新问题。'); $('conversation').append(node);}
function setBusy(value) {
  busy=value;
  for (const id of ['send','mode','strategy','topK','rebuild','clear','export']) $(id).disabled=value;
  $('send').textContent=value ? '处理中…' : '发送 ↑';
}
function renderMessage(role, text, result) {
  const box=element('article',`message ${role}`);
  box.append(element('div','role',role==='user'?'你':'SimpleRAG'));
  if(result) box.append(element('div','mode-caption',modeNames[result.mode]));
  box.append(element('div','content',text));
  if(result) {
    const detail=element('details','trace');
    detail.append(element('summary','',`查看检索过程 · ${result.hits.length} 个块 · ${result.latencies_ms.total} ms`));
    detail.append(element('pre','',`检索 Query：\n${result.search_query}\n\n改写来源：${rewriteNames[result.rewrite_source] || result.rewrite_source}\n检索后端：${result.backend}\n上下文长度：${result.context.length} 字符\n阶段耗时 (ms)：${JSON.stringify(result.latencies_ms)}`));
    if(result.rewrite_error) detail.append(element('p','warning',`重写回退原因：${result.rewrite_error}`));
    if(result.rewrite_trace) detail.append(element('pre','',`重写记录：${JSON.stringify(result.rewrite_trace,null,2)}`));
    for(const hit of result.hits) {
      const source=result.sources.find(s=>s.id===hit.id);
      detail.append(element('div','source',`${source ? '['+source.label+']' : '[未纳入上下文]'} ${hit.document} · score=${Number(hit.score).toFixed(4)}\n${hit.text}`));
    }
    if(!result.hits.length) detail.append(element('p','','没有匹配证据。'));
    box.append(detail);
    for(const warning of result.warnings) box.append(element('p','warning',warning));
  }
  $('conversation').append(box); return box;
}
$('questionForm').addEventListener('submit', async event=>{
  event.preventDefault(); if(busy) return;
  const question=$('question').value.trim(); if(!question) return;
  const count=Number($('topK').value); if(!Number.isInteger(count)||count<1||count>20) { $('notice').textContent='Top-K 必须为1到20的整数。'; return; }
  $('welcome')?.remove();
  renderMessage('user',question); $('question').value=''; setBusy(true);
  const pending=element('div','busy','正在构造 Query、检索资料并处理回答…'); $('conversation').append(pending);
  pending.scrollIntoView({behavior:'smooth',block:'nearest'});
  try {
    const result=await api('/api/answer',{question, history, strategy:$('strategy').value, top_k:count, mode:$('mode').value});
    pending.remove(); const box=renderMessage('assistant',result.answer,result);
    history.push({role:'user',content:question},{role:'assistant',content:result.answer});
    // Keep a bounded browser request; the server also applies its configured turn limit.
    history=history.slice(-historyTurns*2).map(m=>({...m,content:m.content.slice(0,4000)})); records.push(result);
    box.scrollIntoView({behavior:'smooth',block:'start'});
  } catch(error) {
    pending.remove(); const node=renderMessage('assistant',`运行失败：${error.message}\n\nQwen 模式需先启动 Ollama，并安装项目配置中的模型。也可以切换到离线证据摘录。`); node.classList.add('error');
  } finally {setBusy(false); $('question').focus();}
});
$('question').addEventListener('keydown',event=>{if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){event.preventDefault();$('questionForm').requestSubmit();}});
document.querySelectorAll('[data-question]').forEach(button=>button.addEventListener('click',()=>{$('question').value=button.dataset.question;$('question').focus();}));
$('mode').addEventListener('change',()=>{clearChat();updateMode();});
$('clear').addEventListener('click',clearChat);
$('rebuild').addEventListener('click',async()=>{
  if(busy) return; setBusy(true);
  try {const result=await api('/api/rebuild',{});clearChat();updateMode();$('notice').textContent+=' 索引已重建，对话已清空。';showIndex(result.index);}catch(error){$('notice').textContent=error.message;}finally{setBusy(false);}
});
$('export').addEventListener('click',()=>{
  const blob=new Blob([JSON.stringify({exported_at:new Date().toISOString(),records},null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob), link=document.createElement('a');link.href=url;link.download='rag-conversation.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
async function initialize(){
  try {const status=await api('/api/status');$('mode').value=status.mode;$('topK').value=status.config.top_k;
    historyTurns=status.config.history_turns;
    updateMode();
    if (status.initialization?.state === 'building') {
      setBusy(true);
      $('badge').textContent='准备索引中';
      $('indexInfo').textContent='正在建立或加载索引，进度见启动窗口。';
      $('notice').textContent='网页已启动。首次混合索引需要为语料生成向量，请保持启动窗口打开；完成后本页自动恢复。';
      setTimeout(initialize, 1500);
    } else if (status.initialization?.state === 'error') {
      setBusy(false);
      $('badge').textContent='索引准备失败';
      $('indexInfo').textContent=status.initialization.error;
      $('notice').textContent=`索引准备失败：${status.initialization.error}。可查看启动窗口中的报错，修复后重新启动。`;
    } else {
      setBusy(false);
      showIndex(status.index,status.config);
    }
  }catch(error){$('notice').textContent=`服务连接失败：${error.message}`;$('badge').textContent='连接失败';}
}
initialize();
