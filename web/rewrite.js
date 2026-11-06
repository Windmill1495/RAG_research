const $ = id => document.getElementById(id);
const samples = new Map();
const sourceNames = {quac: 'QuAC', nq: 'Natural Questions', trec: 'TREC CAsT'};
const methodNames = {rewrite: 'A · 直接重写', structured: 'B · 结构化重写'};
const resultSources = {
  llm: '模型直接重写', llm_structured: '模型结构化重写',
  heuristic_demo: '离线规则演示', heuristic_fallback: '模型失败后的规则回退',
  no_history: '无历史，保留当前问题', error: '运行失败'
};

function element(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (content !== undefined) node.textContent = content;
  return node;
}

async function api(path, body) {
  const request = body === undefined ? {} : {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  };
  const response = await fetch(path, request);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

function showRuntime({mode, model, model_digest: digest, retriever_backend: backend}) {
  const demo = mode === 'demo';
  $('rewriteMode').textContent = demo ? '离线规则演示' : '本地 Ollama';
  $('rewriteModel').textContent = demo
    ? `配置模型为 ${model}。当前两组使用离线规则，所得分数仅用于检查流程。`
    : `当前模型 ${model}${digest ? ` · 权重摘要 ${digest.slice(0, 12)}` : ''}${backend ? ` · 检索 ${backend.toUpperCase()}` : ''}`;
}

function showSample() {
  const sample = samples.get($('rewriteSample').value);
  $('rewriteQuestion').textContent = sample.question;
  $('rewriteHistory').replaceChildren(...sample.history.map(message => {
    const role = message.role === 'user' ? '用户' : '助手';
    return element('p', 'source', `${role} · ${message.content}`);
  }));
  if (!sample.history.length) $('rewriteHistory').append(element('p', 'rewrite-note', '本题没有历史。'));
  $('rewriteResults').replaceChildren();
  $('rewriteReference').textContent = '';
  $('rewriteReferencePanel').hidden = true;
  $('rewriteReferencePanel').open = false;
  $('rewriteNotice').textContent = '';
}

function showResult(row) {
  const card = element('article', 'rewrite-panel');
  const trace = row.rewrite_trace;
  const latency = row.latencies_ms.rewrite;
  const origin = resultSources[row.rewrite_source] || row.rewrite_source;
  const figures = [
    `来源 ${origin}`,
    `ROUGE-1 F1（词面）${row.rouge1_f1.toFixed(4)}`,
    `耗时 ${typeof latency === 'number' ? `${latency} ms` : '—'}`,
    `模型调用 ${trace.llm_calls ?? '—'} 次`,
    `修复 ${trace.repaired ? '已尝试' : '未尝试'}`
  ];
  card.append(
    element('h2', '', methodNames[row.strategy]),
    element('p', 'content rewrite-output', row.search_query || '没有生成独立问题。'),
    element('p', 'rewrite-note', figures.join(' · '))
  );
  if (row.error || row.rewrite_error) {
    card.append(element('p', 'warning', row.error || row.rewrite_error));
  }
  if (typeof row.gold_recall_at_3 === 'number') {
    card.append(element('p', 'rewrite-note', [
      `Gold Recall@1 ${row.gold_recall_at_1.toFixed(4)}`,
      `Gold Recall@3 ${row.gold_recall_at_3.toFixed(4)}`,
      `MRR@3 ${row.mrr_at_3.toFixed(4)}`
    ].join(' · ')));
    if (row.retrieval_error) card.append(element('p', 'warning', row.retrieval_error));
  }
  if (row.strategy === 'structured' && row.rewrite_source !== 'heuristic_demo') {
    const detail = element('details', 'trace');
    const ambiguity = typeof trace.model_ambiguous === 'boolean'
      ? (trace.model_ambiguous ? '有歧义' : '无歧义') : '—';
    detail.append(
      element('summary', '', '查看结构化信息'),
      element('pre', '', [
        `主题 ${trace.topic || '—'}`,
        `模型提出的指代对象 ${trace.referent_proposed || '—'}`,
        `在历史中匹配到的对象 ${trace.referent_verified || '—'}`,
        `模型报告 ${ambiguity}`,
        `首次生成的问题 ${trace.draft_query || '—'}`,
        `检查记录 ${(trace.draft_validation_issues || []).join('、') || '—'}`
      ].join('\n'))
    );
    card.append(detail);
  }
  $('rewriteResults').append(card);
}

$('rewriteSample').addEventListener('change', showSample);
$('rewriteForm').addEventListener('submit', async event => {
  event.preventDefault();
  if ($('rewriteRun').disabled) return;
  $('rewriteSample').disabled = true;
  $('rewriteRun').disabled = true;
  showSample();
  $('rewriteNotice').textContent = '正在运行 A/B，请等待两组结果…';
  try {
    const result = await api('/api/rewrite/compare', {sample_id: $('rewriteSample').value});
    showRuntime(result);
    result.results.forEach(showResult);
    $('rewriteReference').textContent = result.reference_rewrite;
    $('rewriteReferencePanel').hidden = false;
    $('rewriteNotice').textContent = '';
  } catch (error) {
    $('rewriteNotice').textContent = `运行失败。${error.message}`;
  } finally {
    $('rewriteSample').disabled = false;
    $('rewriteRun').disabled = false;
  }
});

async function initialize() {
  try {
    const result = await api('/api/rewrite/samples');
    showRuntime(result);
    $('rewriteHistoryHeading').textContent = `两组看到的对话历史（最近最多 ${result.history_turns} 轮）`;
    for (const sample of result.samples) {
      samples.set(sample.id, sample);
      const option = element('option', '', `${sample.id} · ${sample.split} · ${sourceNames[sample.source] || sample.source} · 第 ${sample.turn} 轮 · ${sample.question}`);
      option.value = sample.id;
      $('rewriteSample').append(option);
    }
    if (samples.size) {
      showSample();
      $('rewriteSample').disabled = false;
      $('rewriteRun').disabled = false;
    }
  } catch (error) {
    $('rewriteMode').textContent = '连接失败';
    $('rewriteNotice').textContent = `读取 QReCC 题目失败。${error.message}`;
  }
}

initialize();
