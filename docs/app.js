/* Stock Intelligence — static front end.

   Reads only the JSON files the scheduled build committed. There is no
   API call from this page and no key in this file. If a section is
   missing from the payload, the page says which build step failed
   rather than rendering an empty panel. */

const FILES = ['status', 'stocks', 'macro', 'models', 'etf'];
const data = {};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;
  return n;
};

const fmt = (v, d = 2) =>
  v === null || v === undefined || Number.isNaN(v) ? '—' : Number(v).toFixed(d);

function failurePanel(title, body, detail) {
  const box = el('div', 'failure');
  box.append(el('h3', null, title));
  if (body) box.append(el('p', null, body));
  if (detail) box.append(el('pre', null, detail));
  return box;
}

function stamp(provider, asOf) {
  const s = el('span', 'pstamp');
  s.append(document.createTextNode('via '));
  s.append(el('b', null, provider || 'unknown'));
  if (asOf) s.append(document.createTextNode('· as of ' + asOf));
  return s;
}

/* ---------------------------------------------------------------- load */
async function load() {
  await Promise.all(
    FILES.map(async (name) => {
      try {
        const r = await fetch(`data/${name}.json`, { cache: 'no-store' });
        data[name] = r.ok ? await r.json() : { error: `HTTP ${r.status}` };
      } catch (e) {
        data[name] = { error: String(e) };
      }
    })
  );
  renderRecord();
  buildTabs();
  renderHealth();
  renderStocks();
  renderMacro();
  renderModels();
  renderEtf();
}

/* -------------------------------------------------------- record head */
function renderRecord() {
  const s = data.status || {};
  $('rec-run').textContent = s.run_id ? `#${s.run_id}` : '—';
  $('rec-built').textContent = s.built_at || '—';
  $('rec-commit').textContent = s.commit || 'local';

  const live = s.providers_live || [];
  const missing = s.providers_missing_key || [];
  const sections = s.sections || {};
  const broken = Object.entries(sections).filter(([, v]) => !v.ok).map(([k]) => k);

  const trust = $('trust');
  const verdict = $('trust-verdict');
  const detail = $('trust-detail');

  if (s.error) {
    trust.classList.add('is-degraded');
    verdict.textContent = 'NO BUILD RECORD YET';
    detail.textContent =
      'This page shows data committed by the refresh workflow, and it has not run yet. Open the Actions tab, choose "refresh site", and press Run workflow.';
    return;
  }

  if (!live.length) {
    trust.classList.add('is-failed');
    verdict.textContent = 'NO PROVIDER ANSWERED';
    detail.textContent =
      'The build ran with no working data source. Add repository secrets under Settings → Secrets and variables → Actions, then re-run the workflow.';
    return;
  }

  const total = live.length + missing.length;
  verdict.textContent = `DATA LAYER — ${live.length} OF ${total} PROVIDERS LIVE`;

  if (broken.length) {
    trust.classList.add('is-degraded');
    detail.textContent = `Built with failures in: ${broken.join(', ')}. Those sheets show what went wrong instead of a figure.`;
  } else {
    detail.textContent =
      'All sheets built from live sources. Each figure below names the provider it came from and the date it describes.';
  }
}

/* --------------------------------------------------------------- tabs */
const TABS = [
  ['health', 'Health'],
  ['stocks', 'Stocks'],
  ['macro', 'Macro & metals'],
  ['model', 'Model'],
  ['etf', 'ETF overlap'],
];

function buildTabs() {
  const nav = $('tabs');
  TABS.forEach(([id, label], i) => {
    const b = el('button', null, label);
    b.setAttribute('role', 'tab');
    b.setAttribute('aria-selected', i === 0 ? 'true' : 'false');
    b.onclick = () => {
      TABS.forEach(([oid]) => {
        $('sheet-' + oid).hidden = oid !== id;
      });
      [...nav.children].forEach((c) => c.setAttribute('aria-selected', String(c === b)));
    };
    nav.append(b);
  });
}

/* ------------------------------------------------------------- health */
function renderHealth() {
  const root = $('sheet-health');
  const s = data.status || {};

  root.append(el('p', 'eyebrow', 'Providers'));
  root.append(
    el(
      'p',
      'note',
      'The router tries these in order and stops at the first response that passes the schema contract. A provider that returns malformed data is recorded as a failure, not used.'
    )
  );

  const card = el('div', 'card');
  const grid = el('dl', 'figures');
  (s.providers_live || []).forEach((p) => {
    const f = el('div', 'figure');
    f.append(el('dt', null, p));
    const dd = el('dd');
    dd.append(el('span', 'tag ok', 'live'));
    f.append(dd);
    grid.append(f);
  });
  (s.providers_missing_key || []).forEach((p) => {
    const f = el('div', 'figure');
    f.append(el('dt', null, p));
    const dd = el('dd');
    dd.append(el('span', 'tag warn', 'no key'));
    f.append(dd);
    grid.append(f);
  });
  card.append(grid);
  root.append(card);

  const sections = Object.entries(s.sections || {});
  if (sections.length) {
    root.append(el('p', 'eyebrow', 'Build steps'));
    const c = el('div', 'card');
    c.append(
      table(
        ['Step', 'Result', 'Detail'],
        sections.map(([k, v]) => [k, v.ok ? 'ok' : 'failed', v.error || '—'])
      )
    );
    root.append(c);
  }

  const attempts = s.attempts || [];
  if (attempts.length) {
    root.append(el('p', 'eyebrow', 'Provider attempt log'));
    root.append(
      el(
        'p',
        'note',
        'Every request made during the build, in order, with the reason each provider was accepted or rejected. This is the first place to look when a figure is missing.'
      )
    );
    const c = el('div', 'card');
    c.append(
      table(
        ['Request', 'Provider', 'Result', 'ms', 'Detail'],
        attempts.map((a) => [a.request, a.provider, a.ok ? 'ok' : 'failed', a.ms, a.detail])
      )
    );
    root.append(c);
  }
}

function table(headers, rows) {
  const wrap = el('div', 'scroll');
  const t = el('table');
  const thead = el('thead');
  const hr = el('tr');
  headers.forEach((h, i) => {
    const th = el('th', i > 0 && i < headers.length - 1 ? 'num' : null, h);
    hr.append(th);
  });
  thead.append(hr);
  t.append(thead);

  const tb = el('tbody');
  rows.forEach((r) => {
    const tr = el('tr');
    r.forEach((v, i) => {
      const td = el('td', i > 0 && i < r.length - 1 ? 'num' : null);
      if (v === 'ok') td.append(el('span', 'tag ok', 'ok'));
      else if (v === 'failed') td.append(el('span', 'tag bad', 'failed'));
      else if (v === 'STALE') td.append(el('span', 'tag warn', 'stale'));
      else td.textContent = v === null || v === undefined ? '—' : String(v);
      tr.append(td);
    });
    tb.append(tr);
  });
  t.append(tb);
  wrap.append(t);
  return wrap;
}

/* ------------------------------------------------------------- charts */
function lineChart(dates, series, height = 190) {
  const W = 900;
  const H = height;
  const pad = { t: 10, r: 8, b: 20, l: 46 };

  const all = series.flatMap((s) => s.values).filter((v) => v !== null && !Number.isNaN(v));
  if (!all.length || dates.length < 2) return el('div', 'note', 'Not enough data to plot.');

  const min = Math.min(...all);
  const max = Math.max(...all);
  const span = max - min || 1;
  const x = (i) => pad.l + (i / (dates.length - 1)) * (W - pad.l - pad.r);
  const y = (v) => pad.t + (1 - (v - min) / span) * (H - pad.t - pad.b);

  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  svg.setAttribute('preserveAspectRatio', 'none');
  svg.setAttribute('role', 'img');

  [0, 0.5, 1].forEach((f) => {
    const v = min + f * span;
    const gy = y(v);
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('x1', pad.l);
    line.setAttribute('x2', W - pad.r);
    line.setAttribute('y1', gy);
    line.setAttribute('y2', gy);
    line.setAttribute('stroke', 'var(--rule)');
    line.setAttribute('stroke-width', '1');
    svg.append(line);

    const t = document.createElementNS(ns, 'text');
    t.setAttribute('x', 4);
    t.setAttribute('y', gy + 3.5);
    t.setAttribute('fill', 'var(--muted)');
    t.setAttribute('font-size', '10');
    t.setAttribute('font-family', 'IBM Plex Mono, monospace');
    t.textContent = v >= 1000 ? v.toFixed(0) : v.toFixed(2);
    svg.append(t);
  });

  series.forEach((s) => {
    let d = '';
    let pen = false;
    s.values.forEach((v, i) => {
      if (v === null || Number.isNaN(v)) {
        pen = false;
        return;
      }
      d += `${pen ? 'L' : 'M'}${x(i).toFixed(1)} ${y(v).toFixed(1)}`;
      pen = true;
    });
    const p = document.createElementNS(ns, 'path');
    p.setAttribute('d', d);
    p.setAttribute('fill', 'none');
    p.setAttribute('stroke', s.color);
    p.setAttribute('stroke-width', s.width || 1.6);
    if (s.dash) p.setAttribute('stroke-dasharray', s.dash);
    svg.append(p);
  });

  [0, dates.length - 1].forEach((i) => {
    const t = document.createElementNS(ns, 'text');
    t.setAttribute('x', i === 0 ? pad.l : W - pad.r);
    t.setAttribute('y', H - 5);
    t.setAttribute('text-anchor', i === 0 ? 'start' : 'end');
    t.setAttribute('fill', 'var(--muted)');
    t.setAttribute('font-size', '10');
    t.setAttribute('font-family', 'IBM Plex Mono, monospace');
    t.textContent = dates[i];
    svg.append(t);
  });

  const box = el('div', 'chart');
  box.append(svg);
  return box;
}

/* ------------------------------------------------------------- stocks */
function renderStocks() {
  const root = $('sheet-stocks');
  const p = data.stocks || {};

  if (p.error) {
    root.append(
      failurePanel(
        'The stocks step did not complete',
        'Nothing is shown here because nothing could be verified. Open the failed workflow run for the traceback.',
        p.error
      )
    );
    return;
  }

  Object.entries(p.failures || {}).forEach(([sym, reason]) => {
    root.append(
      failurePanel(
        `${sym} — no provider returned usable data`,
        'This ticker is omitted rather than shown with gaps. Every provider is listed below with the reason it was rejected.',
        reason
      )
    );
  });

  Object.entries(p.stocks || {}).forEach(([sym, s]) => {
    const card = el('div', 'card');

    const head = el('div', 'card-head');
    const left = el('div');
    left.append(el('span', 'ticker', sym));
    const right = el('div');
    right.append(el('span', 'price', fmt(s.close)));
    const d = el('span', 'delta ' + (s.change_pct >= 0 ? 'up' : 'down'),
      `${s.change_pct >= 0 ? '+' : ''}${fmt(s.change_pct)}%`);
    right.append(d);
    head.append(left, right);
    card.append(head);

    const grid = el('dl', 'figures');
    Object.entries(s.metrics).forEach(([k, v]) => {
      const f = el('div', 'figure');
      f.append(el('dt', null, k));
      f.append(
        v === null ? el('dd', 'absent', 'not available') : el('dd', null, fmt(v, 2))
      );
      grid.append(f);
    });
    const st = el('div', 'figure');
    st.append(el('dt', null, 'Structure'));
    st.append(el('dd', null, s.structure));
    grid.append(st);
    card.append(grid);

    if (s.chart && s.chart.dates) {
      card.append(
        lineChart(s.chart.dates, [
          { values: s.chart.sma200, color: 'var(--muted)', width: 1, dash: '3 3' },
          { values: s.chart.ema21, color: 'var(--withheld)', width: 1.2 },
          { values: s.chart.close, color: 'var(--accent)', width: 1.8 },
        ])
      );
      const lg = el('div', 'legend');
      lg.append(el('span', 'l-close', 'Close'), el('span', 'l-ema', 'EMA 21'), el('span', 'l-sma', 'SMA 200'));
      card.append(lg);
    }

    if (s.events && s.events.length) {
      const ul = el('ul', 'events');
      s.events.forEach((e) => {
        const li = el('li', null, e.event);
        const t = el('time', null, e.date);
        li.append(t);
        ul.append(li);
      });
      card.append(ul);
    }

    const foot = el('div', 'legend');
    foot.append(stamp(s.provider, s.as_of));
    card.append(foot);

    root.append(card);
  });
}

/* -------------------------------------------------------------- macro */
function renderMacro() {
  const root = $('sheet-macro');
  const m = data.macro || {};

  if (m.error) {
    root.append(
      failurePanel(
        'The macro step did not complete',
        'The macro sheet is built entirely from FRED. There is no second source for TIPS real yields, so this sheet is empty rather than approximate.',
        m.error
      )
    );
    return;
  }

  root.append(el('p', 'eyebrow', 'Regime'));
  const rc = el('div', 'card');
  const head = el('div', 'card-head');
  head.append(el('span', 'ticker', m.regime));
  head.append(el('span', 'pstamp', `confidence ${m.confidence}`));
  rc.append(head);
  rc.append(el('p', 'note', m.playbook || ''));

  const g = el('dl', 'figures');
  Object.entries(m.detail || {}).forEach(([k, v]) => {
    const f = el('div', 'figure');
    f.append(el('dt', null, k.replace(/_/g, ' ')));
    f.append(el('dd', null, fmt(v)));
    g.append(f);
  });
  rc.append(g);

  if (m.missing && m.missing.length) {
    rc.append(
      el('p', 'note', `Classified without: ${m.missing.join(', ')}. Read the regime with that in mind.`)
    );
  }
  root.append(rc);

  if (m.metals_error) {
    root.append(
      failurePanel('Metals prices unavailable', 'The regime above is unaffected — it is built from FRED alone.', m.metals_error)
    );
  }

  if (m.metals) {
    root.append(el('p', 'eyebrow', 'Gold and silver'));
    const fw = m.metals.framework || {};
    const cls = fw.status === 'intact' ? 'ok' : fw.status === 'weak' ? 'warn' : 'bad';
    const c = el('div', 'card');
    const h = el('div', 'card-head');
    const l = el('div');
    l.append(el('span', 'tag ' + cls, `framework ${fw.status || 'unknown'}`));
    h.append(l);
    h.append(stamp('proxies', `${m.metals.gold_proxy} / ${m.metals.silver_proxy}`));
    c.append(h);
    c.append(el('p', 'note', fw.detail || ''));

    const gg = el('dl', 'figures');
    Object.entries(m.metals.latest).forEach(([k, v]) => {
      const f = el('div', 'figure');
      f.append(el('dt', null, k));
      f.append(v === null ? el('dd', 'absent', 'not available') : el('dd', null, fmt(v)));
      gg.append(f);
    });
    c.append(gg);

    if (m.metals.chart && m.metals.chart.dates) {
      c.append(el('p', 'eyebrow', 'Gold / silver ratio'));
      c.append(
        lineChart(m.metals.chart.dates, [
          { values: m.metals.chart.gs_ratio, color: 'var(--accent)', width: 1.8 },
        ], 150)
      );
    }
    root.append(c);
  }

  if (m.freshness && m.freshness.length) {
    root.append(el('p', 'eyebrow', 'Series freshness'));
    root.append(
      el(
        'p',
        'note',
        'Each series is judged against its own release cadence. Monthly CPI at 40 days old is normal; a daily series at 40 days old is broken. A single global threshold cannot tell those apart.'
      )
    );
    const c = el('div', 'card');
    c.append(
      table(
        ['Series', 'FRED id', 'Freq', 'Last observation', 'Age (d)', 'Tolerance', 'Status'],
        m.freshness.map((r) => [
          r.series,
          r.fred_id,
          r.freq,
          (r.last_obs || '').slice(0, 10),
          r.age_days,
          r.tolerance_days,
          r.status === 'ok' ? 'ok' : 'STALE',
        ])
      )
    );
    root.append(c);
  }
}

/* ------------------------------------------------------------- models */
function renderModels() {
  const root = $('sheet-model');
  const p = data.models || {};

  root.append(el('p', 'eyebrow', 'Directional model'));
  root.append(
    el(
      'p',
      'note',
      'The model may not show a probability until it beats the base rate out of sample, stays positive across folds, and survives a block permutation test. When it fails, you get the reasons instead of a number. Most tickers fail — that is the check working, not a bug.'
    )
  );

  if (p.error) {
    root.append(failurePanel('The model step did not complete', null, p.error));
    return;
  }

  Object.entries(p.models || {}).forEach(([sym, m]) => {
    const box = el('div', 'verdict');
    const line = el('div', 'verdict-line');
    line.append(el('span', 'ticker', sym));

    const cls =
      m.status === 'QUALIFIED' ? 'qualified' : m.status === 'ERROR' ? 'error' : 'withheld';
    const label =
      m.status === 'QUALIFIED'
        ? 'QUALIFIED'
        : m.status === 'ERROR'
        ? 'COULD NOT RUN'
        : 'WITHHELD';
    line.append(el('span', 'verdict-status ' + cls, label));
    if (m.horizon_days) line.append(el('span', 'pstamp', `${m.horizon_days}-day horizon`));
    box.append(line);

    if (m.probability !== null && m.probability !== undefined) {
      box.append(el('div', 'prob', `${(m.probability * 100).toFixed(1)}%`));
      box.append(
        el(
          'p',
          'note',
          `Probability the next ${m.horizon_days} days clear round-trip costs. Compare it to the base rate below, not to 50%.`
        )
      );
    }

    if (m.reasons && m.reasons.length) {
      const ul = el('ul', 'reasons');
      m.reasons.forEach((r) => ul.append(el('li', null, r)));
      box.append(ul);
    }

    const sc = Object.entries(m.scorecard || {});
    if (sc.length) {
      const g = el('dl', 'figures');
      sc.forEach(([k, v]) => {
        const f = el('div', 'figure');
        f.append(el('dt', null, k));
        f.append(el('dd', null, typeof v === 'number' ? fmt(v, 4) : String(v)));
        g.append(f);
      });
      box.append(g);
    }

    if (m.reliability && m.reliability.length) {
      box.append(el('p', 'eyebrow', 'Calibration'));
      box.append(
        el('p', 'note', 'Where predicted and observed diverge, the confidence is fiction even if the ranking is sound.')
      );
      box.append(
        table(
          ['Bin', 'n', 'Predicted', 'Observed', 'Gap'],
          m.reliability.map((r) => [r.bin, r.n, fmt(r.predicted, 3), fmt(r.observed, 3), fmt(r.gap, 3)])
        )
      );
    }

    root.append(box);
  });

  if (p.trained_with_macro === false) {
    root.append(
      el('p', 'note', 'Trained without macro features — FRED was unavailable during this build.')
    );
  }
}

/* ---------------------------------------------------------------- etf */
function renderEtf() {
  const root = $('sheet-etf');
  const p = data.etf || {};

  if (p.error) {
    root.append(failurePanel('The ETF step did not complete', null, p.error));
    return;
  }

  (p.notes || []).forEach((n) => root.append(el('p', 'note', n)));

  Object.entries(p.failures || {}).forEach(([etf, reason]) => {
    root.append(
      failurePanel(
        `${etf} — holdings unavailable`,
        'This fund is left out of the overlap figures below rather than counted as zero. For low-AUM thematic funds the fix is the issuer holdings file, not another API key.',
        reason
      )
    );
  });

  if (p.concentration && p.concentration.length) {
    root.append(el('p', 'eyebrow', 'Concentration'));
    root.append(
      el(
        'p',
        'note',
        'Effective holdings is 1/HHI — the number of equally weighted positions that would carry the same concentration. It is usually far below the headline holdings count.'
      )
    );
    const c = el('div', 'card');
    c.append(
      table(
        ['ETF', 'Holdings', 'Top 10 weight', 'HHI', 'Effective holdings', 'Largest', 'Weight'],
        p.concentration.map((r) => [
          r.etf,
          r.holdings,
          fmt(r.top10_weight, 3),
          fmt(r.hhi, 3),
          fmt(r.effective_n, 1),
          r.largest,
          fmt(r.largest_weight, 3),
        ])
      )
    );
    root.append(c);
  }

  if (p.overlap_weight) {
    root.append(el('p', 'eyebrow', 'Shared weight'));
    root.append(
      el(
        'p',
        'note',
        'The fraction of capital genuinely duplicated if both funds are held at equal size. Shared name counts are shown separately because they overstate overlap badly.'
      )
    );
    const c = el('div', 'card');
    c.append(matrix(p.overlap_weight));
    root.append(c);

    if (p.overlap_names) {
      root.append(el('p', 'eyebrow', 'Shared names'));
      const c2 = el('div', 'card');
      c2.append(matrix(p.overlap_names));
      root.append(c2);
    }
  }

  if (p.look_through && p.look_through.length) {
    root.append(el('p', 'eyebrow', 'Look-through exposure'));
    root.append(
      el(
        'p',
        'note',
        'What you actually own across the whole sleeve, assuming equal dollars in each fund. This is where a four-fund sleeve turns out to be one position.'
      )
    );
    const c = el('div', 'card');
    c.append(
      table(
        ['Symbol', 'Effective weight', 'Appears in'],
        p.look_through.map((r) => [
          r.symbol,
          fmt(r.effective_weight * 100, 2) + '%',
          `${r.in_n_etfs} fund${r.in_n_etfs > 1 ? 's' : ''}`,
        ])
      )
    );
    root.append(c);
  }
}

function matrix(m) {
  const wrap = el('div', 'scroll');
  const t = el('table');
  const thead = el('thead');
  const hr = el('tr');
  hr.append(el('th', null, ''));
  m.labels.forEach((l) => hr.append(el('th', 'num', l)));
  thead.append(hr);
  t.append(thead);

  const tb = el('tbody');
  m.matrix.forEach((row, i) => {
    const tr = el('tr');
    tr.append(el('th', null, m.labels[i]));
    row.forEach((v, j) => {
      const td = el('td', 'heat', (v * 100).toFixed(1) + '%');
      if (i !== j) {
        td.style.background = `rgba(27,58,92,${(v * 0.5).toFixed(3)})`;
        if (v > 0.6) td.style.color = '#fff';
      } else {
        td.style.color = 'var(--muted)';
      }
      tr.append(td);
    });
    tb.append(tr);
  });
  t.append(tb);
  wrap.append(t);
  return wrap;
}

load();
