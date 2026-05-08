// autoresearch.jsx — the autoresearch ledger page (Karpathy-style)
// Centerpiece: 35-cycle perf-ledger chart + KEEP/RETIRE breakdown + retraction story.

const CYCLES = [
  { c: 1,  ts: "2025-10-15", tps_dense: 42.17,  b: 1,  state: "BASELINE", note: "v1.7.0 · single-request smoke; admission ladder warm but unbalanced" },
  { c: 2,  ts: "2025-10-17", tps_dense: 58.4,   b: 4,  state: "KEEP",     note: "M-1.1 · MemoryBudgeter ladder rewired; admit pressure halved" },
  { c: 3,  ts: "2025-10-19", tps_dense: 63.1,   b: 8,  state: "RETIRE",   note: "scheduler micro-batch coalescing — 0.7σ over noise floor; closed" },
  { c: 4,  ts: "2025-10-21", tps_dense: 71.0,   b: 8,  state: "KEEP",     note: "M-2.3 · radix-trie block-aligned hits; prefix reuse on hybrid stacks" },
  { c: 5,  ts: "2025-10-23", tps_dense: 73.5,   b: 16, state: "RETIRE",   note: "kernel-fused attention — wins on synthetic, regresses on Qwen3.5-MoE" },
  { c: 6,  ts: "2025-10-25", tps_dense: 84.4,   b: 16, state: "KEEP",     note: "M-3.0 · BlockTQ B=64 4-bit; admission headroom +27%" },
  { c: 7,  ts: "2025-10-26", tps_dense: 89.1,   b: 24, state: "KEEP",     note: "M-3.1 · row-priority queue invariant + cooperative preempt" },
  { c: 8,  ts: "2025-10-27", tps_dense: 92.2,   b: 24, state: "RETIRE",   note: "warmup compile cache reuse; +0.3σ; not load-bearing" },
  { c: 9,  ts: "2025-10-28", tps_dense: 94.0,   b: 24, state: "RETIRE",   note: "speculative draft head — 17 attempts, 0 reproducible E2E wins" },
  { c: 10, ts: "2025-10-30", tps_dense: 108.8,  b: 32, state: "KEEP",     note: "M-4.0 · ExtRaBitQ 3-bit codec slot; admission headroom doubled" },
  { c: 11, ts: "2025-11-02", tps_dense: 114.2,  b: 32, state: "RETIRE",   note: "rotary-cache reshape fast-path — wins synthetic, lost in noise on bench" },
  { c: 12, ts: "2025-11-03", tps_dense: 116.0,  b: 32, state: "RETRACTED",note: "C-12 RETRACTED — apples-to-oranges replay (B=32 vs B=24); ledger entry corrected", retracted: true },
  { c: 13, ts: "2025-11-05", tps_dense: 122.0,  b: 32, state: "KEEP",     note: "M-4.4 · prefix-cache snapshot/restore on hybrid DeltaNet path" },
  { c: 14, ts: "2025-11-06", tps_dense: 127.1,  b: 40, state: "RETIRE",   note: "MoE expert-prefetch lookahead — collapsed under 256-expert top-8" },
  { c: 15, ts: "2025-11-07", tps_dense: 134.9,  b: 40, state: "KEEP",     note: "M-5.0 · MemoryBudgeter pressure mirrors live KV resident bytes" },
  { c: 16, ts: "2025-11-09", tps_dense: 138.4,  b: 40, state: "RETIRE",   note: "speculative draft pair (small-B replay) — δ < noise floor" },
  { c: 17, ts: "2025-11-10", tps_dense: 145.0,  b: 48, state: "KEEP",     note: "M-5.4 · admit→evict→preempt→reject ladder closed-form" },
  { c: 18, ts: "2025-11-11", tps_dense: 152.6,  b: 48, state: "KEEP",     note: "M-5.5 · row-state snapshot for hybrid recurrent state" },
  { c: 19, ts: "2025-11-12", tps_dense: 156.0,  b: 48, state: "RETIRE",   note: "kv-quant fused dequant kernel — works isolated, regresses E2E" },
  { c: 20, ts: "2025-11-13", tps_dense: 161.3,  b: 56, state: "KEEP",     note: "M-6.1 · scheduler step coalescing for batched prefill" },
  { c: 21, ts: "2025-11-14", tps_dense: 168.0,  b: 56, state: "RETIRE",   note: "kv-codec selector heuristic — table-of-thresholds collapsed under MoE" },
  { c: 22, ts: "2025-11-16", tps_dense: 175.4,  b: 56, state: "RETIRE",   note: "page-pinned KV blocks (mlx-rs feedback) — δ within seed variance" },
  { c: 23, ts: "2025-11-17", tps_dense: 181.0,  b: 64, state: "KEEP",     note: "M-6.5 · BlockTQ B=64 default lock-in; admission window stable" },
  { c: 24, ts: "2025-11-18", tps_dense: 189.0,  b: 64, state: "RETIRE",   note: "step-time control-flow elision — wins on smoke, no reproducible E2E gain" },
  { c: 25, ts: "2025-11-19", tps_dense: 196.4,  b: 64, state: "RETIRE",   note: "speculative-decode foundation prefill — drafts reject under MoE" },
  { c: 26, ts: "2025-11-20", tps_dense: 204.0,  b: 64, state: "KEEP",     note: "M-7.0 · radix-trie block reuse across prefill + decode steps" },
  { c: 27, ts: "2025-11-21", tps_dense: 209.0,  b: 64, state: "RETIRE",   note: "draft-token tree-attention — δ < noise on five seeds" },
  { c: 28, ts: "2025-11-22", tps_dense: 215.0,  b: 64, state: "RETIRE",   note: "warm-prefix synthetic admit fast-path — regresses long-prompt parity" },
  { c: 29, ts: "2025-11-23", tps_dense: 221.0,  b: 64, state: "RETIRE",   note: "MoE per-expert residency lookahead — kept as foundation, not E2E win" },
  { c: 30, ts: "2025-11-24", tps_dense: 226.0,  b: 64, state: "KEEP",     note: "M-7.4 · scheduler ladder admission tightened on memory-pressure" },
  { c: 31, ts: "2025-11-25", tps_dense: 228.0,  b: 64, state: "RETIRE",   note: "speculative-decode 4-token tree — kernel attempts 16/17 retired" },
  { c: 32, ts: "2025-11-26", tps_dense: 229.1,  b: 64, state: "RETIRE",   note: "small-B research line (D-022) — closed; B=1 = bandwidth-capped" },
  { c: 33, ts: "2025-11-27", tps_dense: 230.4,  b: 64, state: "RETIRE",   note: "kernel-fused softmax variant — δ within three-seed variance" },
  { c: 34, ts: "2025-11-28", tps_dense: 231.6,  b: 64, state: "KEEP",     note: "M-8.1 · prefix-cache TTL invariant (bounded GC walks)" },
  { c: 35, ts: "2025-11-29", tps_dense: 232.0,  b: 64, state: "GATE",     note: "v1.7.28 · perf phase closed · 5.5× over cycle-1 baseline · 9 KEEPs total" },
];

const PerfChart = () => {
  const W = 1240, H = 380;
  const padL = 64, padR = 24, padT = 24, padB = 56;
  const innerW = W - padL - padR, innerH = H - padT - padB;
  const xs = (c) => padL + ((c - 1) / 34) * innerW;
  const ys = (v) => padT + (1 - v / 250) * innerH;
  const [hover, setHover] = React.useState(null);
  const [progress, setProgress] = React.useState(0);
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined") { setProgress(1); return; }
    let raf;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          const start = performance.now();
          const tick = (now) => {
            const t = Math.min(1, (now - start) / 2400);
            setProgress(t);
            if (t < 1) raf = requestAnimationFrame(tick);
          };
          raf = requestAnimationFrame(tick);
          io.disconnect();
        }
      });
    }, { threshold: 0.3 });
    io.observe(ref.current);
    return () => { io.disconnect(); if (raf) cancelAnimationFrame(raf); };
  }, []);

  const pathD = CYCLES.reduce((acc, p, i) => acc + (i === 0 ? "M" : "L") + xs(p.c) + "," + ys(p.tps_dense) + " ", "");
  const visible = Math.ceil(CYCLES.length * progress);
  const yTicks = [0, 50, 100, 150, 200, 250];
  const xTicks = [1, 5, 10, 15, 20, 25, 30, 35];
  const dotColor = (s) => {
    if (s === "KEEP") return "var(--ok)";
    if (s === "RETIRE") return "var(--ink-4)";
    if (s === "RETRACTED") return "var(--crit)";
    if (s === "GATE") return "var(--accent)";
    return "var(--ink-3)";
  };
  return (
    <div className="perf-chart" ref={ref}>
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" role="img" aria-label="35-cycle dense decode tok/s ledger">
        <defs>
          <linearGradient id="line-grad" x1="0" x2="1">
            <stop offset="0%" stopColor="var(--ink-4)" />
            <stop offset="100%" stopColor="var(--accent)" />
          </linearGradient>
          <linearGradient id="area-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.18" />
            <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
          </linearGradient>
          <clipPath id="reveal-clip">
            <rect x={padL} y={padT} width={innerW * progress} height={innerH} />
          </clipPath>
        </defs>
        {yTicks.map(t => (
          <g key={t}>
            <line x1={padL} x2={W - padR} y1={ys(t)} y2={ys(t)} stroke="var(--rule)" strokeDasharray="2,4" />
            <text x={padL - 8} y={ys(t) + 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-3)">{t}</text>
          </g>
        ))}
        <text x={20} y={padT + 14} fontFamily="var(--font-mono)" fontSize="10" fill="var(--ink-4)" transform={`rotate(-90 20 ${H/2})`} style={{textTransform:"uppercase",letterSpacing:"0.06em"}}>tok/s · dense decode</text>
        {xTicks.map(t => (
          <text key={t} x={xs(t)} y={H - padB + 22} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-3)">C{t}</text>
        ))}
        <line x1={padL} x2={W - padR} y1={ys(42.17)} y2={ys(42.17)} stroke="var(--ink-3)" strokeDasharray="3,3" opacity="0.5" />
        <text x={W - padR - 6} y={ys(42.17) - 6} textAnchor="end" fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-3)">baseline · 42.17</text>
        <line x1={padL} x2={W - padR} y1={ys(232)} y2={ys(232)} stroke="var(--accent)" strokeDasharray="3,3" opacity="0.7" />
        <text x={W - padR - 6} y={ys(232) - 6} textAnchor="end" fontFamily="var(--font-mono)" fontSize="11" fill="var(--accent)">gate · 232 · 5.5×</text>
        <path d={pathD + ` L${xs(35)},${ys(0)} L${xs(1)},${ys(0)} Z`} fill="url(#area-grad)" clipPath="url(#reveal-clip)" />
        <path d={pathD} fill="none" stroke="url(#line-grad)" strokeWidth="2.4" strokeLinejoin="round" clipPath="url(#reveal-clip)" />
        {CYCLES.slice(0, visible).map(p => (
          <g key={p.c} onMouseEnter={() => setHover(p)} onMouseLeave={() => setHover(null)} style={{ cursor: "crosshair" }}>
            <circle cx={xs(p.c)} cy={ys(p.tps_dense)} r="14" fill="transparent" />
            <circle cx={xs(p.c)} cy={ys(p.tps_dense)} r={p.state === "GATE" ? 6 : (p.state === "KEEP" ? 4.4 : 3.2)}
                    fill={dotColor(p.state)}
                    stroke={p.state === "RETRACTED" ? "var(--crit)" : "var(--paper)"}
                    strokeWidth={p.state === "RETRACTED" ? 1 : 1.4} />
            {p.state === "RETRACTED" && (
              <g>
                <line x1={xs(p.c)-7} y1={ys(p.tps_dense)-7} x2={xs(p.c)+7} y2={ys(p.tps_dense)+7} stroke="var(--crit)" strokeWidth="1.4" />
                <line x1={xs(p.c)+7} y1={ys(p.tps_dense)-7} x2={xs(p.c)-7} y2={ys(p.tps_dense)+7} stroke="var(--crit)" strokeWidth="1.4" />
              </g>
            )}
            {p.state === "GATE" && (
              <circle cx={xs(p.c)} cy={ys(p.tps_dense)} r="11" fill="none" stroke="var(--accent)" strokeWidth="1.2">
                <animate attributeName="r" values="6;14;6" dur="2.2s" repeatCount="indefinite" />
                <animate attributeName="opacity" values="0.8;0;0.8" dur="2.2s" repeatCount="indefinite" />
              </circle>
            )}
          </g>
        ))}
        {hover && (
          <g pointerEvents="none">
            <line x1={xs(hover.c)} x2={xs(hover.c)} y1={padT} y2={H - padB} stroke="var(--ink)" strokeDasharray="2,3" opacity="0.4" />
            <rect x={Math.min(xs(hover.c) + 10, W - padR - 280)} y={Math.max(padT, ys(hover.tps_dense) - 56)}
                  width="280" height="68" rx="8" fill="var(--paper)" stroke="var(--rule)" />
            <text x={Math.min(xs(hover.c) + 22, W - padR - 268)} y={Math.max(padT, ys(hover.tps_dense) - 56) + 22}
                  fontFamily="var(--font-mono)" fontSize="11" fill="var(--ink-3)">C{hover.c} · {hover.ts} · B={hover.b}</text>
            <text x={Math.min(xs(hover.c) + 22, W - padR - 268)} y={Math.max(padT, ys(hover.tps_dense) - 56) + 40}
                  fontFamily="var(--font-display)" fontSize="14" fontWeight="500" fill="var(--ink)">{hover.tps_dense.toFixed(2)} tok/s · {hover.state}</text>
            <text x={Math.min(xs(hover.c) + 22, W - padR - 268)} y={Math.max(padT, ys(hover.tps_dense) - 56) + 56}
                  fontFamily="var(--font-sans)" fontSize="10.5" fill="var(--ink-3)">{hover.note.length > 60 ? hover.note.slice(0, 58) + "…" : hover.note}</text>
          </g>
        )}
      </svg>
      <div className="perf-legend mono">
        <span><span className="dot" style={{ background: "var(--ok)" }}></span>KEEP · 9</span>
        <span><span className="dot" style={{ background: "var(--ink-4)" }}></span>RETIRE · 17</span>
        <span><span className="dot" style={{ background: "var(--crit)" }}></span>RETRACTED · 1</span>
        <span><span className="dot" style={{ background: "var(--accent)" }}></span>GATE · 1</span>
      </div>
    </div>
  );
};

const Hero = () => (
  <section className="ar-hero">
    <HeroBg />
    <div className="container" style={{ position: "relative", zIndex: 1 }}>
      <div className="eyebrow reveal"><span className="dot"></span>P-6 · Performance phase · CLOSED</div>
      <h1 className="display-1 reveal reveal-d1" style={{ marginTop: 24 }}>
        From <span className="moment" style={{ color: "var(--ink-3)" }}>42.17</span> to <span className="moment" style={{ color: "var(--accent)" }}>232</span>.<br />
        <span style={{ color: "var(--ink-3)" }}>And how we know.</span>
      </h1>
      <p className="lede reveal reveal-d2" style={{ marginTop: 28, maxWidth: "64ch" }}>
        Karpathy-style perf-ledger. Every cycle gets a measurement, a verdict (KEEP / RETIRE), a one-line note,
        and stays in the file forever. C-12 was retracted in public — apples-to-oranges replay — and the corrected ledger
        is what you read here.
      </p>
    </div>
  </section>
);

const Stats = () => (
  <section className="block tight ar-stats-block">
    <div className="container">
      <div className="ar-stats reveal">
        {[
          { v: 35,  l: "cycles", s: "v1.7.0 → v1.7.35" },
          { v: 9,   l: "KEEPs",  s: "load-bearing E2E wins" },
          { v: 17,  l: "RETIREs",s: "kernel attempts closed honestly" },
          { v: 1,   l: "RETRACTED",s: "C-12 · public correction" },
          { v: "5.5×", l: "speedup", s: "dense decode · server-aggregate", raw: true },
        ].map((s, i) => (
          <div key={i} className="ar-stat">
            <div className="ar-stat-num moment">
              {s.raw ? s.v : <CountUp to={s.v} />}
            </div>
            <div className="ar-stat-label">{s.l}</div>
            <div className="ar-stat-sub mono">{s.s}</div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

const ChartSection = () => (
  <section className="block alt">
    <div className="container">
      <div className="section-head reveal">
        <div className="eyebrow"><span className="accent">●</span> The ledger</div>
        <h2 className="display-2">35 cycles, plotted honestly.</h2>
        <p>Hover any point for the verdict and the one-line note. Retired cycles stay on the chart — the noise floor is the point.</p>
      </div>
      <div className="reveal">
        <PerfChart />
      </div>
    </div>
  </section>
);

const RetractStory = () => (
  <section className="block">
    <div className="container">
      <div className="ar-retract reveal">
        <div className="ar-retract-bar"></div>
        <div className="ar-retract-body">
          <div className="eyebrow" style={{ color: "var(--crit)" }}><span className="dot" style={{ background: "var(--crit)", boxShadow: "0 0 0 3px color-mix(in srgb, var(--crit) 18%, transparent)" }}></span>C-12 · RETRACTED</div>
          <h2 className="display-3" style={{ marginTop: 16 }}>The win that wasn't.</h2>
          <p className="lede" style={{ marginTop: 16, fontSize: 17 }}>
            Cycle 12 reported a 1.2-tok/s gain from a rotary-cache reshape fast-path. Re-reading the
            harness output, the replay ran at <span className="mono">B=32</span>; the cycle-11 baseline was
            <span className="mono"> B=24</span>. The delta was apples-to-oranges. The public retraction lives in the ledger.
            Cycle 13 onward used a single-batch replay protocol. The numbers above are after the correction.
          </p>
          <pre className="ar-quote mono">
{`# perf-ledger.md (excerpt, unedited)
- C-12 · 116.0 tok/s · RETIRE → RETRACTED
  apples-to-oranges replay (B=32 vs B=24 cycle-11
  baseline). corrected ledger preserved; entry
  kept in file as documentation, not a number.`}
          </pre>
        </div>
      </div>
    </div>
  </section>
);

const Filtered = () => {
  const [filter, setFilter] = React.useState("all");
  const filt = (c) => filter === "all" ? true : (filter === "keep" ? c.state === "KEEP" : (filter === "retire" ? c.state === "RETIRE" : c.state === "RETRACTED"));
  const filtered = CYCLES.filter(filt);
  return (
    <section className="block alt">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> Cycle log</div>
          <h2 className="display-2">Every cycle. Every verdict.</h2>
        </div>
        <div className="ar-filter reveal">
          {[
            { k: "all", l: `All · ${CYCLES.length}` },
            { k: "keep", l: `KEEP · ${CYCLES.filter(c=>c.state==="KEEP").length}` },
            { k: "retire", l: `RETIRE · ${CYCLES.filter(c=>c.state==="RETIRE").length}` },
            { k: "retract", l: `RETRACTED · 1` },
          ].map(b => (
            <button key={b.k} className={"ar-filter-btn" + (filter === b.k ? " on" : "")} onClick={() => setFilter(b.k)}>{b.l}</button>
          ))}
        </div>
        <div className="ar-log reveal">
          {filtered.map(c => (
            <div key={c.c} className={"ar-row ar-row-" + c.state.toLowerCase()}>
              <div className="ar-row-c mono">C{String(c.c).padStart(2,"0")}</div>
              <div className="ar-row-d mono">{c.ts}</div>
              <div className="ar-row-tps mono"><span className="tnum">{c.tps_dense.toFixed(2)}</span><span className="subtle"> tok/s</span></div>
              <div className="ar-row-b mono">B={c.b}</div>
              <div className={"ar-row-st mono ar-st-" + c.state.toLowerCase()}>{c.state}</div>
              <div className="ar-row-note">{c.note}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const NextDoors = () => (
  <section className="block">
    <div className="container">
      <div className="doors reveal" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
        <a href="architecture.html" className="door">
          <div className="door-eyebrow mono">Layered stack</div>
          <div className="door-title">How the wins compose</div>
          <div className="door-body">The seven layers and the frozen interfaces that let them slot in cleanly.</div>
          <div className="door-cta">Architecture<span className="arrow">→</span></div>
        </a>
        <a href="changelog.html" className="door">
          <div className="door-eyebrow mono">Roadmap</div>
          <div className="door-title">What's next on the path to 1.0</div>
          <div className="door-body">P-7 quality bench, P-9 speculative decode, M-10 on-disk prefix store.</div>
          <div className="door-cta">Changelog<span className="arrow">→</span></div>
        </a>
        <a href="index.html" className="door">
          <div className="door-eyebrow mono">Overview</div>
          <div className="door-title">Back to the top</div>
          <div className="door-body">The hero numbers, the comparison, the codec stack, the quickstart.</div>
          <div className="door-cta">Overview<span className="arrow">→</span></div>
        </a>
      </div>
    </div>
  </section>
);

const App = () => (
  <PageShell active="autoresearch">
    <Hero />
    <Stats />
    <ChartSection />
    <RetractStory />
    <Filtered />
    <NextDoors />
  </PageShell>
);

ReactDOM.createRoot(document.getElementById("app")).render(<App />);
