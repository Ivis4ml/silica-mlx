// architecture.jsx — 7-layer stack with scheduler animation

const LAYERS = [
  { n: "L7", name: "Frontends",     comp: "silica chat · silica serve",          desc: "FastAPI OpenAI-compat HTTP server, Rich terminal REPL, Python iterator.", color: "v" },
  { n: "L6", name: "Engine",        comp: "Engine.generate · generate_batch",    desc: "One thin facade. Sync iterator and batched event stream over the same scheduler.", color: "v" },
  { n: "L5", name: "Scheduler",     comp: "ContinuousBatcher",                   desc: "admit → evict → preempt → reject. Row-priority queue. Cooperative preemption with replay.", color: "a" },
  { n: "L4", name: "Admission",     comp: "MemoryBudgeter",                      desc: "Closed-form ladder mirroring live KV resident bytes. Headroom is the budget.", color: "a" },
  { n: "L3", name: "KV cache",      comp: "RadixPrefixCache · PrefixBlockStore", desc: "Block-granular trie, per-codec store seam. Block-aligned hits seed row state.", color: "b" },
  { n: "L2", name: "KV codec",      comp: "VectorCodec[P]",                      desc: "BlockTQ · RaBitQ · ExtRaBitQ. Side-level since P-5-A.0.4. ΔPPL accountable.", color: "b" },
  { n: "L1", name: "Model adapter", comp: "ModelAdapter · KVCacheBackend",       desc: "Five families parity-validated: Qwen3 dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B, Qwen3.5-MoE 35B-A3B, Gemma4-MoE 26B-A4B.", color: "g" },
  { n: "L0", name: "MLX runtime",   comp: "mlx-rs",                              desc: "Apple Silicon GPU. Unified memory. M5 Pro 48 GB target.",                              color: "g" },
];

const SchedulerAnim = () => {
  const W = 1200, H = 360;
  const [t, setT] = React.useState(0);
  const [active, setActive] = React.useState(true);
  const ref = React.useRef(null);
  React.useEffect(() => {
    if (!active) return;
    let raf, start = performance.now();
    const tick = (now) => { setT((now - start) / 1000); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  React.useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((es) => es.forEach(e => setActive(e.isIntersecting)), { threshold: 0.05 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, []);
  // 4 zones: queue → batch → KV → out
  const zones = [
    { x: 60,  w: 200, label: "queue", sub: "by priority" },
    { x: 320, w: 360, label: "active batch (B≤64)", sub: "step every token" },
    { x: 740, w: 200, label: "KV resident", sub: "26 GB / 48 GB" },
    { x: 1000,w: 140, label: "out", sub: "tok/s" },
  ];
  // Generate a deterministic stream of "rows": each row enters at t0, sits in queue, then admits, then decodes for some N tokens, then leaves.
  const rows = React.useMemo(() => Array.from({ length: 12 }, (_, i) => ({
    id: i,
    t0: i * 0.55,
    qWait: 0.6 + (i % 4) * 0.3,
    decodeDur: 2.2 + (i % 5) * 0.6,
    track: i % 8, // active-batch slot
  })), []);
  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="xMidYMid meet" className="sched-svg">
      <defs>
        <linearGradient id="sch-grad" x1="0" x2="1">
          <stop offset="0%" stopColor="var(--accent)" stopOpacity="0.08" />
          <stop offset="100%" stopColor="var(--accent)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {zones.map((z, i) => (
        <g key={i}>
          <rect x={z.x} y={36} width={z.w} height={H - 90} rx="14" fill="var(--paper-2)" stroke="var(--rule)" />
          <text x={z.x + 14} y={28} fontFamily="var(--font-mono)" fontSize="10" fill="var(--ink-3)" style={{textTransform:"uppercase",letterSpacing:"0.06em"}}>{z.label}</text>
          <text x={z.x + 14} y={H - 38} fontFamily="var(--font-mono)" fontSize="10" fill="var(--ink-4)">{z.sub}</text>
        </g>
      ))}
      {/* arrows between zones */}
      {[260, 680, 940].map((x, i) => (
        <g key={i}>
          <line x1={x} x2={x + 56} y1={H/2} y2={H/2} stroke="var(--ink-3)" strokeWidth="1" strokeDasharray="2,3" />
          <polygon points={`${x+56},${H/2} ${x+50},${H/2-3} ${x+50},${H/2+3}`} fill="var(--ink-3)" />
        </g>
      ))}
      {/* row tokens */}
      {rows.map(r => {
        const local = (t - r.t0);
        const cycleLen = r.qWait + r.decodeDur + 0.6;
        const cyc = ((local % cycleLen) + cycleLen) % cycleLen;
        let stage, prog;
        if (local < 0) return null;
        if (cyc < r.qWait) { stage = "queue"; prog = cyc / r.qWait; }
        else if (cyc < r.qWait + r.decodeDur) { stage = "batch"; prog = (cyc - r.qWait) / r.decodeDur; }
        else { stage = "out"; prog = (cyc - r.qWait - r.decodeDur) / 0.6; }
        const trackY = 60 + r.track * 28;
        let cx, cy = trackY, fill = "var(--ink-3)";
        if (stage === "queue") {
          cx = 70 + (1 - prog) * 180;
          fill = "var(--ink-3)";
        } else if (stage === "batch") {
          cx = 330 + prog * 340;
          fill = "var(--accent)";
          // pulse during decode
        } else {
          cx = 1010 + prog * 120;
          fill = "var(--ok)";
        }
        return (
          <g key={r.id}>
            <circle cx={cx} cy={cy} r={stage === "batch" ? 5 : 4} fill={fill}
                    opacity={stage === "out" ? 1 - prog : 1}>
              {stage === "batch" && <animate attributeName="r" values="5;6.5;5" dur="0.6s" repeatCount="indefinite" />}
            </circle>
            {stage === "batch" && (
              <line x1={330} x2={cx} y1={cy} y2={cy} stroke="var(--accent)" strokeWidth="1" opacity="0.18" />
            )}
            <text x={cx + 8} y={cy + 3} fontFamily="var(--font-mono)" fontSize="9" fill="var(--ink-4)" opacity={stage === "queue" ? 0.6 : 0}>R{r.id}</text>
          </g>
        );
      })}
      {/* KV memory bar */}
      <rect x={755} y={56} width={170} height={H - 130} rx="6" fill="var(--rule-2)" />
      {(() => {
        const v = 0.5 + 0.18 * Math.sin(t * 0.6);
        const h = (H - 130) * v;
        return (
          <rect x={755} y={56 + (H - 130 - h)} width={170} height={h} rx="6" fill="url(#sch-grad)" stroke="var(--accent)" strokeOpacity="0.5" />
        );
      })()}
      <text x={840} y={H - 70} textAnchor="middle" fontFamily="var(--font-mono)" fontSize="9" fill="var(--ink-3)">resident</text>
    </svg>
  );
};

const Hero = () => (
  <section className="ar-hero">
    <HeroBg />
    <div className="container" style={{ position: "relative", zIndex: 1 }}>
      <div className="eyebrow reveal"><span className="dot"></span>Architecture · 7 layers · frozen seams</div>
      <h1 className="display-1 reveal reveal-d1" style={{ marginTop: 24 }}>
        Seven layers.<br /><span style={{ color: "var(--ink-3)" }}>Frozen seams.</span>
      </h1>
      <p className="lede reveal reveal-d2" style={{ marginTop: 24 }}>
        Capabilities slot in below the interface. The interface doesn't move. That's why a 1.7-series codec change
        didn't ripple out of L2, and why the OpenAI server boots on top without touching the scheduler.
      </p>
    </div>
  </section>
);

const Stack = () => (
  <section className="block">
    <div className="container">
      <div className="section-head reveal">
        <div className="eyebrow"><span className="accent">●</span> The stack</div>
        <h2 className="display-2">Top to bottom.</h2>
      </div>
      <div className="layers reveal">
        {LAYERS.map((l, i) => (
          <div key={l.n} className={"layer layer-" + l.color}>
            <div className="layer-n mono">{l.n}</div>
            <div className="layer-name">
              <div className="layer-title">{l.name}</div>
              <div className="layer-comp mono">{l.comp}</div>
            </div>
            <div className="layer-desc">{l.desc}</div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

const Sched = () => (
  <section className="block alt">
    <div className="container">
      <div className="section-head reveal">
        <div className="eyebrow"><span className="accent">●</span> Scheduler · L5</div>
        <h2 className="display-2">admit → evict → preempt → reject.</h2>
        <p>Rows arrive on the left, sit in the priority queue, get admitted into the active batch (B≤64 today), decode in lockstep, and exit on the right. KV memory at L3 mirrors live resident bytes; the budgeter at L4 reads the same number.</p>
      </div>
      <div className="sched-frame reveal">
        <SchedulerAnim />
        <div className="sched-foot mono">
          <span>step time = max(row-step) · all rows decode the same token-step</span>
          <span>preempt is cooperative · row state snapshots live at L1</span>
        </div>
      </div>
    </div>
  </section>
);

const Decisions = () => {
  const items = [
    { n: "D-001", t: "One scheduler, one path", b: "Single-request and batched share the same code. No 'fast path' that can drift." },
    { n: "D-014", t: "Side-level codec seam", b: "Codecs live at the prefix-block store, not the model. Swap BlockTQ for ExtRaBitQ without touching attention." },
    { n: "D-019", t: "Recurrent state snapshot/restore", b: "Hybrid DeltaNet's recurrent state moves with the row. RadixPrefixCache works on hybrid stacks." },
    { n: "D-022", t: "Small-B research line, closed", b: "B=1 is bandwidth-capped; further wins live at B>1. Investigations folded into the perf phase ledger as RETIREs." },
    { n: "D-027", t: "Server-aggregate is the metric", b: "Per-row decode is misleading at high B. The bench reports server-aggregate tok/s, three seeds, with confidence bands." },
  ];
  return (
    <section className="block">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> Architectural decisions</div>
          <h2 className="display-2">Five that shaped the rest.</h2>
        </div>
        <div className="dec-grid reveal">
          {items.map(d => (
            <div key={d.n} className="dec-card">
              <div className="dec-n mono">{d.n}</div>
              <div className="dec-t">{d.t}</div>
              <div className="dec-b">{d.b}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const App = () => (
  <PageShell active="architecture">
    <Hero />
    <Stack />
    <Sched />
    <Decisions />
  </PageShell>
);
ReactDOM.createRoot(document.getElementById("app")).render(<App />);
