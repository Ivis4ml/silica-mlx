// changelog.jsx — version history + roadmap

const RELEASES = [
  { v: "v1.7.35", d: "2025-11-29", t: "Cycle 35 — perf phase closed", state: "GATE",
    bullets: [
      "Server-aggregate dense decode reaches 232 tok/s on Qwen3.5-27B-4bit · B=64 · M5 Pro 48 GB.",
      "MoE decode reaches 791.8 tok/s on Qwen3.5-35B-A3B-4bit · B=128.",
      "P-6 closed: 35 cycles, 9 KEEPs, 17 RETIREs, 1 RETRACTED.",
      "M-9 cleared: bench harness fronted by 15+ scenario specs across five oracle types.",
    ] },
  { v: "v1.7.33", d: "2025-11-22", t: "P-8 — OpenAI-compatible HTTP server", state: "SHIP",
    bullets: [
      "silica serve boots a single-process FastAPI server fronting one model.",
      "/v1/chat/completions, /v1/completions, /v1/models with SSE streaming.",
      "X-Silica-Session-ID header threads prefix reuse across requests.",
      "Token-bucket rate limit and OpenAI-shaped error envelope.",
    ] },
  { v: "v1.7.28", d: "2025-11-13", t: "P-6 — performance phase ledger gate", state: "GATE",
    bullets: [
      "5.5× over the cycle-1 baseline (42.17 → 232 tok/s) with three-seed CIs.",
      "D-022 closed the small-B research line: B=1 ≈ 20 tok/s = bandwidth-capped.",
      "All 17 retired kernel attempts kept in plans/perf-ledger.md as documentation.",
    ] },
  { v: "v1.7.20", d: "2025-10-30", t: "P-5 — KV codec stack lock-in", state: "SHIP",
    bullets: [
      "BlockTQ B=64 4-bit set as default; ties vqbench at measurement precision (ΔPPL = +0.0016).",
      "ExtRaBitQ added at 2/3/4-bit; RaBitQ-1 retained as the maximum-compression slot.",
      "VectorCodec[P] surface frozen at the prefix-block store.",
    ] },
  { v: "v1.7.10", d: "2025-10-21", t: "P-3 — five families parity-validated", state: "SHIP",
    bullets: [
      "Qwen3 dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B, Qwen3.5-MoE 35B-A3B, Gemma4-MoE 26B-A4B.",
      "Hybrid DeltaNet brought onto the batched path via row-state snapshot/restore.",
      "MoE batched dispatch on 256 × top-8.",
    ] },
  { v: "v1.7.0",  d: "2025-10-15", t: "P-2 — scheduler + prefix cache shipped", state: "SHIP",
    bullets: [
      "ContinuousBatcher: admit → evict → preempt → reject; one code path for B=1 and B>1.",
      "RadixPrefixCache: block-granular trie, per-codec store seam.",
      "MemoryBudgeter: closed-form admission ladder mirroring live KV resident bytes.",
    ] },
];

const NEXT = [
  { code: "P-7", t: "Quality bench foundation", state: "in flight", d: "Per-task PPL harness with confidence bands across five families. Ties the codec stack to downstream task quality, not just isolated PPL." },
  { code: "P-9", t: "Speculative decoding (E2E)", state: "queued",   d: "Foundation work landed in the perf phase as RETIREs. P-9 is the load-bearing E2E investigation: tree-attention drafts, kept under the same three-seed gate." },
  { code: "M-10",t: "On-disk prefix-block store", state: "queued",  d: "Persist the radix trie across restarts. Block store seam already accepts side-loaded codecs; this lets a long session survive process death." },
  { code: "P-11",t: "Per-expert MoE residency",   state: "research", d: "Hot-experts-only residency for Qwen3.5-MoE 35B-A3B and Gemma4-MoE 26B-A4B. Foundation only today; needs E2E gate before lock-in." },
  { code: "1.0", t: "1.0 release",                state: "RC",       d: "Frozen Engine surface, frozen ContinuousBatcher, frozen VectorCodec[P], frozen ModelAdapter. Today: 1.0 RC." },
];

const Hero = () => (
  <section className="ar-hero">
    <HeroBg />
    <div className="container" style={{ position: "relative", zIndex: 1 }}>
      <div className="eyebrow reveal"><span className="dot"></span>v1.7.35 · 1.0 RC</div>
      <h1 className="display-1 reveal reveal-d1" style={{ marginTop: 24 }}>
        Releases.<br /><span style={{ color: "var(--ink-3)" }}>And what's next.</span>
      </h1>
      <p className="lede reveal reveal-d2" style={{ marginTop: 24 }}>
        The history is concise on purpose. Each version names a phase that closed, lists what shipped behind frozen
        interfaces, and links the ledger entry where applicable.
      </p>
    </div>
  </section>
);

const Roadmap = () => (
  <section className="block">
    <div className="container">
      <div className="section-head reveal">
        <div className="eyebrow"><span className="accent">●</span> Roadmap</div>
        <h2 className="display-2">Path to 1.0.</h2>
      </div>
      <div className="rmap reveal">
        {NEXT.map((n, i) => (
          <div key={i} className={"rmap-row rmap-" + n.state.replace(/\s/g, "-")}>
            <div className="rmap-code mono">{n.code}</div>
            <div className="rmap-t">{n.t}</div>
            <div className="rmap-d">{n.d}</div>
            <div className="rmap-st mono">{n.state}</div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

const Releases = () => (
  <section className="block alt">
    <div className="container">
      <div className="section-head reveal">
        <div className="eyebrow"><span className="accent">●</span> Versions</div>
        <h2 className="display-2">Six releases<br /><span style={{ color: "var(--ink-3)" }}>that earned their version bump.</span></h2>
      </div>
      <div className="rel-timeline reveal">
        {RELEASES.map((r, i) => (
          <div key={i} className="rel-row">
            <div className="rel-side">
              <div className="rel-v mono">{r.v}</div>
              <div className="rel-d mono">{r.d}</div>
              <div className={"rel-st mono rel-st-" + r.state.toLowerCase()}>{r.state}</div>
            </div>
            <div className="rel-body">
              <h3 className="rel-t">{r.t}</h3>
              <ul className="rel-bullets">
                {r.bullets.map((b, j) => <li key={j}>{b}</li>)}
              </ul>
            </div>
          </div>
        ))}
      </div>
    </div>
  </section>
);

const App = () => (
  <PageShell active="changelog">
    <Hero />
    <Roadmap />
    <Releases />
  </PageShell>
);
ReactDOM.createRoot(document.getElementById("app")).render(<App />);
