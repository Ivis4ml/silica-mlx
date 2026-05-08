// chat.jsx — Chat REPL page · live-typing terminal mock + features

const SCRIPT = [
  { who: "user", text: "Explain TTFT in one sentence." },
  { who: "asst", text: "TTFT (time-to-first-token) is the latency from request arrival to the model emitting its first generated token." },
  { who: "metrics", text: "ttft=25.1ms prefill=596.9tok/s decode=151.4tok/s resident_kv=29.4MB peak=1261.5MB logical_kv=29.4MB prompt=15 out=64 wall=0.44s finish=max_tokens" },
  { who: "user", text: "Now show me a haiku about silicon." },
  { who: "asst", text: "Wafers in moonlight,\ntiny lattices align —\nthought without a soul." },
  { who: "metrics", text: "ttft=18.4ms prefill=712.2tok/s decode=164.8tok/s resident_kv=31.2MB peak=1264.3MB prefix_hit=92% prompt=8 out=21 wall=0.31s finish=stop_token" },
];

const TypingTerminal = () => {
  const [step, setStep] = React.useState(0);
  const [partial, setPartial] = React.useState("");
  React.useEffect(() => {
    if (step >= SCRIPT.length) { const t = setTimeout(() => { setStep(0); setPartial(""); }, 4500); return () => clearTimeout(t); }
    const cur = SCRIPT[step];
    let i = 0;
    const speed = cur.who === "metrics" ? 8 : (cur.who === "user" ? 24 : 14);
    const id = setInterval(() => {
      i++;
      setPartial(cur.text.slice(0, i));
      if (i >= cur.text.length) {
        clearInterval(id);
        setTimeout(() => { setStep(s => s + 1); setPartial(""); }, cur.who === "metrics" ? 600 : 1100);
      }
    }, speed);
    return () => clearInterval(id);
  }, [step]);
  const lines = SCRIPT.slice(0, step).concat(step < SCRIPT.length ? [{ who: SCRIPT[step].who, text: partial, partial: true }] : []);
  const ref = React.useRef(null);
  React.useEffect(() => { if (ref.current) ref.current.scrollTop = ref.current.scrollHeight; }, [lines.length, partial]);
  return (
    <div className="repl">
      <div className="repl-bar">
        <span className="repl-dots"><span></span><span></span><span></span></span>
        <span className="repl-title mono">silica chat — Qwen/Qwen3-0.6B · session=#3a17</span>
        <span className="repl-meta mono">B=1 · prefix-hit 92% · ~20 tok/s · M5 Pro</span>
      </div>
      <div className="repl-body" ref={ref}>
        <div className="repl-line repl-sys mono"><span className="c-cm"># silica v1.7.35 · adapter Qwen3 dense · KV BlockTQ B=64 4-bit · max-tokens 256</span></div>
        <div className="repl-line repl-sys mono"><span className="c-cm"># type ":help" for commands · ":t 0.7 :p 0.9 :n 256 :sys ..." inline tweaks</span></div>
        {lines.map((l, i) => (
          <div key={i} className={"repl-line repl-" + l.who}>
            {l.who === "user" && <><span className="repl-prompt">›</span> <span className="repl-text">{l.text}{l.partial && <span className="repl-cursor"></span>}</span></>}
            {l.who === "asst" && <span className="repl-text repl-asst-text">{l.text}{l.partial && <span className="repl-cursor"></span>}</span>}
            {l.who === "metrics" && (
              <span className="repl-metrics mono">
                {l.text.split(" ").map((kv, j) => {
                  const [k, v] = kv.split("=");
                  return v ? <span key={j} className="repl-mkv"><span className="repl-mk">{k}</span>=<span className="repl-mv">{v}</span></span> : <span key={j}>{kv}</span>;
                })}
                {l.partial && <span className="repl-cursor"></span>}
              </span>
            )}
          </div>
        ))}
      </div>
      <div className="repl-input mono">
        <span className="repl-prompt">›</span>
        <span className="repl-cursor repl-cursor-input"></span>
      </div>
    </div>
  );
};

const Hero = () => (
  <section className="ar-hero">
    <HeroBg />
    <div className="container" style={{ position: "relative", zIndex: 1 }}>
      <div className="eyebrow reveal"><span className="dot"></span>silica chat · 1.0 RC · interactive REPL</div>
      <h1 className="display-1 reveal reveal-d1" style={{ marginTop: 24 }}>
        A REPL.<br /><span style={{ color: "var(--ink-3)" }}>With instrumentation.</span>
      </h1>
      <p className="lede reveal reveal-d2" style={{ marginTop: 24 }}>
        Live decode tok/s, prefix-hit fraction, peak resident KV. The same numbers the bench harness reports — visible per turn,
        right under the model's reply.
      </p>
    </div>
  </section>
);

const TerminalSection = () => (
  <section className="block tight">
    <div className="container reveal">
      <TypingTerminal />
    </div>
  </section>
);

const Features = () => {
  const items = [
    { eb: "instrumentation", t: "Per-turn metrics block.", b: "TTFT, prefill tok/s, decode tok/s, resident_kv, peak, prefix-hit, finish reason. The same numbers the bench prints." },
    { eb: "thinking", t: "Three-axis thinking model.", b: "Tag thoughts as PLAN, REASON, REPORT. The REPL renders them folded by default; ':think open' expands inline." },
    { eb: "shortcuts", t: "Inline knobs, no menus.", b: ":t 0.7  :p 0.9  :n 256  :sys you-are-…  :clear  :stash  :history  :seed 42  ↑↓ recall." },
    { eb: "sessions", t: "X-Silica-Session-ID prefix reuse.", b: "The same prefix-cache that powers the server keeps your conversation cheap. ~92% block reuse on follow-ups." },
    { eb: "models", t: "Five families, one binary.", b: "silica chat --model <repo>. Qwen3 dense, Qwen3.5 hybrid, Gemma4-31B, Qwen3.5-MoE 35B-A3B, Gemma4-MoE 26B-A4B." },
    { eb: "harness", t: "Drives the bench.", b: "silica bench --scenario long-decode --kv-codec ext-rabitq-3 --seeds 42,43,44 — the harness is the same code path." },
  ];
  return (
    <section className="block alt">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> What's in the REPL</div>
          <h2 className="display-2">Six things you'd<br /><span style={{ color: "var(--ink-3)" }}>otherwise have to build.</span></h2>
        </div>
        <div className="hl-grid reveal" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
          {items.map((it, i) => (
            <div key={i} className="hl-card">
              <div className="hl-tag mono">{it.eb}</div>
              <h3 className="hl-title">{it.t}</h3>
              <p className="hl-body">{it.b}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const App = () => (
  <PageShell active="chat">
    <Hero />
    <TerminalSection />
    <Features />
  </PageShell>
);
ReactDOM.createRoot(document.getElementById("app")).render(<App />);
