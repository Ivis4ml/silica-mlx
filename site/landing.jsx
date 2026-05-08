// landing.jsx — composes the index.html landing page

const HERO_NUMBERS = [
  { label: "Dense decode", to: 232,   suf: " tok/s",   sub: "Qwen3.5-27B-4bit · B=64 · 48 GB",     mult: "5.5×" },
  { label: "MoE decode",   to: 791.8, suf: " tok/s", dec: 1, sub: "Qwen3.5-35B-A3B-4bit · B=128 · 48 GB", mult: "4.2×" },
  { label: "Single-user",  to: 20,    suf: " tok/s",   sub: "B=1 · bandwidth-capped · D-022 closed",     mult: "—"   },
  { label: "Cycles",       to: 35,    suf: "",         sub: "Karpathy-style ledger · public retraction", mult: "9 KEEPs" },
];

const HERO_PILLS = [
  "Continuous batching", "Radix prefix cache", "Memory-budget admission",
  "Preempt + replay", "BlockTQ B=64 4-bit", "OpenAI HTTP server",
];

const Hero = () => {
  return (
    <section className="hero">
      <HeroBg />
      <div className="container">
        <div className="hero-eyebrow eyebrow reveal">
          <span className="dot"></span>
          <span>Status v1.7.35</span>
          <span className="subtle">·</span>
          <span>1.0 RC</span>
          <span className="subtle">·</span>
          <span className="accent">P-6 closed · P-8 shipped · M-9 cleared</span>
        </div>
        <h1 className="display-1 reveal reveal-d1">
          <span>Continuous-batching</span><br />
          <span>LLM serving,</span><br />
          <span className="hero-italic moment">native to <em>Apple Silicon</em>.</span>
        </h1>
        <p className="lede reveal reveal-d2 hero-lede">
          The vLLM scheduler patterns reimplemented from scratch on MLX&mdash;continuous batching,
          radix prefix cache, memory-budget admission ladder&mdash;with one integrated MLX-native runtime.
          On M5 Pro 48&nbsp;GB.
        </p>
        <div className="hero-ctas reveal reveal-d3">
          <a className="btn btn-primary" href="autoresearch.html">
            See the autoresearch loop<span className="arrow">→</span>
          </a>
          <a className="btn btn-ghost" href="#quickstart">
            <span>$</span> silica chat
          </a>
          <a className="btn btn-link" href="https://github.com/Ivis4ml/silica-mlx" target="_blank" rel="noreferrer">
            View on GitHub<span className="arrow">↗</span>
          </a>
        </div>
      </div>

      {/* the moment — oversized number stack */}
      <div className="container hero-moment reveal reveal-d4">
        <div className="moment-card">
          <div className="moment-eyebrow eyebrow"><span className="accent">●</span> Server-aggregate · M5 Pro 48&nbsp;GB</div>
          <div className="moment-rail">
            <div className="moment-cell">
              <div className="moment-num moment">
                <CountUp to={232} duration={1800} />
              </div>
              <div className="moment-unit">tok/s</div>
              <div className="moment-cap">Dense Qwen3.5-27B-4bit · B=64</div>
            </div>
            <div className="moment-divider"></div>
            <div className="moment-cell">
              <div className="moment-num moment">
                <CountUp to={791.8} decimals={1} duration={2200} />
              </div>
              <div className="moment-unit">tok/s</div>
              <div className="moment-cap">MoE Qwen3.5-35B-A3B-4bit · B=128</div>
            </div>
          </div>
          <div className="moment-foot mono">
            From 42.17 and 188.5 cycle-1 baselines · 35 experiments · 17 kernel attempts closed without a load-bearing E2E win
          </div>
        </div>
      </div>

      {/* batch flow visualization */}
      <div className="container reveal">
        <div className="bf-frame">
          <div className="bf-head">
            <div>
              <div className="eyebrow"><span className="accent">●</span> Continuous batching</div>
              <div className="bf-title">Many parallel users. One chip. One step per token.</div>
            </div>
            <div className="bf-stats mono">
              <span><strong className="tnum">8</strong> rows</span>
              <span><strong className="tnum">B=64</strong> peak</span>
              <span><strong className="tnum">~26</strong> GB resident</span>
            </div>
          </div>
          <BatchFlow rows={8} height={240} />
          <div className="bf-foot mono">
            admit → evict → preempt → reject · radix-trie prefix reuse · single code path
          </div>
        </div>
      </div>
    </section>
  );
};

const PillStrip = () => (
  <div className="pill-strip-wrap">
    <Marquee items={HERO_PILLS} speed={48} />
  </div>
);

const Highlights = () => {
  const items = [
    { tag: "P-2 · shipped", title: "vLLM-style scheduler.", body: "Continuous batching with admit → evict → preempt → reject. Single-request and batched share one code path.", pills: ["ContinuousBatcher", "MemoryBudgeter"] },
    { tag: "P-2 · shipped", title: "Radix prefix cache.", body: "Block-granular trie with a per-codec store seam. Block-aligned hits seed the row's KV cache.", pills: ["RadixPrefixCache", "PrefixBlockStore"] },
    { tag: "P-5 · shipped", title: "Native KV codec compression.", body: "BlockTQ B=64 4-bit ties the vqbench baseline at measurement precision (ΔPPL = +0.0016, three seeds).", pills: ["BlockTQ", "RaBitQ-1", "ExtRaBitQ"] },
    { tag: "P-3 · shipped", title: "Five model families, batched.", body: "Qwen3 dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B, Qwen3.5-MoE 35B-A3B, Gemma4-MoE 26B-A4B — parity-validated.", pills: ["0.6B → 35B-A3B", "256 × top-8 MoE"] },
    { tag: "P-3-C5", title: "Hybrid DeltaNet on the batched path.", body: "Recurrent-state snapshot + restore unlock RadixPrefixCache + Qwen3.5 hybrid cooperation end-to-end.", pills: ["state snapshot", "α-MVP slice"] },
    { tag: "P-4 · shipped", title: "Unified bench harness.", body: "15+ scenarios across five oracle types — smoke, B=1 parity, B>1 reference, teacher-forced, perplexity.", pills: ["JSONL + Markdown", "--all-kv-codecs"] },
    { tag: "P-6 · v1.7.28", title: "Performance phase closed.", body: "Every server-aggregate gate cleared 3.4–5.5× over cycle-1 baseline across 35 cycles. D-022 closed the small-B research line.", pills: ["232 tok/s", "~20 tok/s B=1", "D-022 closed"] },
    { tag: "P-8 · v1.7.33", title: "OpenAI-compatible HTTP server.", body: "silica serve boots a single-process FastAPI server fronting one model: chat/completions/models, SSE, X-Silica-Session-ID.", pills: ["silica serve", "X-Silica-Session-ID"] },
  ];
  return (
    <section className="block" id="shipped">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> What's shipped</div>
          <h2 className="display-2">Eight load-bearing pieces,<br /><span style={{ color: "var(--ink-3)" }}>behind frozen interfaces.</span></h2>
          <p>Every architectural decision and acceptance gate lives in <span className="mono">plans/PLAN.md</span>. Interfaces don't move; capabilities slot in below them.</p>
        </div>
        <div className="hl-grid reveal">
          {items.map((it, i) => (
            <div key={i} className="hl-card" style={{ "--i": i }}>
              <div className="hl-tag mono">{it.tag}</div>
              <h3 className="hl-title">{it.title}</h3>
              <p className="hl-body">{it.body}</p>
              <div className="hl-pills">
                {it.pills.map(p => <span key={p} className="hl-pill mono">{p}</span>)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
};

const Comparison = () => {
  const cols = [
    { id: "mlx", label: "mlx-lm", sub: "MLX" },
    { id: "vllm", label: "vLLM", sub: "CUDA" },
    { id: "sglang", label: "SGLang", sub: "CUDA" },
    { id: "silica", label: "silica-mlx", sub: "MLX", hi: true },
  ];
  const rows = [
    ["Continuous batching",       "no", "yes", "yes", "yes"],
    ["Radix prefix cache",        "no", "block-level", "yes", "yes"],
    ["Memory-budget admission",   "no", "yes", "yes", "yes"],
    ["Preempt + replay",          "no", "yes", "yes", "yes"],
    ["KV codec compression",      "no", "FP8 / INT8", "limited", "BlockTQ + RaBitQ"],
    ["Hybrid DeltaNet (batched)", "single-req", "no", "no", "yes"],
    ["MoE batched dispatch",      "single-req", "yes", "yes", "yes"],
    ["OpenAI HTTP server",        "no", "yes", "yes", "yes"],
    ["Speculative decoding",      "no", "yes", "yes", "foundation"],
    ["Per-expert MoE residency",  "no", "limited", "no", "planned"],
  ];
  const cell = (v) => {
    if (v === "yes") return <svg width="18" height="18" viewBox="0 0 16 16"><path d="M3.5 8.5l3 3 6-6.5" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/></svg>;
    if (v === "no") return <svg width="14" height="14" viewBox="0 0 16 16"><path d="M3 3l10 10M13 3L3 13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/></svg>;
    if (v === "planned") return <span className="cmp-tag">planned</span>;
    return <span className="cmp-text mono">{v}</span>;
  };
  return (
    <section className="block alt" id="compare">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> The niche</div>
          <h2 className="display-2">vLLM-style architecture,<br /><span style={{ color: "var(--ink-3)" }}>native to Apple Silicon.</span></h2>
          <p>vLLM and SGLang are CUDA-first and don't run on Apple Silicon. mlx-lm is single-request and solves a different problem. silica-mlx fills the gap with one MLX-native runtime &mdash; not a port.</p>
        </div>
        <div className="cmp-card reveal">
          <div className="cmp-grid">
            <div className="cmp-corner"></div>
            {cols.map(c => (
              <div key={c.id} className={"cmp-col" + (c.hi ? " cmp-col-hi" : "")}>
                <div className="cmp-col-label">{c.label}</div>
                <div className="cmp-col-sub mono">{c.sub}</div>
              </div>
            ))}
            {rows.map((r, i) => (
              <React.Fragment key={i}>
                <div className="cmp-row-label">{r[0]}</div>
                {r.slice(1).map((v, j) => (
                  <div key={j} className={"cmp-cell" + (cols[j].hi ? " cmp-cell-hi" : "") + " cmp-" + (v === "yes" ? "yes" : v === "no" ? "no" : v === "planned" ? "plan" : "txt")}>
                    {cell(v)}
                  </div>
                ))}
              </React.Fragment>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
};

const CodecBars = () => {
  const codecs = [
    { name: "Identity (fp16)",     bits: 16,   savings: "1.0×",  ppl: "0.000",  sub: "baseline" },
    { name: "BlockTQ B=32 4-bit",   bits: 4.5,  savings: "3.5×",  ppl: "+0.002", sub: "block turbo-quant" },
    { name: "BlockTQ B=64 4-bit",   bits: 4.25, savings: "3.8×",  ppl: "+0.002", sub: "default · ties vqbench at measurement precision", hi: true },
    { name: "ExtRaBitQ 4-bit",      bits: 4.5,  savings: "3.5×",  ppl: "+0.004", sub: "extended rotation grid" },
    { name: "ExtRaBitQ 3-bit",      bits: 3.5,  savings: "4.6×",  ppl: "+0.018", sub: "extended rotation grid" },
    { name: "ExtRaBitQ 2-bit",      bits: 2.5,  savings: "6.4×",  ppl: "+0.092", sub: "aggressive · admission headroom" },
    { name: "RaBitQ-1",             bits: 1.5,  savings: "10.7×", ppl: "+0.318", sub: "1-bit · maximum compression" },
  ];
  const max = 16;
  return (
    <section className="block" id="codec">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> P-5 · KV codec stack</div>
          <h2 className="display-2">Lossless<br /><span style={{ color: "var(--ink-3)" }}>at measurement precision.</span></h2>
          <p>BlockTQ B=64 4-bit matches the vqbench baseline on Qwen3.5-4B WikiText-2: ΔPPL = +0.0016 across three seeds. The codec lives at the prefix-block store; admission headroom turns straight into more admitted requests.</p>
        </div>
        <div className="codec-card reveal">
          {codecs.map(c => (
            <div key={c.name} className={"codec-row" + (c.hi ? " codec-row-hi" : "")}>
              <div className="codec-name">
                <div className="codec-name-t">{c.name}{c.hi && <span className="codec-badge mono">default</span>}</div>
                <div className="codec-name-sub">{c.sub}</div>
              </div>
              <div className="codec-bar-wrap">
                <div className="codec-bar" style={{ "--w": `${(c.bits / max) * 100}%` }}>
                  <span className="codec-bar-label mono">{c.bits.toFixed(2)}<span className="subtle"> bpe</span></span>
                </div>
              </div>
              <div className="codec-num mono">{c.savings}</div>
              <div className={"codec-num mono " + (parseFloat(c.ppl) < 0.01 ? "codec-good" : "codec-meh")}>{c.ppl}</div>
            </div>
          ))}
          <div className="codec-foot mono">
            <span>Qwen3.5-4B · WikiText-2 · seeds (42, 43, 44)</span>
            <span>VectorCodec[P] · side-level since P-5-A.0.4</span>
          </div>
        </div>
      </div>
    </section>
  );
};

const QuickStart = () => {
  const [tab, setTab] = React.useState("repl");
  const [copied, setCopied] = React.useState(false);
  const tabs = {
    repl: { label: "Chat REPL", lang: "bash", code: `$ silica chat --model Qwen/Qwen3-0.6B \\
    --system "You are a concise assistant." \\
    --temperature 0.7 --top-p 0.9 --max-tokens 256

› Explain TTFT in one sentence.
TTFT (time-to-first-token) is the latency from request
arrival to the model emitting its first generated token.

[ttft=25.1ms prefill=596.9tok/s decode=151.4tok/s
 resident_kv=29.4MB peak=1261.5MB logical_kv=29.4MB
 prompt=15 out=64 wall=0.44s finish=max_tokens]`,
    },
    serve: { label: "OpenAI server", lang: "bash", code: `$ silica serve --model mlx-community/Qwen3.5-27B-4bit \\
    --host 0.0.0.0 --port 8000 \\
    --max-batch-size 64 --bearer-token \$SILICA_TOKEN

# in another shell — same surface as openai-python
$ openai api chat.completions.create \\
    -m Qwen3.5-27B-4bit \\
    --message user "haiku about silicon"

[FastAPI · SSE streaming · X-Silica-Session-ID prefix reuse
 token-bucket rate limit · OpenAI-shaped error envelope]`,
    },
    py: { label: "Engine.generate", lang: "python", code: `from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

tok = adapter.tokenizer()
params = SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=128,
    stop_token_ids=tuple(tok.eos_token_ids or ()),
)

ids = list(engine.generate("Write a haiku about silicon.", params))
print(tok.decode(ids))
print(engine.metrics.snapshot())`,
    },
    batch: { label: "Continuous batching", lang: "python", code: `from silica import Engine
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

pc = RadixPrefixCache(
    block_size=16,
    store=SyntheticPrefixBlockStore(block_size=16),
)

for event in engine.generate_batch(
    prompts, params, max_batch_size=8, prefix_cache=pc,
):
    ...   # event.kind in {"token", "done", "aborted"}`,
    },
  };
  const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  const highlight = (code, lang) => {
    if (lang === "python") {
      const pat = /(#[^\n]*)|("[^"\n]*"|'[^'\n]*')|\b(from|import|for|in|print|list|tuple|return|def|class|if|else|with|as|None|True|False)\b|\b(Engine|SamplingParams|RadixPrefixCache|SyntheticPrefixBlockStore|adapter_for_repo)\b|\b(\d+(?:\.\d+)?)\b/g;
      return escape(code).replace(pat, (m, cm, st, kw, ty, nm) => {
        if (cm) return `<span class="c-cm">${cm}</span>`;
        if (st) return `<span class="c-st">${st}</span>`;
        if (kw) return `<span class="c-kw">${kw}</span>`;
        if (ty) return `<span class="c-ty">${ty}</span>`;
        if (nm) return `<span class="c-nm">${nm}</span>`;
        return m;
      });
    }
    const pat = /(^\$.*$)|(#[^\n]*)|(\[[^\]\n]*\])|("[^"\n]*"|'[^'\n]*')|(--?[a-zA-Z][a-zA-Z0-9-]*)|(\b\d+(?:\.\d+)?(?:ms|MB|GB|tok\/s|s)?\b)/gm;
    return escape(code).replace(pat, (m, pr, cm, br, st, fl, nm) => {
      if (pr) return `<span class="c-pr">${pr}</span>`;
      if (cm) return `<span class="c-cm">${cm}</span>`;
      if (br) return `<span class="c-br">${br}</span>`;
      if (st) return `<span class="c-st">${st}</span>`;
      if (fl) return `<span class="c-fl">${fl}</span>`;
      if (nm) return `<span class="c-nm">${nm}</span>`;
      return m;
    });
  };
  const copy = () => {
    navigator.clipboard?.writeText(tabs[tab].code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };
  return (
    <section className="block alt" id="quickstart">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> Quickstart</div>
          <h2 className="display-2">Four entry points.<br /><span style={{ color: "var(--ink-3)" }}>Same engine.</span></h2>
          <p>The CLI, the OpenAI-compatible HTTP server, the Python iterator, and the batched event stream all drive the same scheduler.</p>
        </div>
        <div className="cs-card reveal">
          <div className="cs-head">
            <div className="cs-tabs">
              {Object.entries(tabs).map(([k, t]) => (
                <button key={k} className={"cs-tab" + (tab === k ? " on" : "")} onClick={() => setTab(k)}>{t.label}</button>
              ))}
            </div>
            <div className="cs-actions">
              <span className="cs-lang mono">{tabs[tab].lang}</span>
              <button className="cs-copy" onClick={copy}>{copied ? "✓ Copied" : "⎘ Copy"}</button>
            </div>
          </div>
          <pre className="cs-code mono"><code dangerouslySetInnerHTML={{ __html: highlight(tabs[tab].code, tabs[tab].lang) }} /></pre>
        </div>
      </div>
    </section>
  );
};

const NextDoors = () => {
  const doors = [
    { href: "autoresearch.html", eyebrow: "P-6 · 35 cycles", title: "The autoresearch loop", body: "9 KEEPs, 17 closed kernel attempts, one public retraction. From 42 to 232 tok/s on dense — and how we know.", cta: "Read the ledger" },
    { href: "architecture.html", eyebrow: "Layered stack", title: "Architecture", body: "Seven layers, frozen Protocol seams. ContinuousBatcher, MemoryBudgeter, RadixPrefixCache, VectorCodec[P], ModelAdapter.", cta: "See the layers" },
    { href: "chat.html", eyebrow: "silica chat", title: "Chat REPL", body: "A REPL with serving instrumentation baked in. Live decode tok/s, prefix-hit fraction, three-axis thinking model.", cta: "Open the REPL" },
    { href: "changelog.html", eyebrow: "Roadmap", title: "Changelog", body: "P-6 closed. P-8 shipped. M-9 cleared. What's coming next on the path to 1.0.", cta: "View the changelog" },
  ];
  return (
    <section className="block">
      <div className="container">
        <div className="section-head reveal">
          <div className="eyebrow"><span className="accent">●</span> Keep reading</div>
          <h2 className="display-2">Pick a door.</h2>
        </div>
        <div className="doors reveal">
          {doors.map(d => (
            <a key={d.href} href={d.href} className="door">
              <div className="door-eyebrow mono">{d.eyebrow}</div>
              <div className="door-title">{d.title}</div>
              <div className="door-body">{d.body}</div>
              <div className="door-cta">{d.cta}<span className="arrow">→</span></div>
            </a>
          ))}
        </div>
      </div>
    </section>
  );
};

const App = () => (
  <PageShell active="overview">
    <Hero />
    <PillStrip />
    <Highlights />
    <Comparison />
    <CodecBars />
    <QuickStart />
    <NextDoors />
  </PageShell>
);

ReactDOM.createRoot(document.getElementById("app")).render(<App />);
