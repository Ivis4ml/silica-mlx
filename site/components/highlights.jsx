// Highlights — what's shipped grid

const Highlights = () => {
  const items = [
    {
      tag: "P-2 · shipped",
      title: "vLLM-core scheduler.",
      body: "Continuous batching with admit → evict → preempt → reject. Single-request and batched share one code path.",
      detail: ["ContinuousBatcher", "MemoryBudgeter", "preempt + replay"],
    },
    {
      tag: "P-2 · shipped",
      title: "Radix prefix cache.",
      body: "Block-granular trie with a per-codec store seam. Block-aligned hits seed the row's KV cache; misses chunk back into the tree on termination.",
      detail: ["RadixPrefixCache", "PrefixBlockStore"],
    },
    {
      tag: "P-5 · shipped",
      title: "Native KV codec compression.",
      body: "BlockTQ B=64 4-bit ties the vqbench baseline at measurement precision (ΔPPL = +0.0016 on Qwen3.5-4B WikiText-2, three seeds).",
      detail: ["BlockTQ", "RaBitQ-1", "ExtRaBitQ 2/3/4-bit"],
    },
    {
      tag: "P-3 · shipped",
      title: "Five model families, batched.",
      body: "Qwen3 dense, Qwen3.5 hybrid DeltaNet, Gemma4-31B dense, Qwen3.5-MoE 35B-A3B, Gemma4-MoE 26B-A4B — all parity-validated against direct mlx-lm batched references.",
      detail: ["0.6B → 35B-A3B", "256 × top-8 MoE"],
    },
    {
      tag: "P-3-C5",
      title: "Hybrid DeltaNet on the batched path.",
      body: "Recurrent-state snapshot and restore unlock RadixPrefixCache + Qwen3.5 hybrid cooperation end-to-end on real Qwen3.5-0.8B.",
      detail: ["state snapshot", "α-MVP slice prefill"],
    },
    {
      tag: "P-4 · shipped",
      title: "Unified bench harness.",
      body: "15+ scenarios across five oracle types — smoke, B=1 parity, B>1 reference, teacher-forced argmax, WikiText-2 perplexity. Optional vqbench cross-check column.",
      detail: ["JSONL + Markdown", "--all-kv-codecs sweep"],
    },
    {
      tag: "P-6 · v1.7.28",
      title: "Performance phase closed (D-022 closed).",
      body: "Every P-6 server-aggregate acceptance gate cleared 3.4-5.5× over the cycle-1 baseline across a 35-cycle autoresearch loop. Dense Qwen3.5-27B-4bit hits 232 tok/s at B=64; MoE 35B-A3B hits 791.8 tok/s at B=128. Per-row speed still sits near ~20 tok/s at B=1, bandwidth-capped. D-022 closed with β narrow-scope, γ tiny-gain, and δ real-compute-overhead negatives.",
      detail: ["232 tok/s aggregate", "~20 tok/s B=1", "D-022 closed"],
    },
    {
      tag: "P-8 · v1.7.33",
      title: "OpenAI-compatible HTTP server.",
      body: "silica serve boots a single-process FastAPI server fronting one loaded model: chat / completions / models endpoints, SSE streaming, X-Silica-Session-ID cross-request prefix reuse, bearer auth + token-bucket rate limit, OpenAI-shaped error envelope. silica.llm.LLM gives mlx-lm-style ergonomics over the same engine. M-9 milestone cleared.",
      detail: ["silica serve", "X-Silica-Session-ID", "M-9 cleared"],
    },
  ];

  return (
    <section className="block" id="shipped">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">What's shipped</div>
          <h2>Eight load-bearing pieces, behind frozen interfaces.</h2>
          <p>Every architectural decision and acceptance gate lives in <span className="mono">plans/PLAN.md</span>. The interfaces don't move; capabilities slot in below them.</p>
        </div>

        <div className="hl-grid">
          {items.map((it, i) => (
            <div key={i} className="hl-card">
              <div className="hl-tag mono">{it.tag}</div>
              <h3 className="hl-title">{it.title}</h3>
              <p className="hl-body">{it.body}</p>
              <div className="hl-detail">
                {it.detail.map(d => <span key={d} className="hl-pill mono">{d}</span>)}
              </div>
            </div>
          ))}
        </div>
      </div>

      <style>{`
        .hl-grid {
          display: grid;
          grid-template-columns: repeat(3, 1fr);
          gap: 1px;
          background: var(--rule);
          border-radius: var(--radius);
          overflow: hidden;
          border: 1px solid var(--rule);
        }
        @media (max-width: 920px) { .hl-grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 600px) { .hl-grid { grid-template-columns: 1fr; } }
        .hl-card {
          padding: 28px 28px 26px;
          background: var(--bg-elev);
          display: flex; flex-direction: column;
          min-height: 240px;
        }
        .hl-tag {
          font-size: 11px;
          color: var(--accent);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 18px;
          font-weight: 600;
        }
        .hl-title {
          font-size: 18px;
          letter-spacing: -0.018em;
          font-weight: 600;
          margin: 0 0 10px;
          line-height: 1.25;
          color: var(--ink);
        }
        .hl-body {
          font-size: 14px;
          color: var(--ink-2);
          line-height: 1.55;
          margin: 0 0 18px;
          flex: 1;
        }
        .hl-detail {
          display: flex; flex-wrap: wrap; gap: 6px;
        }
        .hl-pill {
          font-size: 11px;
          padding: 3px 9px;
          border-radius: 5px;
          background: var(--bg-sunken);
          color: var(--ink-3);
          border: 1px solid var(--rule);
        }
      `}</style>
    </section>
  );
};

window.Highlights = Highlights;
