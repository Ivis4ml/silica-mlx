// Code snippets section — three tabs: CLI, Python single, Python batch

const CodeSnippets = () => {
  const [tab, setTab] = React.useState("repl");
  const [copied, setCopied] = React.useState(false);

  const tabs = {
    repl: {
      label: "Chat REPL",
      lang: "bash",
      code: `$ python scripts/chat.py --model Qwen/Qwen3-0.6B \\
    --system "You are a concise assistant." \\
    --temperature 0.7 --top-p 0.9 --max-tokens 256

> Explain TTFT in one sentence.
TTFT (time-to-first-token) is the latency from request
arrival to the model emitting its first generated token.

[ttft=25.1ms prefill=596.9tok/s decode=151.4tok/s
 resident_kv=29.4MB peak=1261.5MB logical_kv=29.4MB
 prompt=15 out=64 wall=0.44s finish=max_tokens]`,
    },
    single: {
      label: "Engine.generate",
      lang: "python",
      code: `from silica import Engine
from silica.core.sampling import SamplingParams
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

tokenizer = adapter.tokenizer()
params = SamplingParams(
    temperature=0.7, top_p=0.9, max_tokens=128,
    stop_token_ids=tuple(tokenizer.eos_token_ids or ()),
)

token_ids = list(engine.generate("Write a haiku about silicon.", params))
print(tokenizer.decode(token_ids))
print(engine.metrics.snapshot())`,
    },
    batch: {
      label: "Continuous batching",
      lang: "python",
      code: `from silica import Engine
from silica.kvcache.prefix import RadixPrefixCache
from silica.kvcache.store import SyntheticPrefixBlockStore
from silica.models.factory import adapter_for_repo

adapter, kv = adapter_for_repo("Qwen/Qwen3-0.6B")
engine = Engine(adapter, kv)

block_size = 16
pc = RadixPrefixCache(
    block_size=block_size,
    store=SyntheticPrefixBlockStore(block_size=block_size),
)

for event in engine.generate_batch(
    prompts, params, max_batch_size=8, prefix_cache=pc,
):
    ...   # event.kind in {"token", "done", "aborted"}`,
    },
  };

  const highlight = (code, lang) => {
    // Token-based highlighter — single pass so spans don't nest.
    const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    if (lang === "python") {
      const pattern = /(#[^\n]*)|("[^"\n]*"|'[^'\n]*')|\b(from|import|for|in|print|list|tuple|return|def|class|if|else|with|as)\b|\b(Engine|SamplingParams|RadixPrefixCache|SyntheticPrefixBlockStore|adapter_for_repo)\b|\b(\d+(?:\.\d+)?)\b/g;
      return escape(code).replace(pattern, (m, cm, st, kw, ty, nm) => {
        if (cm) return '<span class="c-cm">' + cm + '</span>';
        if (st) return '<span class="c-st">' + st + '</span>';
        if (kw) return '<span class="c-kw">' + kw + '</span>';
        if (ty) return '<span class="c-ty">' + ty + '</span>';
        if (nm) return '<span class="c-nm">' + nm + '</span>';
        return m;
      });
    }
    // bash — single pass, line-by-line so prompts work
    const bashPattern = /(^\$.*$)|(\[[^\]\n]*\])|("[^"\n]*"|'[^'\n]*')|(--?[a-zA-Z][a-zA-Z0-9-]*)|(\b\d+(?:\.\d+)?(?:ms|MB|tok\/s|s)?\b)/gm;
    return escape(code).replace(bashPattern, (m, pr, cm, st, fl, nm) => {
      if (pr) return '<span class="c-pr">' + pr + '</span>';
      if (cm) return '<span class="c-cm">' + cm + '</span>';
      if (st) return '<span class="c-st">' + st + '</span>';
      if (fl) return '<span class="c-fl">' + fl + '</span>';
      if (nm) return '<span class="c-nm">' + nm + '</span>';
      return m;
    });
  };

  const copy = () => {
    navigator.clipboard?.writeText(tabs[tab].code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1400);
  };

  return (
    <section className="block" id="quickstart" style={{ background: "var(--bg-sunken)" }}>
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">Quickstart</div>
          <h2>Three entry points. Same engine.</h2>
          <p>The CLI, the Python iterator, and the batched event stream all drive the same scheduler. Single-shot, multi-turn, and concurrent share one code path.</p>
        </div>

        <div className="cs-card">
          <div className="cs-head">
            <div className="cs-tabs">
              {Object.entries(tabs).map(([k, t]) => (
                <button
                  key={k}
                  className={"cs-tab" + (tab === k ? " cs-tab-on" : "")}
                  onClick={() => setTab(k)}
                >{t.label}</button>
              ))}
            </div>
            <div className="cs-actions">
              <span className="cs-lang mono">{tabs[tab].lang}</span>
              <button className="cs-copy" onClick={copy}>
                <Icon name="copy" size={12} />
                {copied ? "Copied" : "Copy"}
              </button>
            </div>
          </div>
          <pre className="cs-code mono"><code dangerouslySetInnerHTML={{ __html: highlight(tabs[tab].code, tabs[tab].lang) }} /></pre>
        </div>
      </div>

      <style>{`
        .cs-card {
          background: var(--code-bg);
          border-radius: var(--radius);
          overflow: hidden;
          box-shadow: 0 1px 2px rgba(0,0,0,0.1), 0 16px 40px -16px rgba(0,0,0,0.18), 0 0 0 1px rgba(255,255,255,0.04);
        }
        .cs-head {
          display: flex; justify-content: space-between; align-items: center;
          padding: 10px 14px;
          background: rgba(255,255,255,0.02);
          border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .cs-tabs { display: flex; gap: 2px; }
        .cs-tab {
          background: transparent; border: none;
          color: rgba(255,255,255,0.55);
          font-size: 12.5px;
          padding: 6px 12px;
          border-radius: 6px;
          cursor: pointer;
          font-family: inherit;
        }
        .cs-tab:hover { color: rgba(255,255,255,0.85); }
        .cs-tab-on {
          color: #fff;
          background: rgba(255,255,255,0.08);
        }
        .cs-actions { display: flex; gap: 10px; align-items: center; }
        .cs-lang {
          font-size: 11px; color: rgba(255,255,255,0.4);
          text-transform: uppercase; letter-spacing: 0.06em;
        }
        .cs-copy {
          display: inline-flex; align-items: center; gap: 5px;
          background: transparent; border: 1px solid rgba(255,255,255,0.12);
          color: rgba(255,255,255,0.7);
          padding: 4px 10px; border-radius: 6px;
          font-size: 11.5px; cursor: pointer;
          font-family: inherit;
        }
        .cs-copy:hover { background: rgba(255,255,255,0.06); color: #fff; }
        .cs-code {
          margin: 0;
          padding: 22px 26px;
          color: var(--code-fg);
          font-size: 13px;
          line-height: 1.65;
          overflow-x: auto;
          font-family: var(--font-mono);
        }
        .cs-code .c-kw { color: #ff7ab2; }
        .cs-code .c-ty { color: #67d4ff; }
        .cs-code .c-st { color: #a5e07a; }
        .cs-code .c-cm { color: #6e6e73; font-style: italic; }
        .cs-code .c-nm { color: #f5b06b; }
        .cs-code .c-pr { color: #6e6e73; }
        .cs-code .c-fl { color: #67d4ff; }
      `}</style>
    </section>
  );
};

window.CodeSnippets = CodeSnippets;
