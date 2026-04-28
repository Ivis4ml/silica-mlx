// Chat REPL showcase — recreation of `silica chat` terminal output

const ChatRepl = () => {
  const [tick, setTick] = React.useState(0);
  React.useEffect(() => {
    const id = setInterval(() => setTick(t => (t + 1) % 60), 500);
    return () => clearInterval(id);
  }, []);

  const cursor = tick % 2 === 0 ? "█" : " ";

  // toolbar fields from screenshot
  const toolbar = [
    { k: "state", v: "idle", c: "tb-mut" },
    { k: "model", v: "Qwen3.5-35B-A3B-4bit", c: "tb-mut" },
    { k: "MLX", v: "", c: "tb-acc" },
    { k: "tok/s", v: "57.7", c: "tb-acc" },
    { k: "tokens", v: "35/1024", c: "tb-mut" },
    { k: "ttft", v: "389.3ms", c: "tb-acc" },
    { k: "peak", v: "21.65GB", c: "tb-mut" },
    { k: "kv", v: "3.3MB", c: "tb-mut" },
    { k: "kv_log", v: "3.3MB", c: "tb-mut" },
    { k: "compr", v: "—", c: "tb-mut" },
    { k: "prefix_hit", v: "92/116", c: "tb-ok" },
    { k: "turn", v: "3", c: "tb-mut" },
    { k: "finish", v: "stop_token", c: "tb-mut" },
  ];

  return (
    <section className="block" id="chat" style={{ background: "var(--bg-sunken)" }}>
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">silica chat</div>
          <h2>A REPL with real serving instrumentation, baked in.</h2>
          <p>Persistent bottom toolbar shows live engine state — decode tok/s, KV residency, prefix-hit fraction, finish reason — every turn, every model. Three-axis thinking model: <span className="mono">thinking_mode</span> · <span className="mono">thinking</span> · <span className="mono">thinking_history</span> are independent.</p>
        </div>

        <div className="repl-wrap">
          <div className="repl-card">
            <div className="repl-chrome">
              <span className="repl-dot repl-dot-r"></span>
              <span className="repl-dot repl-dot-y"></span>
              <span className="repl-dot repl-dot-g"></span>
              <span className="repl-title mono">silica chat — Qwen3.5-35B-A3B-4bit</span>
            </div>
            <pre className="repl-body mono" dangerouslySetInnerHTML={{ __html: `<span class="repl-shell">(silica-mlx-venv) (base) → </span><span class="repl-cyan">silica-mlx</span> <span class="repl-mag">git:(</span><span class="repl-yellow">sonnet</span><span class="repl-mag">)</span> <span class="repl-red">✗</span> silica chat --model mlx-community/Qwen3.5-35B-A3B-4bit
<span class="repl-dim">Loading mlx-community/Qwen3.5-35B-A3B-4bit ...</span>
<span class="repl-dim">Fetching 14 files: 100%|████████████████████████| 14/14 [00:00&lt;00:00, 203184.28it/s]</span>
<span class="repl-dim">Download complete: 0.00B [00:00, 7B/s]</span>
<span class="repl-cyan">silica chat — Qwen3.5-35B-A3B-4bit (fp16). Type /help for commands, /exit to quit.</span>

<span class="repl-green">You ›</span> /config thinking_mode = off
<span class="repl-dim">config: thinking_mode = False</span>
<span class="repl-green">You ›</span> Calculate 9.9 - 9.11
<span class="repl-yellow">silica ›</span> -0.21

<span class="repl-green">You ›</span> How many 'r' in strawberry
<span class="repl-yellow">silica ›</span> 3

<span class="repl-green">You ›</span> write a 20 words poem
<span class="repl-yellow">silica ›</span> Soft wind whispers through the trees,
        Green leaves dance beneath the breeze,
        Birds sing songs of joy and glee,
        As the world wakes up to see.

<span class="repl-green">You ›</span> ${cursor}` }}>
            </pre>
            <div className="repl-toolbar mono">
              {toolbar.map((f, i) => (
                <span key={i} className={"tb " + f.c}>
                  {f.k}{f.v && <>=<span className="tb-v">{f.v}</span></>}
                </span>
              ))}
            </div>
          </div>

          <div className="repl-side">
            <div className="repl-side-h">
              <Icon name="spark" size={14} />
              <span>Slash commands</span>
            </div>
            <ul className="repl-cmds">
              <li><span className="mono">/config thinking_mode=off</span><em>thread enable_thinking=False to chat template</em></li>
              <li><span className="mono">/continue</span><em>extend last turn when it stopped at max_tokens</em></li>
              <li><span className="mono">/regenerate</span><em>redo previous turn with a fresh sample</em></li>
              <li><span className="mono">/model &lt;repo&gt; --keep-history</span><em>swap models, re-tokenise stored text</em></li>
              <li><span className="mono">/save · /load</span><em>persist or restore conversation as JSON</em></li>
              <li><span className="mono">/showcase</span><em>one-paragraph session narrative</em></li>
              <li><span className="mono">/expand</span><em>reprint collapsed &lt;think&gt; content</em></li>
            </ul>

            <div className="repl-side-h" style={{ marginTop: 22 }}>
              <Icon name="sliders" size={14} />
              <span>Three-axis thinking</span>
            </div>
            <div className="repl-axes">
              <div className="repl-axis">
                <div className="repl-axis-k mono">thinking_mode</div>
                <div className="repl-axis-v">model side · enable_thinking flag</div>
              </div>
              <div className="repl-axis">
                <div className="repl-axis-k mono">thinking</div>
                <div className="repl-axis-v">display side · auto · show · hidden</div>
              </div>
              <div className="repl-axis">
                <div className="repl-axis-k mono">thinking_history</div>
                <div className="repl-axis-v">history side · strip · keep</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <style>{`
        .repl-wrap {
          display: grid;
          grid-template-columns: 1.6fr 1fr;
          gap: 24px;
        }
        @media (max-width: 920px) { .repl-wrap { grid-template-columns: 1fr; } }
        .repl-card {
          background: #0b0b0d;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 1px 2px rgba(0,0,0,0.1), 0 24px 56px -24px rgba(0,0,0,0.25), 0 0 0 1px rgba(255,255,255,0.05);
          display: flex; flex-direction: column;
          min-height: 520px;
        }
        .repl-chrome {
          display: flex; align-items: center; gap: 8px;
          padding: 10px 14px;
          background: rgba(255,255,255,0.03);
          border-bottom: 1px solid rgba(255,255,255,0.06);
        }
        .repl-dot { width: 11px; height: 11px; border-radius: 50%; }
        .repl-dot-r { background: #ff5f57; }
        .repl-dot-y { background: #febc2e; }
        .repl-dot-g { background: #28c840; }
        .repl-title {
          margin-left: 12px;
          font-size: 11.5px;
          color: rgba(255,255,255,0.5);
        }
        .repl-body {
          margin: 0;
          padding: 18px 22px;
          color: #e7e7ea;
          font-size: 12.5px;
          line-height: 1.55;
          flex: 1;
          white-space: pre-wrap;
          word-break: break-word;
          background: #0b0b0d;
        }
        .repl-shell { color: #67d4ff; }
        .repl-cyan { color: #67d4ff; }
        .repl-mag { color: #ff7ab2; }
        .repl-yellow { color: #f5b06b; font-weight: 500; }
        .repl-red { color: #ff5e5e; }
        .repl-green { color: #6ed27a; font-weight: 600; }
        .repl-dim { color: #6e6e73; }

        .repl-toolbar {
          display: flex; flex-wrap: wrap;
          gap: 4px 14px;
          padding: 10px 14px;
          background: #1c1c1f;
          border-top: 1px solid rgba(255,255,255,0.06);
          font-size: 10.5px;
          color: rgba(255,255,255,0.55);
          letter-spacing: 0.01em;
        }
        .tb-acc { color: #67d4ff; }
        .tb-acc .tb-v { color: #67d4ff; font-weight: 600; }
        .tb-ok .tb-v { color: #6ed27a; font-weight: 600; }
        .tb-mut .tb-v { color: #d6d6d9; }

        .repl-side {
          background: var(--bg-elev);
          border-radius: 12px;
          padding: 20px 22px;
          box-shadow: var(--shadow-sm);
        }
        .repl-side-h {
          display: flex; align-items: center; gap: 8px;
          font-size: 12px;
          font-weight: 600;
          color: var(--ink-3);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          margin-bottom: 14px;
        }
        .repl-cmds {
          list-style: none; padding: 0; margin: 0;
          display: flex; flex-direction: column; gap: 10px;
        }
        .repl-cmds li {
          display: flex; flex-direction: column; gap: 2px;
          padding-bottom: 10px;
          border-bottom: 1px solid var(--rule-2);
        }
        .repl-cmds li:last-child { border-bottom: none; padding-bottom: 0; }
        .repl-cmds .mono {
          font-size: 12px;
          color: var(--accent);
          font-weight: 500;
        }
        .repl-cmds em {
          font-style: normal;
          font-size: 12.5px;
          color: var(--ink-2);
        }
        .repl-axes { display: flex; flex-direction: column; gap: 8px; }
        .repl-axis {
          padding: 10px 12px;
          background: var(--bg-sunken);
          border-radius: 8px;
          border: 1px solid var(--rule);
        }
        .repl-axis-k {
          font-size: 11.5px;
          color: var(--accent);
          font-weight: 600;
          margin-bottom: 2px;
        }
        .repl-axis-v {
          font-size: 12px;
          color: var(--ink-2);
        }
      `}</style>
    </section>
  );
};

window.ChatRepl = ChatRepl;
