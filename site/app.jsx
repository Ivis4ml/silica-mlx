// Main app — composes all sections, handles tweaks, theme, accent.

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light",
  "accent": "blue",
  "font": "inter",
  "density": "roomy",
  "hero": "metrics"
}/*EDITMODE-END*/;

const ACCENTS = {
  blue:     { light: "#0066ff", dark: "#4d94ff", soft_l: "rgba(0,102,255,0.08)", soft_d: "rgba(77,148,255,0.12)", deep_l: "#0040d6", deep_d: "#80b3ff" },
  graphite: { light: "#1c1c1e", dark: "#e0e0e3", soft_l: "rgba(28,28,30,0.07)",  soft_d: "rgba(255,255,255,0.08)", deep_l: "#000",     deep_d: "#fff" },
  violet:   { light: "#6e3aff", dark: "#9d7dff", soft_l: "rgba(110,58,255,0.08)",soft_d: "rgba(157,125,255,0.14)", deep_l: "#4a1bcf", deep_d: "#bba3ff" },
  ember:    { light: "#d6510a", dark: "#ff8c4a", soft_l: "rgba(214,81,10,0.08)", soft_d: "rgba(255,140,74,0.14)", deep_l: "#a83b00", deep_d: "#ffb38c" },
  moss:     { light: "#1f7a3b", dark: "#4cc26b", soft_l: "rgba(31,122,59,0.08)", soft_d: "rgba(76,194,107,0.14)", deep_l: "#0e5224", deep_d: "#85e09a" },
};

const useTweaks = (defaults) => {
  const [tweaks, setTweaks] = React.useState(defaults);
  const setTweak = React.useCallback((k, v) => {
    setTweaks(prev => {
      const next = typeof k === "object" ? { ...prev, ...k } : { ...prev, [k]: v };
      try { window.parent.postMessage({ type: "__edit_mode_set_keys", edits: typeof k === "object" ? k : { [k]: v } }, "*"); } catch (e) {}
      return next;
    });
  }, []);
  return [tweaks, setTweak];
};

const Hero = ({ tweaks }) => {
  if (tweaks.hero === "type") {
    return (
      <section className="hero">
        <div className="container">
          <div className="hero-eyebrow"><span className="dot"></span>Status v1.7.35 · silica-mlx 1.0 release candidate · P-6 closed · P-8 OpenAI HTTP server shipped · M-9 cleared</div>
          <h1>
            Continuous-batching<br/>
            LLM serving,<br/>
            <em>native to Apple Silicon.</em>
          </h1>
          <p className="hero-sub">
            silica-mlx reimplements vLLM's scheduler patterns — continuous batching, memory-budget admission,
            preempt+replay, radix prefix cache — on MLX. silica-mlx does not depend on vLLM; vLLM is the architectural reference. The 35-cycle P-6
            autoresearch loop closed in May 2026 with every <em>server-throughput</em> acceptance
            gate cleared 3.4-5.5× over the cycle-1 baseline: 232 tok/s on dense Qwen3.5-27B-4bit
            at B=64, 791.8 tok/s on MoE Qwen3.5-35B-A3B-4bit at B=128. Single-user latency at B=1
            still sits ~20 tok/s, bandwidth-capped; D-022 closed that research line with β/γ/δ negatives
            and ≤0.6% recoverable Python-hygiene headroom. P-8 closed at v1.7.33: <code>silica serve</code> boots
            a single-process FastAPI server with the openai-client surface (chat / completions / models),
            <code>X-Silica-Session-ID</code> cross-request prefix reuse, bearer auth + token-bucket rate limit.
          </p>
          <div className="hero-ctas">
            <a className="btn btn-primary" href="#quickstart">Quickstart<Icon name="arrow" size={14} className="arrow" /></a>
            <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx" target="_blank" rel="noreferrer"><Icon name="github" size={14} />View on GitHub</a>
          </div>
        </div>
      </section>
    );
  }

  return (
    <section className="hero">
      <div className="container">
        <div className="hero-eyebrow"><span className="dot"></span>Status v1.7.35 · silica-mlx 1.0 release candidate · P-6 closed · P-8 OpenAI HTTP server shipped · M-9 cleared</div>
        <h1>Continuous-batching LLM serving, <em>native to Apple Silicon.</em></h1>
        <p className="hero-sub">
          The vLLM scheduler patterns — continuous batching, the radix prefix cache, and the memory-budget admission ladder — reimplemented from scratch on MLX's unified-memory model on M5 Pro 48 GB. The 35-cycle P-6 autoresearch loop closed in May 2026 with every <em>server-throughput</em> acceptance gate cleared 3.4-5.5× over the cycle-1 baseline: 232 tok/s on dense Qwen3.5-27B-4bit at B=64 (48 GB ceiling), 791.8 tok/s on MoE Qwen3.5-35B-A3B-4bit at B=128. <strong>Single-user latency at B=1 is ~20 tok/s, bandwidth-capped and unchanged by this phase</strong>; D-022 closed the small-B research line at v1.7.28 with β/γ/δ measurement-anchored negatives. Two parameter changes carried the aggregate result; 17 custom-kernel attempts closed without a load-bearing E2E win. <strong>P-8 closed at v1.7.33:</strong> <code>silica serve</code> boots a single-process FastAPI server fronting one loaded model — chat / completions / models endpoints, SSE streaming, <code>X-Silica-Session-ID</code> cross-request prefix reuse, bearer auth + token-bucket rate limit, OpenAI-shaped error envelope.
        </p>
        <div className="hero-ctas">
          <a className="btn btn-primary" href="#performance">Performance<Icon name="arrow" size={14} className="arrow" /></a>
          <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx" target="_blank" rel="noreferrer"><Icon name="github" size={14} />View on GitHub</a>
          <a className="btn btn-ghost" href="#architecture">Architecture</a>
        </div>

        <div className="hero-meta">
          <div className="hero-meta-item">
            <div className="hero-meta-label">Dense 27B decode</div>
            <div className="hero-meta-value">232<span className="unit">tok/s</span></div>
            <div className="hero-meta-sub">Qwen3.5-27B-4bit · B=64 · 48 GB ceiling</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">MoE 35B-A3B decode</div>
            <div className="hero-meta-value">791.8<span className="unit">tok/s</span></div>
            <div className="hero-meta-sub">Qwen3.5-35B-A3B-4bit · B=128</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">Single-user · B=1</div>
            <div className="hero-meta-value">~20<span className="unit">tok/s</span></div>
            <div className="hero-meta-sub">Bandwidth-capped · D-022 closed · ≤0.6% recoverable</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">Autoresearch</div>
            <div className="hero-meta-value">35<span className="unit">cycles</span></div>
            <div className="hero-meta-sub">Karpathy-style ledger · opus branch</div>
          </div>
        </div>
      </div>
    </section>
  );
};

const Footer = () => (
  <footer>
    <div className="container">
      <div className="foot-grid">
        <div className="foot-col">
          <div className="brand">
            <div className="brand-mark">SLMX</div>
            <span>silica-mlx</span>
          </div>
          <p className="foot-tagline">Continuous-batching LLM serving on Apple Silicon. vLLM-style scheduler, MLX-native.</p>
        </div>
        <div className="foot-col">
          <h4>Product</h4>
          <a href="#shipped">What's shipped</a>
          <a href="#performance">Performance</a>
          <a href="#roadmap">Roadmap</a>
          <a href="#codec">KV codec</a>
          <a href="#chat">Chat REPL</a>
          <a href="#compare">vs vLLM / SGLang</a>
        </div>
        <div className="foot-col">
          <h4>Docs</h4>
          <a href="https://silica-mlx.readthedocs.io/en/latest/" target="_blank" rel="noreferrer">Read the Docs</a>
          <a href="#quickstart">Quickstart</a>
          <a href="#architecture">Architecture</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/docs/bench.md">Benchmark guide</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/PLAN.md">PLAN.md</a>
        </div>
        <div className="foot-col">
          <h4>Project</h4>
          <a href="https://github.com/Ivis4ml/silica-mlx">GitHub</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/tree/sonnet/plans">plans/</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/tree/sonnet/docs">docs/</a>
        </div>
      </div>
      <div className="foot-bottom">
        <div>Apache-2.0 · Target hardware: M5 Pro 48 GB</div>
        <div className="mono">v1.7.35 · silica-mlx 1.0 RC · P-6 closed · P-8 shipped · M-9 cleared</div>
      </div>
    </div>
  </footer>
);

const Nav = () => (
  <nav className="nav">
    <div className="container nav-inner">
      <a className="brand" href="#" style={{ textDecoration: "none", color: "inherit" }}>
        <div className="brand-mark">SLMX</div>
        <span>silica-mlx</span>
      </a>
      <div className="nav-links">
        <a href="#architecture">Architecture</a>
        <a href="#shipped">Shipped</a>
        <a href="#performance">Performance</a>
        <a href="#codec">KV codec</a>
        <a href="#chat">Chat REPL</a>
        <a href="#roadmap">Roadmap</a>
      </div>
      <div className="nav-cta">
        <a href="https://github.com/Ivis4ml/silica-mlx" target="_blank" rel="noreferrer">
          <Icon name="github" size={14} />GitHub
        </a>
      </div>
    </div>
  </nav>
);

const App = () => {
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [tweaksOpen, setTweaksOpen] = React.useState(false);

  // wire up host edit-mode protocol
  React.useEffect(() => {
    const handler = (e) => {
      if (!e.data || typeof e.data !== "object") return;
      if (e.data.type === "__activate_edit_mode") setTweaksOpen(true);
      else if (e.data.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", handler);
    try { window.parent.postMessage({ type: "__edit_mode_available" }, "*"); } catch (e) {}
    return () => window.removeEventListener("message", handler);
  }, []);

  // apply tweaks to root
  React.useEffect(() => {
    const r = document.documentElement;
    r.dataset.theme = tweaks.theme;
    const a = ACCENTS[tweaks.accent] || ACCENTS.blue;
    if (tweaks.theme === "dark") {
      r.style.setProperty("--accent", a.dark);
      r.style.setProperty("--accent-2", a.deep_d);
      r.style.setProperty("--accent-soft", a.soft_d);
    } else {
      r.style.setProperty("--accent", a.light);
      r.style.setProperty("--accent-2", a.deep_l);
      r.style.setProperty("--accent-soft", a.soft_l);
    }
    if (tweaks.font === "system") {
      r.style.setProperty("--font-sans", "-apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Helvetica Neue', system-ui, sans-serif");
    } else {
      r.style.setProperty("--font-sans", "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif");
    }
    if (tweaks.density === "compact") {
      document.body.style.setProperty("--density", "compact");
      document.body.classList.add("compact");
    } else {
      document.body.classList.remove("compact");
    }
  }, [tweaks]);

  const closeTweaks = () => {
    setTweaksOpen(false);
    try { window.parent.postMessage({ type: "__edit_mode_dismissed" }, "*"); } catch (e) {}
  };

  return (
    <>
      <Nav />
      <Hero tweaks={tweaks} />
      <Comparison />
      <Performance />
      <Architecture />
      <SchedulerAnim />
      <Highlights />
      <Codec />
      <ChatRepl />
      <CodeSnippets />
      <Roadmap />
      <Footer />
      {tweaksOpen && <Tweaks tweaks={tweaks} setTweak={setTweak} onClose={closeTweaks} />}
      <style>{`
        body.compact section.block { padding: 64px 0; }
        body.compact .hero { padding: 56px 0 40px; }
        body.compact .section-head { margin-bottom: 32px; }
      `}</style>
    </>
  );
};

ReactDOM.createRoot(document.getElementById("app")).render(<App />);
