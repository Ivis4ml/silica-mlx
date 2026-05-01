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
          <div className="hero-eyebrow"><span className="dot"></span>Status v1.7.19: scheduler core · family adapters · KV codec · spec foundation — shipped</div>
          <h1>
            Continuous-batching<br/>
            LLM serving,<br/>
            <em>native to Apple Silicon.</em>
          </h1>
          <p className="hero-sub">
            silica-mlx ports the vLLM scheduler core — continuous batching, memory-budget admission,
            preempt+replay, radix prefix cache — onto MLX's unified-memory model. Five model families
            with batched-output parity. Native KV codec compression and the speculative-decoding
            foundation (DraftTarget + three rollback paths) shipped at v1.7.19.
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
        <div className="hero-eyebrow"><span className="dot"></span>Status: scheduler core, family adapters, KV codec — shipped</div>
        <h1>Continuous-batching LLM serving, <em>native to Apple Silicon.</em></h1>
        <p className="hero-sub">
          The vLLM scheduler core, the radix prefix cache, and the memory-budget admission ladder that production-grade serving frameworks rely on — ported to MLX's unified-memory model on M5 Pro 48 GB. Five model families validated against batched mlx-lm references. Speculative-decoding foundation closed at v1.7.19 (DraftTarget + three rollback paths + spec-metrics into the bench harness).
        </p>
        <div className="hero-ctas">
          <a className="btn btn-primary" href="#quickstart">Quickstart<Icon name="arrow" size={14} className="arrow" /></a>
          <a className="btn btn-ghost" href="https://github.com/Ivis4ml/silica-mlx" target="_blank" rel="noreferrer"><Icon name="github" size={14} />View on GitHub</a>
          <a className="btn btn-ghost" href="#architecture">Architecture</a>
        </div>

        <div className="hero-meta">
          <div className="hero-meta-item">
            <div className="hero-meta-label">Decode</div>
            <div className="hero-meta-value">151.4<span className="unit">tok/s</span></div>
            <div className="hero-meta-sub">Qwen3-0.6B · M5 Pro</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">TTFT</div>
            <div className="hero-meta-value">25.1<span className="unit">ms</span></div>
            <div className="hero-meta-sub">15-token prompt</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">KV compression</div>
            <div className="hero-meta-value">3.8<span className="unit">×</span></div>
            <div className="hero-meta-sub">BlockTQ B=64 4-bit</div>
          </div>
          <div className="hero-meta-item">
            <div className="hero-meta-label">Δ PPL vs fp16</div>
            <div className="hero-meta-value">+0.0016</div>
            <div className="hero-meta-sub">WikiText-2 · 3 seeds</div>
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
          <p className="foot-tagline">Continuous-batching LLM serving on Apple Silicon. vLLM-core architecture, MLX-native.</p>
        </div>
        <div className="foot-col">
          <h4>Product</h4>
          <a href="#shipped">What's shipped</a>
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
        <div className="mono">v1.7.19 · scheduler core + KV codec + spec foundation shipped</div>
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
        <a href="#codec">KV codec</a>
        <a href="#chat">Chat REPL</a>
        <a href="#roadmap">Roadmap</a>
        <a href="#quickstart">Quickstart</a>
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
