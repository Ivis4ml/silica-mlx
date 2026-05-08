// shell.jsx — Nav, Footer, theme + tweaks scaffolding shared across pages

// BrandMark — the silica-mlx logo (prefix tree). One shared root forks into
// three concurrent branches; the root dot picks up the active accent.
// Pure inline SVG so it lives inside the page and inherits theme colors.
const BrandMark = ({ size = 30 }) => (
  <svg className="brand-svg" width={size} height={size} viewBox="0 0 88 88" fill="none" aria-hidden="true">
    <g stroke="currentColor" strokeWidth="5" strokeLinecap="round" fill="none">
      <path d="M44 72 V 50" />
      <path d="M44 50 C 44 38, 22 38, 22 22" />
      <path d="M44 50 V 22" />
      <path d="M44 50 C 44 38, 66 38, 66 22" />
    </g>
    <circle cx="44" cy="72" r="8" fill="var(--accent)" />
    <circle cx="22" cy="22" r="6" fill="currentColor" />
    <circle cx="44" cy="22" r="6" fill="currentColor" />
    <circle cx="66" cy="22" r="6" fill="currentColor" />
  </svg>
);

const NAV_LINKS = [
  { href: "index.html", label: "Overview", key: "overview" },
  { href: "architecture.html", label: "Architecture", key: "architecture" },
  { href: "autoresearch.html", label: "Autoresearch", key: "autoresearch" },
  { href: "chat.html", label: "Chat REPL", key: "chat" },
  { href: "changelog.html", label: "Changelog", key: "changelog" },
];

const Nav = ({ active }) => (
  <nav className="nav">
    <div className="container nav-inner">
      <a className="brand" href="index.html">
        <BrandMark size={28} />
        <span className="brand-text">silica<span className="brand-text-dim">-mlx</span></span>
      </a>
      <div className="nav-links">
        {NAV_LINKS.map(l => (
          <a key={l.key} href={l.href} className={active === l.key ? "active" : ""}>{l.label}</a>
        ))}
      </div>
      <div className="nav-cta">
        <a href="https://github.com/Ivis4ml/silica-mlx" target="_blank" rel="noreferrer" className="nav-pill">GitHub ↗</a>
      </div>
    </div>
  </nav>
);

const Footer = () => (
  <footer className="foot">
    <div className="container">
      <div className="foot-grid">
        <div>
          <a className="brand" href="index.html">
            <BrandMark size={32} />
            <span className="brand-text">silica<span className="brand-text-dim">-mlx</span></span>
          </a>
          <p className="foot-tagline">vLLM-style continuous-batching LLM serving, native to Apple Silicon. M5 Pro 48 GB target. Apache-2.0.</p>
        </div>
        <div className="foot-col">
          <h4>Product</h4>
          <a href="index.html#shipped">What's shipped</a>
          <a href="autoresearch.html">Autoresearch</a>
          <a href="changelog.html">Roadmap</a>
          <a href="index.html#codec">KV codec</a>
          <a href="chat.html">Chat REPL</a>
        </div>
        <div className="foot-col">
          <h4>Docs</h4>
          <a href="https://silica-mlx.readthedocs.io/" target="_blank" rel="noreferrer">Read the Docs ↗</a>
          <a href="index.html#quickstart">Quickstart</a>
          <a href="architecture.html">Architecture</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/blob/sonnet/plans/PLAN.md">PLAN.md ↗</a>
        </div>
        <div className="foot-col">
          <h4>Project</h4>
          <a href="https://github.com/Ivis4ml/silica-mlx">GitHub ↗</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/tree/sonnet/plans">plans/ ↗</a>
          <a href="https://github.com/Ivis4ml/silica-mlx/tree/sonnet/docs">docs/ ↗</a>
        </div>
      </div>
      <div className="foot-bottom">
        <div>Apache-2.0 · M5 Pro 48 GB target</div>
        <div>v1.7.35 · 1.0 RC · P-6 closed · P-8 shipped · M-9 cleared</div>
      </div>
    </div>
  </footer>
);

// ─── tweaks system ──────────────────────────────────────
const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "light",
  "accent": "violet",
  "density": "roomy"
}/*EDITMODE-END*/;

const ACCENTS = {
  violet: { l: "#4f4afe", d: "#8c84ff", glow_l: "rgba(79,74,254,0.35)", glow_d: "rgba(140,132,255,0.45)", soft_l: "rgba(79,74,254,0.08)", soft_d: "rgba(140,132,255,0.12)", deep_l: "#3a35d6", deep_d: "#b1abff" },
  ember:  { l: "#d6510a", d: "#ff8c4a", glow_l: "rgba(214,81,10,0.35)",  glow_d: "rgba(255,140,74,0.45)",  soft_l: "rgba(214,81,10,0.08)",  soft_d: "rgba(255,140,74,0.14)",  deep_l: "#a83b00", deep_d: "#ffb38c" },
  moss:   { l: "#1f7a3b", d: "#4cc26b", glow_l: "rgba(31,122,59,0.35)",  glow_d: "rgba(76,194,107,0.45)",  soft_l: "rgba(31,122,59,0.08)",  soft_d: "rgba(76,194,107,0.14)",  deep_l: "#0e5224", deep_d: "#85e09a" },
  ink:    { l: "#0c0c0e", d: "#f6f5f1", glow_l: "rgba(12,12,14,0.20)",   glow_d: "rgba(246,245,241,0.30)", soft_l: "rgba(12,12,14,0.07)",   soft_d: "rgba(246,245,241,0.10)", deep_l: "#000",     deep_d: "#fff" },
};

const useTweaks = (defaults) => {
  const [tweaks, setTweaks] = React.useState(() => {
    try {
      const saved = localStorage.getItem("slmx_tweaks");
      if (saved) return { ...defaults, ...JSON.parse(saved) };
    } catch (_) {}
    return defaults;
  });
  const setTweak = React.useCallback((k, v) => {
    setTweaks(prev => {
      const next = typeof k === "object" ? { ...prev, ...k } : { ...prev, [k]: v };
      try {
        localStorage.setItem("slmx_tweaks", JSON.stringify(next));
        window.parent.postMessage({ type: "__edit_mode_set_keys", edits: typeof k === "object" ? k : { [k]: v } }, "*");
      } catch (_) {}
      return next;
    });
  }, []);
  return [tweaks, setTweak];
};

const applyTweaks = (tweaks) => {
  const r = document.documentElement;
  r.dataset.theme = tweaks.theme;
  const a = ACCENTS[tweaks.accent] || ACCENTS.violet;
  const isDark = tweaks.theme === "dark";
  r.style.setProperty("--accent",      isDark ? a.d      : a.l);
  r.style.setProperty("--accent-2",    isDark ? a.deep_d : a.deep_l);
  r.style.setProperty("--accent-soft", isDark ? a.soft_d : a.soft_l);
  r.style.setProperty("--accent-glow", isDark ? a.glow_d : a.glow_l);
  document.body.classList.toggle("compact", tweaks.density === "compact");
};

const Tweaks = ({ tweaks, setTweak, onClose }) => (
  <div className="tweaks">
    <div className="tweaks-head">
      <span className="tweaks-title">Tweaks</span>
      <button className="tweaks-close" onClick={onClose} aria-label="close">✕</button>
    </div>
    <div className="tweak-row">
      <span className="tweak-label">Theme</span>
      <span className="seg">
        <button className={tweaks.theme === "light" ? "on" : ""} onClick={() => setTweak("theme", "light")}>Light</button>
        <button className={tweaks.theme === "dark" ? "on" : ""} onClick={() => setTweak("theme", "dark")}>Dark</button>
      </span>
    </div>
    <div className="tweak-row">
      <span className="tweak-label">Accent</span>
      <div className="swatches">
        {Object.entries(ACCENTS).map(([k, v]) => (
          <button
            key={k}
            className={"swatch" + (tweaks.accent === k ? " on" : "")}
            onClick={() => setTweak("accent", k)}
            style={{ background: tweaks.theme === "dark" ? v.d : v.l }}
            aria-label={k}
          />
        ))}
      </div>
    </div>
    <div className="tweak-row">
      <span className="tweak-label">Density</span>
      <span className="seg">
        <button className={tweaks.density === "roomy" ? "on" : ""} onClick={() => setTweak("density", "roomy")}>Roomy</button>
        <button className={tweaks.density === "compact" ? "on" : ""} onClick={() => setTweak("density", "compact")}>Compact</button>
      </span>
    </div>
  </div>
);

// reveal-on-scroll observer
const useReveal = () => {
  React.useEffect(() => {
    if (typeof IntersectionObserver === "undefined") {
      document.querySelectorAll(".reveal").forEach(el => el.classList.add("in"));
      return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add("in");
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.15, rootMargin: "0px 0px -8% 0px" });
    document.querySelectorAll(".reveal").forEach(el => io.observe(el));
    return () => io.disconnect();
  }, []);
};

// page-level wrapper that handles theme + tweaks edit-mode protocol
const PageShell = ({ active, children }) => {
  const [tweaks, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const [tweaksOpen, setTweaksOpen] = React.useState(false);
  React.useEffect(() => { applyTweaks(tweaks); }, [tweaks]);
  React.useEffect(() => {
    const handler = (e) => {
      if (!e.data || typeof e.data !== "object") return;
      if (e.data.type === "__activate_edit_mode") setTweaksOpen(true);
      else if (e.data.type === "__deactivate_edit_mode") setTweaksOpen(false);
    };
    window.addEventListener("message", handler);
    try { window.parent.postMessage({ type: "__edit_mode_available" }, "*"); } catch (_) {}
    return () => window.removeEventListener("message", handler);
  }, []);
  useReveal();
  const closeTweaks = () => {
    setTweaksOpen(false);
    try { window.parent.postMessage({ type: "__edit_mode_dismissed" }, "*"); } catch (_) {}
  };
  return (
    <>
      <Nav active={active} />
      {children}
      <Footer />
      {tweaksOpen && <Tweaks tweaks={tweaks} setTweak={setTweak} onClose={closeTweaks} />}
    </>
  );
};

Object.assign(window, { Nav, Footer, Tweaks, PageShell, useTweaks, applyTweaks, useReveal, TWEAK_DEFAULTS, ACCENTS });
