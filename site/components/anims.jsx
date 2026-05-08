// anims.jsx — animation primitives shared across pages

// CountUp — animates from 0 to `to` when element scrolls into view.
const CountUp = ({ to, decimals = 0, duration = 1600, prefix = "", suffix = "", className = "", easing }) => {
  const ref = React.useRef(null);
  const [val, setVal] = React.useState(0);
  const [started, setStarted] = React.useState(false);

  React.useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined") {
      setStarted(true); return;
    }
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => { if (e.isIntersecting) { setStarted(true); io.disconnect(); } });
    }, { threshold: 0.4 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, []);

  React.useEffect(() => {
    if (!started) return;
    const start = performance.now();
    const ease = easing || ((t) => 1 - Math.pow(1 - t, 3));
    let raf;
    const tick = (now) => {
      const t01 = Math.min(1, (now - start) / duration);
      setVal(to * ease(t01));
      if (t01 < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [started, to, duration]);

  const display = decimals > 0 ? val.toFixed(decimals) : Math.round(val).toLocaleString();
  return <span ref={ref} className={"tnum " + className}>{prefix}{display}{suffix}</span>;
};

// BatchFlow — animated SVG showing tokens flowing through a continuous-batched
// scheduler. Multiple "rows" of dots stream right; each row gets pre-empted /
// admitted independently. The visual story: many parallel users → one chip.
const BatchFlow = ({ rows = 8, height = 280, accent = "var(--accent)" }) => {
  const ref = React.useRef(null);
  const [t, setT] = React.useState(0);
  const [active, setActive] = React.useState(true);
  React.useEffect(() => {
    if (!active) return;
    let raf;
    const start = performance.now();
    const tick = (now) => {
      setT((now - start) / 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [active]);
  React.useEffect(() => {
    if (!ref.current || typeof IntersectionObserver === "undefined") return;
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => setActive(e.isIntersecting));
    }, { threshold: 0.05 });
    io.observe(ref.current);
    return () => io.disconnect();
  }, []);

  const W = 1200, H = height;
  const padX = 24;
  const innerW = W - padX * 2;
  const rowH = (H - 24) / rows;
  // Row metadata — admit/preempt phases
  const rowMeta = React.useMemo(() => Array.from({ length: rows }, (_, i) => {
    const phase = (i * 0.37) % 1;
    const speed = 0.22 + (i * 0.07) % 0.22; // px ratio per second
    const dotCount = 14 + ((i * 5) % 8);
    return { phase, speed, dotCount, len: 0.30 + (i % 3) * 0.10 };
  }), [rows]);

  return (
    <svg ref={ref} viewBox={`0 0 ${W} ${H}`} className="batch-flow" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Continuous batching: many parallel users decoding on one chip">
      <defs>
        <linearGradient id="bf-fade" x1="0" x2="1">
          <stop offset="0%" stopColor={accent} stopOpacity="0" />
          <stop offset="35%" stopColor={accent} stopOpacity="1" />
          <stop offset="65%" stopColor={accent} stopOpacity="1" />
          <stop offset="100%" stopColor={accent} stopOpacity="0" />
        </linearGradient>
        <linearGradient id="bf-track" x1="0" x2="1">
          <stop offset="0%" stopColor="var(--ink)" stopOpacity="0" />
          <stop offset="50%" stopColor="var(--ink)" stopOpacity="0.06" />
          <stop offset="100%" stopColor="var(--ink)" stopOpacity="0" />
        </linearGradient>
      </defs>
      {/* tracks */}
      {Array.from({ length: rows }, (_, i) => {
        const y = 12 + i * rowH + rowH / 2;
        return (
          <line key={`tr-${i}`} x1={padX} x2={W - padX} y1={y} y2={y}
            stroke="url(#bf-track)" strokeWidth="1" />
        );
      })}
      {/* row labels (R0..) */}
      {Array.from({ length: rows }, (_, i) => {
        const y = 12 + i * rowH + rowH / 2 + 4;
        return (
          <text key={`lab-${i}`} x={padX - 8} y={y} textAnchor="end"
            fontFamily="var(--font-mono)" fontSize="9" fill="var(--ink-4)">R{i.toString().padStart(2, "0")}</text>
        );
      })}
      {/* moving dots */}
      {rowMeta.flatMap((m, i) => {
        const y = 12 + i * rowH + rowH / 2;
        const speed = m.speed * 0.45;
        const phase = (t * speed + m.phase) % 1;
        return Array.from({ length: m.dotCount }, (_, j) => {
          const local = (phase - j * (m.len / m.dotCount)) % 1;
          const wrapped = local < 0 ? local + 1 : local;
          const cx = padX + wrapped * innerW;
          const visible = wrapped > 0.05 && wrapped < 0.95;
          const opacity = visible ? Math.sin(wrapped * Math.PI) : 0;
          return (
            <circle key={`d-${i}-${j}`} cx={cx} cy={y} r={2.4}
              fill={accent} opacity={opacity * 0.92} />
          );
        });
      })}
      {/* prefill markers — periodic stronger pulse */}
      {rowMeta.map((m, i) => {
        const y = 12 + i * rowH + rowH / 2;
        const cycle = ((t * m.speed * 0.45 + m.phase) % 1);
        const xMarker = padX + cycle * innerW;
        const op = Math.max(0, 1 - Math.abs(cycle - 0.5) * 4);
        return (
          <circle key={`pm-${i}`} cx={xMarker} cy={y} r={4}
            fill={accent} opacity={op * 0.8}>
            <animate attributeName="r" values="4;6;4" dur="1.4s" repeatCount="indefinite" />
          </circle>
        );
      })}
      {/* head + tail edges */}
      <rect x={padX} y={0} width={2} height={H} fill="var(--ink-3)" opacity="0.18" />
      <rect x={W - padX - 2} y={0} width={2} height={H} fill="var(--ink-3)" opacity="0.18" />
      <text x={padX + 2} y={H - 4} fontFamily="var(--font-mono)" fontSize="9" fill="var(--ink-4)">prefill</text>
      <text x={W - padX - 2} y={H - 4} textAnchor="end" fontFamily="var(--font-mono)" fontSize="9" fill="var(--ink-4)">stop</text>
    </svg>
  );
};

// ParallaxHeroBg — soft animated gradient blobs in the hero background
const HeroBg = () => {
  return (
    <div className="hero-bg" aria-hidden="true">
      <div className="hero-blob hero-blob-1"></div>
      <div className="hero-blob hero-blob-2"></div>
      <div className="hero-blob hero-blob-3"></div>
      <div className="hero-grain"></div>
    </div>
  );
};

// MarqueeRow — horizontal scrolling tagline strip
const Marquee = ({ items, speed = 30 }) => {
  const list = [...items, ...items];
  return (
    <div className="marquee" style={{ "--marquee-speed": `${speed}s` }}>
      <div className="marquee-track">
        {list.map((it, i) => (
          <span key={i} className="marquee-item">{it}<span className="marquee-sep">◆</span></span>
        ))}
      </div>
    </div>
  );
};

Object.assign(window, { CountUp, BatchFlow, HeroBg, Marquee });
