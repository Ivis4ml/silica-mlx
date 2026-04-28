// Tweaks panel — light/dark, accent, density, animations

const Tweaks = ({ tweaks, setTweak, onClose }) => {
  const accents = [
    { id: "blue", color: "#0066ff", dark: "#4d94ff" },
    { id: "graphite", color: "#1c1c1e", dark: "#e0e0e3" },
    { id: "violet", color: "#6e3aff", dark: "#9d7dff" },
    { id: "ember", color: "#d6510a", dark: "#ff8c4a" },
    { id: "moss", color: "#1f7a3b", dark: "#4cc26b" },
  ];

  return (
    <div className="tweaks">
      <div className="tweaks-head">
        <div className="tweaks-title">Tweaks</div>
        <button className="tweaks-close" onClick={onClose} aria-label="Close">×</button>
      </div>

      <div className="tweak-row">
        <span className="tweak-label">Theme</span>
        <div className="seg">
          <button className={tweaks.theme === "light" ? "on" : ""} onClick={() => setTweak("theme", "light")}>Light</button>
          <button className={tweaks.theme === "dark" ? "on" : ""} onClick={() => setTweak("theme", "dark")}>Dark</button>
        </div>
      </div>

      <div className="tweak-row">
        <span className="tweak-label">Accent</span>
        <div className="swatches">
          {accents.map(a => (
            <button
              key={a.id}
              className={"swatch" + (tweaks.accent === a.id ? " on" : "")}
              style={{ background: tweaks.theme === "dark" ? a.dark : a.color }}
              onClick={() => setTweak("accent", a.id)}
              aria-label={a.id}
            />
          ))}
        </div>
      </div>

      <div className="tweak-row">
        <span className="tweak-label">Type</span>
        <div className="seg">
          <button className={tweaks.font === "inter" ? "on" : ""} onClick={() => setTweak("font", "inter")}>Inter</button>
          <button className={tweaks.font === "system" ? "on" : ""} onClick={() => setTweak("font", "system")}>System</button>
        </div>
      </div>

      <div className="tweak-row">
        <span className="tweak-label">Density</span>
        <div className="seg">
          <button className={tweaks.density === "roomy" ? "on" : ""} onClick={() => setTweak("density", "roomy")}>Roomy</button>
          <button className={tweaks.density === "compact" ? "on" : ""} onClick={() => setTweak("density", "compact")}>Compact</button>
        </div>
      </div>

      <div className="tweak-row">
        <span className="tweak-label">Hero</span>
        <div className="seg">
          <button className={tweaks.hero === "type" ? "on" : ""} onClick={() => setTweak("hero", "type")}>Type</button>
          <button className={tweaks.hero === "metrics" ? "on" : ""} onClick={() => setTweak("hero", "metrics")}>Metrics</button>
        </div>
      </div>
    </div>
  );
};

window.Tweaks = Tweaks;
