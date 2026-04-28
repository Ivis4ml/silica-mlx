// KV codec compression deep-dive — visualizes BlockTQ vs RaBitQ vs Identity

const Codec = () => {
  // bits per element / PPL deltas
  const codecs = [
    { id: "identity", name: "Identity", bits: 16, savings: "1.0×", ppl: "0.000", hi: false, body: "fp16 baseline reference" },
    { id: "btq-b32-b4", name: "BlockTQ B=32", bits: 4.5, savings: "3.5×", ppl: "+0.002", hi: false, body: "block turbo-quant" },
    { id: "btq-b64-b4", name: "BlockTQ B=64", bits: 4.25, savings: "3.8×", ppl: "+0.002", hi: true, body: "default · ties vqbench at measurement precision" },
    { id: "ext-b4", name: "ExtRaBitQ 4-bit", bits: 4.5, savings: "3.5×", ppl: "+0.004", hi: false, body: "extended rotation grid" },
    { id: "ext-b3", name: "ExtRaBitQ 3-bit", bits: 3.5, savings: "4.6×", ppl: "+0.018", hi: false, body: "extended rotation grid" },
    { id: "ext-b2", name: "ExtRaBitQ 2-bit", bits: 2.5, savings: "6.4×", ppl: "+0.092", hi: false, body: "aggressive · admission headroom" },
    { id: "rabit-1", name: "RaBitQ-1", bits: 1.5, savings: "10.7×", ppl: "+0.318", hi: false, body: "1-bit · maximum compression" },
  ];

  const maxBits = 16;

  return (
    <section className="block" id="codec">
      <div className="container">
        <div className="section-head">
          <div className="section-eyebrow">P-5 · KV codec stack</div>
          <h2>Lossless at measurement precision.</h2>
          <p>BlockTQ B=64 4-bit matches the vqbench baseline on Qwen3.5-4B WikiText-2: ΔPPL = +0.0016 across three seeds, statistically indistinguishable from vqbench's reported <span className="mono">±0.000%</span>. The codec lives at the prefix-block store; admission headroom turns straight into more admitted requests.</p>
        </div>

        <div className="cdc-card">
          <div className="cdc-grid">
            <div className="cdc-h cdc-h-name">Codec</div>
            <div className="cdc-h cdc-h-bar">Bits per element</div>
            <div className="cdc-h cdc-h-num">Savings</div>
            <div className="cdc-h cdc-h-num">Δ PPL</div>

            {codecs.map(c => (
              <React.Fragment key={c.id}>
                <div className={"cdc-name" + (c.hi ? " cdc-name-hi" : "")}>
                  <div className="cdc-name-row">
                    <span className="cdc-name-t">{c.name}</span>
                    {c.hi && <span className="cdc-badge mono">default</span>}
                  </div>
                  <div className="cdc-name-sub">{c.body}</div>
                </div>
                <div className="cdc-bar-wrap">
                  <div
                    className={"cdc-bar" + (c.hi ? " cdc-bar-hi" : "")}
                    style={{ width: `${(c.bits / maxBits) * 100}%` }}
                  >
                    <span className="cdc-bar-label mono">{c.bits.toFixed(2)}</span>
                  </div>
                </div>
                <div className="cdc-num mono">{c.savings}</div>
                <div className={"cdc-num mono " + (c.ppl.startsWith("+0.0") && c.ppl.length <= 6 ? "cdc-good" : "cdc-meh")}>{c.ppl}</div>
              </React.Fragment>
            ))}
          </div>
          <div className="cdc-foot">
            <div>Qwen3.5-4B · WikiText-2 · three seeds (42, 43, 44).</div>
            <div className="mono">VectorCodec[P] · side-level since P-5-A.0.4</div>
          </div>
        </div>
      </div>

      <style>{`
        .cdc-card {
          background: var(--bg-elev);
          border-radius: var(--radius);
          box-shadow: var(--shadow-sm);
          overflow: hidden;
        }
        .cdc-grid {
          display: grid;
          grid-template-columns: minmax(220px, 1.6fr) minmax(200px, 2.4fr) minmax(80px, 0.7fr) minmax(80px, 0.7fr);
          align-items: center;
        }
        .cdc-h {
          padding: 14px 18px;
          font-size: 11px;
          color: var(--ink-4);
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-weight: 500;
          border-bottom: 1px solid var(--rule);
        }
        .cdc-h-num { text-align: right; }
        .cdc-name {
          padding: 16px 18px;
          border-bottom: 1px solid var(--rule-2);
        }
        .cdc-name-hi { background: var(--accent-soft); }
        .cdc-name-row { display: flex; align-items: center; gap: 8px; }
        .cdc-name-t { font-size: 14px; font-weight: 500; color: var(--ink); }
        .cdc-name-hi .cdc-name-t { color: var(--accent); font-weight: 600; }
        .cdc-name-sub { font-size: 12px; color: var(--ink-3); margin-top: 2px; }
        .cdc-badge {
          font-size: 10px;
          color: var(--accent);
          background: var(--bg-elev);
          border: 1px solid color-mix(in srgb, var(--accent) 30%, transparent);
          padding: 1px 6px; border-radius: 4px;
          letter-spacing: 0.04em;
        }
        .cdc-bar-wrap {
          padding: 16px 18px;
          border-bottom: 1px solid var(--rule-2);
        }
        .cdc-name-hi + .cdc-bar-wrap { background: var(--accent-soft); }
        .cdc-bar {
          height: 22px;
          background: linear-gradient(90deg, var(--ink-3), var(--ink-2));
          border-radius: 4px;
          display: flex; align-items: center;
          justify-content: flex-end;
          padding: 0 8px;
          color: var(--bg-elev);
          font-size: 11px;
          min-width: 40px;
          transition: width 0.4s cubic-bezier(.2,.7,.2,1);
        }
        .cdc-bar-hi {
          background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }
        .cdc-num {
          padding: 16px 18px;
          font-size: 13px;
          color: var(--ink-2);
          text-align: right;
          border-bottom: 1px solid var(--rule-2);
        }
        .cdc-name-hi ~ .cdc-num:nth-of-type(-n+2) {}
        .cdc-good { color: var(--ok); }
        .cdc-meh { color: var(--ink-3); }
        .cdc-grid > div:nth-last-child(-n+4) { border-bottom: none; }
        .cdc-foot {
          display: flex; justify-content: space-between;
          padding: 14px 18px;
          font-size: 12px; color: var(--ink-3);
          background: var(--bg-sunken);
          border-top: 1px solid var(--rule);
          flex-wrap: wrap; gap: 8px;
        }
        @media (max-width: 720px) {
          .cdc-grid { grid-template-columns: 1.4fr 2fr 0.6fr 0.6fr; font-size: 12px; }
          .cdc-h, .cdc-name, .cdc-bar-wrap, .cdc-num { padding: 10px 8px; }
          .cdc-name-sub { display: none; }
        }
      `}</style>
    </section>
  );
};

window.Codec = Codec;
