import { useEffect, useState } from 'react'
import type { Data, Granularity, Selection } from './data'
import { configId, loadData } from './data'
import { Histogram } from './Histogram'
import { Heatmap, METRICS, type Metric } from './Heatmap'
import { Generations } from './Generations'

function Toggle<T extends string | number | boolean>({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: T
  options: [T, string][]
  onChange: (v: T) => void
}) {
  return (
    <div className="toggle" role="group" aria-label={label}>
      <span className="toggle-label">{label}</span>
      <div className="seg">
        {options.map(([v, text]) => (
          <button key={String(v)} className={v === value ? 'on' : ''} aria-pressed={v === value} onClick={() => onChange(v)}>
            {text}
          </button>
        ))}
      </div>
    </div>
  )
}

export default function App() {
  const [data, setData] = useState<Data | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [sel, setSel] = useState<Selection>({ bits: 4, symmetric: true, granularity: 'per-tensor', clip: 100 })
  const [layerId, setLayerId] = useState('h.0.attn.c_attn')
  const [metric, setMetric] = useState<Metric>('one_minus_cos')
  const [logY, setLogY] = useState(true)

  useEffect(() => {
    loadData().then(setData, (e) => setError(String(e)))
  }, [])

  if (error) return <main className="wrap"><p>Failed to load data: {error}</p></main>
  if (!data) return <main className="wrap"><p className="muted">Loading sweep results…</p></main>

  const set = <K extends keyof Selection>(k: K) => (v: Selection[K]) => setSel((s) => ({ ...s, [k]: v }))
  const baseline = data.byId.get('fp32')!
  const id = configId(sel)
  const config = data.byId.get(id)
  const layer = data.layers.find((l) => l.id === layerId)!
  const ranked = data.configs.filter((c) => c.id !== 'fp32').sort((a, b) => a.perplexity - b.perplexity)
  const pplMax = Math.log10(Math.max(...ranked.map((c) => c.perplexity)))
  const pplMin = Math.log10(baseline.perplexity)

  return (
    <main className="wrap">
      <header>
        <h1>QuantLab</h1>
        <p className="lede">
          Round every weight in GPT-2 (124M) to 8 or 4 bits and see what breaks. Each weight snaps to an evenly spaced
          grid; the controls decide how many weights share one grid, and that choice decides whether the model survives.
        </p>
      </header>

      <section className="controls card">
        <Toggle label="bits" value={sel.bits} options={[[8, '8-bit'], [4, '4-bit']]} onChange={set('bits')} />
        <Toggle label="range" value={sel.symmetric} options={[[true, 'symmetric'], [false, 'asymmetric']]} onChange={set('symmetric')} />
        <Toggle<Granularity>
          label="one grid per"
          value={sel.granularity}
          options={[['per-tensor', 'matrix'], ['per-channel', 'channel'], ['group-128', '128 weights']]}
          onChange={set('granularity')}
        />
        <Toggle label="calibrate on" value={sel.clip} options={[[100, 'true max'], [99.9, '99.9th pct']]} onChange={set('clip')} />
        <div className="ppl">
          <div className="ppl-label">perplexity</div>
          <div className="ppl-value">{config ? config.perplexity.toFixed(1) : '—'}</div>
          <div className="muted small">
            fp32 {baseline.perplexity.toFixed(1)}
            {config && ` · ${(config.perplexity / baseline.perplexity).toFixed(2)}×`}
          </div>
        </div>
      </section>

      {!config ? (
        <p className="card">Config <code>{id}</code> isn't in this sweep.</p>
      ) : (
        <>
          <section className="card">
            <div className="card-head">
              <h2>1 · The grid on the weights</h2>
              <div className="inline">
                <select value={layerId} onChange={(e) => setLayerId(e.target.value)} aria-label="Layer">
                  {data.layers.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.id} ({l.shape.join('×')})
                    </option>
                  ))}
                </select>
                <label className="check">
                  <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log counts
                </label>
              </div>
            </div>
            <Histogram layer={layer} config={config} whist={data.whist[id]?.[layerId]} logY={logY} />
            <LayerStats config={config} layerId={layerId} />
          </section>

          <section className="card">
            <div className="card-head">
              <h2>2 · Where the damage lands</h2>
              <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)} aria-label="Metric">
                {Object.entries(METRICS).map(([k, v]) => (
                  <option key={k} value={k}>
                    {v.label}
                  </option>
                ))}
              </select>
            </div>
            <Heatmap config={config} metric={metric} selected={layerId} onSelect={setLayerId} />
          </section>

          <section className="card">
            <h2>3 · What the model says</h2>
            <Generations prompts={data.prompts} baseline={baseline} config={config} />
          </section>
        </>
      )}

      <section className="card">
        <h2>All configs by perplexity</h2>
        <p className="muted small">
          Lower is better. Log scale, starting at the fp32 baseline ({baseline.perplexity.toFixed(1)}). Click a row to select it.
          A few 8-bit configs land slightly <em>below</em> fp32: on a 2,048-token passage, differences of about ±1 are noise,
          not an improvement.
</p>
        <ol className="ranking">
          {ranked.map((c) => {
            const w = pplMax === pplMin ? 0 : (Math.log10(c.perplexity) - pplMin) / (pplMax - pplMin)
            return (
              <li key={c.id}>
                <button
                  className={c.id === id ? 'on' : ''}
                  onClick={() =>
                    setSel({ bits: c.bits as 8 | 4, symmetric: c.symmetric!, granularity: c.granularity!, clip: c.clip_pct as 100 | 99.9 })
                  }
                >
                  <span className="rk-id">{c.id}</span>
                  <span className="rk-bar"><span style={{ width: `${Math.max(w * 100, 0.5)}%` }} /></span>
                  <span className="rk-val">{c.perplexity.toFixed(1)}</span>
                </button>
              </li>
            )
          })}
        </ol>
      </section>

      <footer className="muted small">
        <p>
          <strong>Weight-only.</strong> Everything here rounds the stored weights; the activations flowing between layers stay
          fp32. Quantizing activations is a harder problem: a few channels run 10–100× larger than the rest, which is what
          LLM.int8(), SmoothQuant and AWQ each address.
        </p>
        <p>
          The 48 matrices are the four per block (attention QKV and output, MLP up and down). Embeddings and the tied
          lm_head stay fp32. That's a choice, not a law. Perplexity is measured on 2,048 tokens of <em>Pride and Prejudice</em>;
          everything is precomputed offline in Python and served as static JSON.
        </p>
      </footer>
    </main>
  )
}

function LayerStats({ config, layerId }: { config: Data['configs'][number]; layerId: string }) {
  const m = config.layers?.[layerId]
  if (!m) return null
  const items: [string, string][] = [
    ['cosine sim', m.cos.toFixed(5)],
    ['MSE', m.mse.toExponential(2)],
    ['max |error|', m.max_abs_err.toExponential(2)],
    ['clipped', `${m.pct_clipped.toFixed(3)}%`],
    ['levels used', `${(m.levels_used * 100).toFixed(0)}%`],
  ]
  return (
    <dl className="stats">
      {items.map(([k, v]) => (
        <div key={k}>
          <dt>{k}</dt>
          <dd>{v}</dd>
        </div>
      ))}
    </dl>
  )
}
