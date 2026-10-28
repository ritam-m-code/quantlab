import type { Config, LayerMetrics } from './data'

export type Metric = 'one_minus_cos' | 'mse' | 'pct_clipped' | 'levels_used'

export const METRICS: Record<Metric, { label: string; get: (m: LayerMetrics) => number; log: boolean; worseHigh: boolean }> = {
  one_minus_cos: { label: '1 − cosine sim', get: (m) => 1 - m.cos, log: true, worseHigh: true },
  mse: { label: 'MSE', get: (m) => m.mse, log: true, worseHigh: true },
  pct_clipped: { label: '% clipped', get: (m) => m.pct_clipped, log: false, worseHigh: true },
  levels_used: { label: 'levels used', get: (m) => m.levels_used, log: false, worseHigh: false },
}

const COLS = [
  ['attn.c_attn', 'attn QKV'],
  ['attn.c_proj', 'attn out'],
  ['mlp.c_fc', 'MLP up'],
  ['mlp.c_proj', 'MLP down'],
] as const

const fmt = (v: number) => (v === 0 ? '0' : Math.abs(v) < 0.01 || Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(3))

interface Props {
  config: Config
  metric: Metric
  selected: string
  onSelect: (layerId: string) => void
}

/** 12 blocks x 4 matrices, coloured by the chosen metric (darker = more damage). */
export function Heatmap({ config, metric, selected, onSelect }: Props) {
  const spec = METRICS[metric]
  const layers = config.layers ?? {}
  const t = (v: number) => (spec.log ? Math.log10(Math.max(v, 1e-12)) : v)
  const vals = Object.values(layers).map((m) => t(spec.get(m)))
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const level = (m: LayerMetrics) => {
    const p = hi === lo ? 0.5 : (t(spec.get(m)) - lo) / (hi - lo)
    return spec.worseHigh ? p : 1 - p
  }

  return (
    <div className="heatmap">
      <div className="hm-row hm-head">
        <span />
        {COLS.map(([, label]) => (
          <span key={label}>{label}</span>
        ))}
      </div>
      {Array.from({ length: 12 }, (_, b) => (
        <div className="hm-row" key={b}>
          <span className="hm-block">block {b}</span>
          {COLS.map(([suffix]) => {
            const id = `h.${b}.${suffix}`
            const m = layers[id]
            if (!m) return <span key={id} />
            const v = spec.get(m)
            return (
              <button
                key={id}
                className={`hm-cell${id === selected ? ' selected' : ''}`}
                style={{ background: `color-mix(in oklab, var(--heat-hi) ${Math.round(level(m) * 100)}%, var(--heat-lo))` }}
                onClick={() => onSelect(id)}
                title={`${id}: ${spec.label} = ${fmt(v)}`}
                aria-label={`${id}: ${spec.label} ${fmt(v)}`}
              >
                <span style={{ color: level(m) > 0.55 ? 'var(--heat-text-dark)' : 'var(--heat-text-light)' }}>{fmt(v)}</span>
              </button>
            )
          })}
        </div>
      ))}
      <p className="muted small">
        Colour is scaled within this config ({spec.log ? 'log scale, ' : ''}
        {fmt(spec.log ? 10 ** lo : lo)} to {fmt(spec.log ? 10 ** hi : hi)}). Click a cell to inspect that matrix above.
      </p>
    </div>
  )
}
