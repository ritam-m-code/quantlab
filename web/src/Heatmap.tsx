import type { Config, LayerMetrics } from './data'

export type Metric = 'one_minus_cos' | 'mse' | 'pct_clipped' | 'levels_used'

export const METRICS: Record<Metric, { label: string; get: (m: LayerMetrics) => number; log: boolean; worseHigh: boolean }> = {
  one_minus_cos: { label: '1 − cosine similarity', get: (m) => 1 - m.cos, log: true, worseHigh: true },
  mse: { label: 'mean squared error', get: (m) => m.mse, log: true, worseHigh: true },
  pct_clipped: { label: '% of weights clipped', get: (m) => m.pct_clipped, log: false, worseHigh: true },
  levels_used: { label: 'fraction of grid levels used', get: (m) => m.levels_used, log: false, worseHigh: false },
}

export const COLS = [
  ['attn.c_attn', 'attn QKV (768→2304)'],
  ['attn.c_proj', 'attn out (768→768)'],
  ['mlp.c_fc', 'MLP up (768→3072)'],
  ['mlp.c_proj', 'MLP down (3072→768)'],
] as const

export const fmt = (v: number) => (v === 0 ? '0' : Math.abs(v) < 0.01 || Math.abs(v) >= 1000 ? v.toExponential(1) : v.toFixed(3))

interface Props {
  config: Config
  metric: Metric
  selected: string
  onSelect: (layerId: string) => void
}

export function Heatmap({ config, metric, selected, onSelect }: Props) {
  const spec = METRICS[metric]
  const layers = config.layers ?? {}
  const t = (v: number) => (spec.log ? Math.log10(Math.max(v, 1e-12)) : v)
  const vals = Object.values(layers).map((m) => t(spec.get(m)))
  const lo = Math.min(...vals)
  const hi = Math.max(...vals)
  const shade = (m: LayerMetrics) => {
    const p = hi === lo ? 0.5 : (t(spec.get(m)) - lo) / (hi - lo)
    return spec.worseHigh ? p : 1 - p
  }

  return (
    <table>
      <thead>
        <tr>
          <th>block</th>
          {COLS.map(([, label]) => (
            <th key={label}>{label}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {Array.from({ length: 12 }, (_, b) => (
          <tr key={b}>
            <td>{b}</td>
            {COLS.map(([suffix]) => {
              const id = `h.${b}.${suffix}`
              const m = layers[id]
              if (!m) return <td key={id} />
              return (
                <td
                  key={id}
                  className={`cell${id === selected ? ' selected' : ''}`}
                  style={{ background: `rgba(220, 50, 50, ${(shade(m) * 0.8).toFixed(2)})` }}
                  onClick={() => onSelect(id)}
                  title={id}
                >
                  {fmt(spec.get(m))}
                </td>
              )
            })}
          </tr>
        ))}
      </tbody>
    </table>
  )
}
