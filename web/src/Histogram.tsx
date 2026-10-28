import type { Config, LayerInfo } from './data'
import { qrange } from './data'

const W = 760
const H = 280
const PAD = { l: 8, r: 8, t: 12, b: 26 }

interface Props {
  layer: LayerInfo
  config: Config
  whist: number[] | undefined
  logY: boolean
}

/** Original weight histogram, the dequantized histogram on top, and (per-tensor) the exact grid. */
export function Histogram({ layer, config, whist, logY }: Props) {
  const { bins, counts, outside } = layer.hist
  const lo = bins[0]
  const hi = bins[bins.length - 1]
  const x = (v: number) => PAD.l + ((v - lo) / (hi - lo)) * (W - PAD.l - PAD.r)
  const f = (c: number) => (logY ? Math.log10(1 + c) : c)
  const yMax = Math.max(...counts.map(f), ...(whist ?? []).map(f))
  const y = (c: number) => H - PAD.b - (f(c) / yMax) * (H - PAD.t - PAD.b)
  const bw = (W - PAD.l - PAD.r) / counts.length

  const area =
    `M${x(lo)},${y(0)} ` +
    counts.map((c, i) => `L${x(bins[i])},${y(c)} L${x(bins[i + 1])},${y(c)}`).join(' ') +
    ` L${x(hi)},${y(0)} Z`

  const m = config.layers?.[layer.id]
  // Per-tensor: rebuild every grid level from scale / zero_point, exactly as dequantize would.
  const levels: number[] = []
  if (m?.scale !== undefined && config.bits !== undefined && config.symmetric !== undefined) {
    const [qmin, qmax] = qrange(config.bits, config.symmetric)
    for (let q = qmin; q <= qmax; q++) levels.push((q - (m.zero_point ?? 0)) * m.scale)
  }
  const grid = levels.filter((v) => v >= lo && v <= hi)

  const ticks = [lo, lo / 2, 0, hi / 2, hi]
  const nOutside = outside[0] + outside[1]

  return (
    <figure className="hist">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Weight histogram of ${layer.id} under ${config.id}`}>
        <path d={area} className="hist-orig" />
        {grid.map((v, i) => (
          <line key={i} x1={x(v)} x2={x(v)} y1={PAD.t} y2={H - PAD.b} className="hist-grid" />
        ))}
        {whist?.map((c, i) =>
          c > 0 ? (
            <rect
              key={i}
              x={x(bins[i]) + bw * 0.15}
              width={Math.max(bw * 0.7, 1)}
              y={y(c)}
              height={H - PAD.b - y(c)}
              className="hist-q"
            />
          ) : null,
        )}
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="axis" />
        {ticks.map((t, i) => (
          <text key={i} x={x(t)} y={H - 8} className="tick" textAnchor={i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}>
            {t === 0 ? '0' : t.toFixed(3)}
          </text>
        ))}
      </svg>
      <figcaption>
        <span className="key key-orig" /> original fp32 weights
        <span className="key key-q" /> after quantize → dequantize
        {m?.scale !== undefined ? (
          <>
            <span className="key key-grid" /> grid: {grid.length} of {levels.length} levels in view, step {m.scale.toExponential(2)}
          </>
        ) : m ? (
          <> · each {config.granularity === 'per-channel' ? 'channel' : 'group of 128'} has its own grid; median step{' '}
            {m.scale_quantiles[2].toExponential(2)} (range {m.scale_quantiles[0].toExponential(1)}–
            {m.scale_quantiles[4].toExponential(1)})
          </>
        ) : null}
        {nOutside > 0 && (
          <span className="muted">
            {' '}
            · {nOutside.toLocaleString()} outlier weights beyond the plotted range (true range {layer.min.toFixed(2)} to{' '}
            {layer.max.toFixed(2)})
          </span>
        )}
      </figcaption>
    </figure>
  )
}
