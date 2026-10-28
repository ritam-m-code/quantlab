import type { Config, LayerInfo } from './data'
import { qrange } from './data'

const W = 760
const H = 260
const PAD = { l: 6, r: 6, t: 10, b: 24 }

/** Grid levels for a per-tensor config (one scale for the whole matrix); empty otherwise. */
export function gridLevels(layer: LayerInfo, config: Config): number[] {
  const m = config.layers?.[layer.id]
  if (m?.scale === undefined || config.bits === undefined || config.symmetric === undefined) return []
  const [qmin, qmax] = qrange(config.bits, config.symmetric)
  const out = []
  for (let q = qmin; q <= qmax; q++) out.push((q - (m.zero_point ?? 0)) * m.scale)
  return out
}

/** Approximate fraction of weights with |w| < step/2, i.e. that round to exactly 0. From the 256-bin histogram. */
export function fracToZero(layer: LayerInfo, step: number): number {
  const { bins, counts, outside } = layer.hist
  const total = counts.reduce((a, b) => a + b, 0) + outside[0] + outside[1]
  let n = 0
  counts.forEach((c, i) => {
    const mid = (bins[i] + bins[i + 1]) / 2
    if (Math.abs(mid) < step / 2) n += c
  })
  return n / total
}

interface Props {
  layer: LayerInfo
  config: Config
  whist: number[] | undefined
  logY: boolean
}

export function Histogram({ layer, config, whist, logY }: Props) {
  const { bins, counts } = layer.hist
  const lo = bins[0]
  const hi = bins[bins.length - 1]
  const x = (v: number) => PAD.l + ((v - lo) / (hi - lo)) * (W - PAD.l - PAD.r)
  const f = (c: number) => (logY ? Math.log10(1 + c) : c)
  const yMax = Math.max(...counts.map(f), ...(whist ?? []).map(f))
  const y = (c: number) => H - PAD.b - (f(c) / yMax) * (H - PAD.t - PAD.b)
  const bw = (W - PAD.l - PAD.r) / counts.length
  const grid = gridLevels(layer, config).filter((v) => v >= lo && v <= hi)

  return (
    <svg viewBox={`0 0 ${W} ${H}`}>
      {counts.map((c, i) => (c > 0 ? <rect key={i} x={x(bins[i])} width={bw} y={y(c)} height={H - PAD.b - y(c)} className="orig" /> : null))}
      {grid.map((v, i) => (
        <line key={i} x1={x(v)} x2={x(v)} y1={PAD.t} y2={H - PAD.b} className="gridline" />
      ))}
      {whist?.map((c, i) =>
        c > 0 ? <rect key={i} x={x(bins[i]) + bw * 0.2} width={Math.max(bw * 0.6, 1)} y={y(c)} height={H - PAD.b - y(c)} className="q" /> : null,
      )}
      <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} className="axis" />
      {[lo, lo / 2, 0, hi / 2, hi].map((t, i) => (
        <text key={i} x={x(t)} y={H - 7} className="tick" textAnchor={i === 0 ? 'start' : i === 4 ? 'end' : 'middle'}>
          {t === 0 ? '0' : t.toFixed(3)}
        </text>
      ))}
    </svg>
  )
}
