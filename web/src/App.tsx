import { useEffect, useState } from 'react'
import type { Config, Data, Granularity, LayerInfo, Selection } from './data'
import { configId, loadData, qrange } from './data'
import { fracToZero, gridLevels, Histogram } from './Histogram'
import { COLS, fmt, Heatmap, METRICS, type Metric } from './Heatmap'
import { Generations } from './Generations'

function Radio<T extends string | number | boolean>({
  legend,
  value,
  options,
  onChange,
}: {
  legend: string
  value: T
  options: [T, string][]
  onChange: (v: T) => void
}) {
  return (
    <fieldset>
      <legend>{legend}</legend>
      {options.map(([v, text]) => (
        <label key={String(v)} style={{ marginRight: 12 }}>
          <input type="radio" checked={v === value} onChange={() => onChange(v)} /> {text}
        </label>
      ))}
    </fieldset>
  )
}

const pct = (x: number) => `${(x * 100).toFixed(x < 0.01 ? 2 : 1)}%`
const n = (x: number) => x.toLocaleString()

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

  if (error) return <p>Failed to load data: {error}</p>
  if (!data) return <p>Loading…</p>

  const set = <K extends keyof Selection>(k: K) => (v: Selection[K]) => setSel((s) => ({ ...s, [k]: v }))
  const baseline = data.byId.get('fp32')!
  const id = configId(sel)
  const config = data.byId.get(id)
  const layer = data.layers.find((l) => l.id === layerId)!
  const ranked = data.configs.filter((c) => c.id !== 'fp32').sort((a, b) => a.perplexity - b.perplexity)

  return (
    <>
      <h1>QuantLab: rounding GPT-2's weights</h1>
      <p>
        GPT-2 small has 124 million weights, each stored as a 32-bit float (4 bytes). <b>Quantization</b> stores each
        weight in far fewer bits instead: 8 or 4 here. With 4 bits you only get 16 different values, so every weight has
        to be <i>rounded</i> to the nearest point on a small, evenly spaced grid. The grid is defined by a <b>scale</b>{' '}
        (the spacing between points), chosen so the grid just covers the weights. The weight that gets stored is the
        index of the nearest grid point, and at runtime it's turned back into a number: <code>index × scale</code>.
      </p>
      <p>
        That rounding throws information away. This page shows how much, and where, for 24 different ways of doing it.
        Everything was precomputed in Python (<code>core/quant.py</code>, <code>core/sweep.py</code>); this page only
        displays the results.
      </p>

      <h2>Settings</h2>
      <Radio legend="bits per weight" value={sel.bits} options={[[8, '8 bits'], [4, '4 bits']]} onChange={set('bits')} />
      <Radio
        legend="grid shape"
        value={sel.symmetric}
        options={[[true, 'symmetric (centred on 0)'], [false, 'asymmetric (fit to min…max)']]}
        onChange={set('symmetric')}
      />
      <Radio<Granularity>
        legend="how many weights share one grid"
        value={sel.granularity}
        options={[['per-tensor', 'whole matrix'], ['per-channel', 'each output channel'], ['group-128', 'each block of 128']]}
        onChange={set('granularity')}
      />
      <Radio
        legend="grid covers"
        value={sel.clip}
        options={[[100, 'the largest weight'], [99.9, 'the 99.9th percentile (clip the top 0.1%)']]}
        onChange={set('clip')}
      />
      {config ? <SettingsExplainer sel={sel} layer={layer} config={config} baseline={baseline} /> : <p>Config {id} is not in this sweep.</p>}

      {config && (
        <>
          <h2>1. One matrix, before and after rounding</h2>
          <p>
            Matrix:{' '}
            <select value={layerId} onChange={(e) => setLayerId(e.target.value)}>
              {data.layers.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.id} ({l.shape.join('×')})
                </option>
              ))}
            </select>{' '}
            <label>
              <input type="checkbox" checked={logY} onChange={(e) => setLogY(e.target.checked)} /> log-scale counts
            </label>
          </p>
          <Histogram layer={layer} config={config} whist={data.whist[id]?.[layerId]} logY={logY} />
          <HistogramExplainer layer={layer} config={config} logY={logY} />

          <h2>2. Which matrices take the most damage</h2>
          <p>
            Metric:{' '}
            <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
              {Object.entries(METRICS).map(([k, v]) => (
                <option key={k} value={k}>
                  {v.label}
                </option>
              ))}
            </select>
          </p>
          <Heatmap config={config} metric={metric} selected={layerId} onSelect={setLayerId} />
          <HeatmapExplainer config={config} metric={metric} />

          <h2>3. What the model writes</h2>
          <p>
            Four fixed prompts, each continued for 40 tokens. Left: the original model. Right: the same model with the
            rounded weights from the settings above. Decoding is <i>greedy</i> (always the single most likely next token)
            rather than random sampling, so if the two sides differ, the rounding caused it. The{' '}
            <mark>highlighted</mark> part is where the quantized model first departs from the original.
          </p>
          <Generations prompts={data.prompts} baseline={baseline} config={config} />
        </>
      )}

      <h2>All 24 settings ranked</h2>
      <p>
        Perplexity for every combination, best first. The original model scores {baseline.perplexity.toFixed(2)}. A few
        8-bit settings score slightly <i>below</i> that; on a 2,048-token test passage, differences of about ±1 are noise,
        not an improvement. Click a row to load it above.
      </p>
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>bits</th>
            <th>grid shape</th>
            <th>one grid per</th>
            <th>covers</th>
            <th>perplexity</th>
            <th>× original</th>
          </tr>
        </thead>
        <tbody>
          {ranked.map((c, i) => (
            <tr
              key={c.id}
              style={{ cursor: 'pointer', fontWeight: c.id === id ? 'bold' : undefined }}
              onClick={() => setSel({ bits: c.bits as 8 | 4, symmetric: c.symmetric!, granularity: c.granularity!, clip: c.clip_pct as 100 | 99.9 })}
            >
              <td>{i + 1}</td>
              <td>{c.bits}</td>
              <td>{c.symmetric ? 'symmetric' : 'asymmetric'}</td>
              <td>{{ 'per-tensor': 'whole matrix', 'per-channel': 'channel', 'group-128': '128 weights' }[c.granularity!]}</td>
              <td>{c.clip_pct === 100 ? 'max' : '99.9th pct'}</td>
              <td className="num">{c.perplexity.toFixed(1)}</td>
              <td className="num">{(c.perplexity / baseline.perplexity).toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>What this does not cover</h2>
      <p>
        This is <b>weight-only</b> quantization: only the stored weights are rounded. The <i>activations</i> (the
        numbers flowing between layers while the model runs) stay 32-bit. Rounding activations is a harder problem,
        because a few activation channels run 10–100× larger than the rest, so a single shared scale crushes everything
        else into two or three levels. LLM.int8(), SmoothQuant and AWQ are each ways around that.
      </p>
      <p>
        Only the 48 matrices inside the 12 transformer blocks are rounded. The token embeddings and output layer stay
        32-bit; that's a choice, not a requirement. In a real deployment the rounded weights stay small in memory, which
        is where the size and speed gains come from. Here they're converted back to 32-bit immediately, because the
        point is to measure the accuracy cost, not to save memory.
      </p>
    </>
  )
}

function grids(layer: LayerInfo, g: Granularity) {
  const [nIn, nOut] = layer.shape
  if (g === 'per-tensor') return { count: 1, size: nIn * nOut }
  if (g === 'per-channel') return { count: nOut, size: nIn }
  return { count: (nIn / 128) * nOut, size: 128 }
}

function SettingsExplainer({ sel, layer, config, baseline }: { sel: Selection; layer: LayerInfo; config: Config; baseline: Config }) {
  const [qmin, qmax] = qrange(sel.bits, sel.symmetric)
  const g = grids(layer, sel.granularity)
  const [nIn, nOut] = layer.shape
  const ratio = config.perplexity / baseline.perplexity
  return (
    <div className="ex">
      <p>
        <b>What these settings mean.</b> Each weight is stored as an integer from {qmin} to {qmax}, so there are{' '}
        {qmax - qmin + 1} possible values.{' '}
        {sel.symmetric
          ? 'Symmetric: the grid is centred on zero and reaches out to the largest weight in either direction, so 0 is always exactly a grid point.'
          : "Asymmetric: the grid stretches from the smallest weight to the largest, plus a 'zero point' that shifts it so real 0 still lands exactly on a grid point. No grid points are wasted on values the weights never reach."}
      </p>
      <p>
        Take the selected matrix, <code>{layer.id}</code>. It's {nIn}×{nOut} = {n(nIn * nOut)} weights.{' '}
        {sel.granularity === 'per-tensor' &&
          `With one grid for the whole matrix, all ${n(g.size)} weights share a single scale. A single unusually large weight anywhere in the matrix stretches the grid for everyone.`}
        {sel.granularity === 'per-channel' &&
          `With one grid per output channel, each of the ${n(nOut)} columns gets its own scale, shared by that column's ${n(nIn)} weights. A column is everything one output number is computed from, so a large weight now only stretches the grid for its own column.`}
        {sel.granularity === 'group-128' &&
          `With one grid per 128 weights, each column is cut into ${nIn / 128} blocks of 128, so there are ${n(g.count)} separate scales. A large weight only stretches the grid for the 127 weights next to it. The cost is storing all those scales: about ${(16 / 128).toFixed(3)} extra bits per weight if each scale is a 16-bit float.`}
      </p>
      <p>
        {sel.clip === 100
          ? 'The grid reaches exactly to the largest weight, so no weight falls off the end.'
          : 'The grid only reaches the 99.9th percentile of weight sizes. The largest 0.1% get clipped to the last grid point (big error for them), and in exchange the grid points are closer together for everyone else.'}
      </p>
      <p>
        <b>Result: perplexity {config.perplexity.toFixed(1)}</b>, against {baseline.perplexity.toFixed(1)} for the original,
        so {ratio.toFixed(2)}× worse. Perplexity measures how surprised the model is by real text: roughly, how many
        tokens it's choosing between at each step. It was measured on 2,048 tokens of <i>Pride and Prejudice</i>.{' '}
        {ratio < 1.05
          ? 'This setting is essentially lossless.'
          : ratio < 1.3
            ? 'A small loss; the text below should still read fine.'
            : ratio < 3
              ? 'A clear loss; the model still works but is noticeably worse.'
              : 'The model is broken; expect the text below to be garbage.'}
      </p>
    </div>
  )
}

function HistogramExplainer({ layer, config, logY }: { layer: LayerInfo; config: Config; logY: boolean }) {
  const m = config.layers?.[layer.id]
  if (!m) return null
  const levels = gridLevels(layer, config)
  const lo = layer.hist.bins[0]
  const hi = layer.hist.bins[layer.hist.bins.length - 1]
  const inView = levels.filter((v) => v >= lo && v <= hi).length
  const perTensor = m.scale !== undefined
  const step = perTensor ? m.scale! : m.scale_quantiles[2]
  const zero = fracToZero(layer, step)
  const nOutside = layer.hist.outside[0] + layer.hist.outside[1]
  return (
    <div className="ex">
      <p>
        <b>How to read this.</b> The horizontal axis is weight value. The vertical axis is how many weights have that
        value{logY ? ' (log scale, so small counts stay visible; untick the box to see true proportions)' : ''}.{' '}
        <b>Grey</b> is the original 32-bit weights of this one matrix: a bell shape centred on 0, where most weights are
        tiny. <b>Red</b> is the same weights after rounding. Rounded weights can only take grid values, so the red bars
        are spikes at the grid points, and the height of each spike is how many weights got snapped there.
        {perTensor && (
          <>
            {' '}
            <b>Blue lines</b> are the grid itself.
          </>
        )}
      </p>
      {perTensor ? (
        <p>
          <b>What it shows here.</b> There's one grid for the whole matrix, with a spacing of {step.toExponential(2)}.
          The plot covers the central 99.9% of weights, and {inView} of the {levels.length} grid points fall inside it.
          The rest of the grid is spent on the {n(nOutside)} outlier weights beyond the plotted range (this matrix's
          weights actually go from {layer.min.toFixed(2)} to {layer.max.toFixed(2)}). Every weight within half a step of
          zero rounds to exactly 0: about <b>{pct(zero)}</b> of this matrix.{' '}
          {zero > 0.3
            ? "That's the damage: a large share of the weights are erased outright, because a handful of outliers forced the grid to be coarse."
            : 'Most weights still land on a distinct grid point, so little is lost.'}
        </p>
      ) : (
        <p>
          <b>What it shows here.</b> Every {config.granularity === 'per-channel' ? 'column' : 'block of 128 weights'} has its
          own grid, so there's no single grid to draw (the blue lines only appear with "whole matrix"). The spacings
          range from {m.scale_quantiles[0].toExponential(1)} to {m.scale_quantiles[4].toExponential(1)}, with a median of{' '}
          {step.toExponential(2)}. Because each grid fits its own small set of weights, the spacings are fine where the
          weights are small, and the red spikes overlap into something close to the grey shape. That's what "not much
          was lost" looks like. Using the median spacing, roughly {pct(zero)} of weights round to 0.
        </p>
      )}
      <p>
        <b>Numbers for this matrix:</b> cosine similarity between original and rounded weights{' '}
        <b>{m.cos.toFixed(5)}</b> (1 means identical in direction) · mean squared error {m.mse.toExponential(2)} · worst
        single error {m.max_abs_err.toExponential(2)} · {m.pct_clipped.toFixed(3)}% of weights clipped at the grid's end ·{' '}
        {pct(m.levels_used)} of available grid points actually used (averaged over this matrix's grids; a low number means
        most of the grid sits where there are no weights).
      </p>
    </div>
  )
}

function HeatmapExplainer({ config, metric }: { config: Config; metric: Metric }) {
  const spec = METRICS[metric]
  const layers = config.layers ?? {}
  const entries = Object.entries(layers)
  const worse = (a: number, b: number) => (spec.worseHigh ? a > b : a < b)
  const [worstId, worstM] = entries.reduce((w, e) => (worse(spec.get(e[1]), spec.get(w[1])) ? e : w))
  const colMeans = COLS.map(([suffix, label]) => {
    const vs = entries.filter(([k]) => k.endsWith(suffix)).map(([, m]) => spec.get(m))
    return { label, mean: vs.reduce((a, b) => a + b, 0) / vs.length }
  })
  const worstCol = colMeans.reduce((w, c) => (worse(c.mean, w.mean) ? c : w))
  return (
    <div className="ex">
      <p>
        <b>How to read this.</b> Each of GPT-2's 12 blocks has four weight matrices that get rounded: in attention, one
        that makes the query/key/value vectors and one that combines the result; in the MLP, one that expands 768 numbers
        to 3,072 and one that shrinks them back. Each cell is one matrix, and darker red means more damage{' '}
        <i>relative to the other cells for this setting</i>. Click a cell to show that matrix in the plot above.
      </p>
      <p>
        <b>The metric.</b>{' '}
        {metric === 'one_minus_cos' &&
          'Cosine similarity compares the original and rounded matrix as two long vectors: 1 means they point exactly the same way. This shows 1 minus that, so 0 means no damage and bigger is worse.'}
        {metric === 'mse' && 'The average of (original − rounded)² over every weight in the matrix. Bigger is worse.'}
        {metric === 'pct_clipped' &&
          'The share of weights that fell off the end of the grid and were clamped to the last grid point. Always 0 unless the grid covers only the 99.9th percentile.'}
        {metric === 'levels_used' &&
          'How many of the available grid points actually have weights on them, averaged over the grids. Low means the grid was stretched by outliers and most of its points sit in empty space. Lower is worse here.'}
      </p>
      <p>
        <b>In this setting</b>, the worst matrix is <code>{worstId}</code> ({fmt(spec.get(worstM))}), and on average the{' '}
        <b>{worstCol.label}</b> column is worst. These are weight-level measurements. They don't directly say how much
        each matrix hurts the model's output, since some matrices matter more than others. Comparing this table with the
        perplexity ranking below is how you'd check whether they agree.
      </p>
    </div>
  )
}
