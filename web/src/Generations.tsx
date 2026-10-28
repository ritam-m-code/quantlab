import type { Config } from './data'

/** Length of the shared prefix, backed up to a word boundary. */
function commonPrefix(a: string, b: string) {
  let i = 0
  while (i < a.length && i < b.length && a[i] === b[i]) i++
  if (i < a.length || i < b.length) while (i > 0 && !/\s/.test(a[i - 1])) i--
  return i
}

interface Props {
  prompts: string[]
  baseline: Config
  config: Config
}

export function Generations({ prompts, baseline, config }: Props) {
  return (
    <div className="gens">
      {prompts.map((prompt, i) => {
        const key = `prompt_${i}`
        const ref = baseline.generations[key] ?? ''
        const out = config.generations[key] ?? ''
        const n = commonPrefix(ref, out)
        return (
          <div className="gen" key={key}>
            <div className="gen-prompt">{prompt}</div>
            <div className="gen-cols">
              <div>
                <div className="gen-label">fp32</div>
                <pre>{ref}</pre>
              </div>
              <div>
                <div className="gen-label">{config.id}</div>
                <pre>
                  {out.slice(0, n)}
                  <mark>{out.slice(n)}</mark>
                </pre>
              </div>
            </div>
          </div>
        )
      })}
      <p className="muted small">Greedy decoding, so any difference is caused by the quantized weights, not sampling. Highlight marks where the text first departs from fp32.</p>
    </div>
  )
}
