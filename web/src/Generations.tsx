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
    <>
      {prompts.map((prompt, i) => {
        const key = `prompt_${i}`
        const ref = baseline.generations[key] ?? ''
        const out = config.generations[key] ?? ''
        const n = commonPrefix(ref, out)
        return (
          <div key={key}>
            <p>
              <b>Prompt:</b> <code>{prompt}</code>
            </p>
            <div className="cols">
              <div>
                <i>original (fp32)</i>
                <pre>{ref}</pre>
              </div>
              <div>
                <i>{config.id}</i>
                <pre>
                  {out.slice(0, n)}
                  <mark>{out.slice(n)}</mark>
                </pre>
              </div>
            </div>
            <hr />
          </div>
        )
      })}
    </>
  )
}
