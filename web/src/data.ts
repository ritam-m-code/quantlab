export type Granularity = 'per-tensor' | 'per-channel' | 'group-128'

export interface LayerInfo {
  id: string
  shape: [number, number]
  min: number
  max: number
  hist: { bins: number[]; counts: number[]; outside: [number, number] }
}

export interface LayerMetrics {
  mse: number
  cos: number
  max_abs_err: number
  pct_clipped: number
  levels_used: number
  scale_quantiles: number[]
  scale?: number // per-tensor only
  zero_point?: number
}

export interface Config {
  id: string
  bits?: number
  symmetric?: boolean
  granularity?: Granularity
  clip_pct?: number
  perplexity: number
  generations: Record<string, string>
  layers?: Record<string, LayerMetrics>
}

export interface Data {
  layers: LayerInfo[]
  prompts: string[]
  configs: Config[]
  byId: Map<string, Config>
  whist: Record<string, Record<string, number[]>>
}

export interface Selection {
  bits: 8 | 4
  symmetric: boolean
  granularity: Granularity
  clip: 100 | 99.9
}

const SHORT: Record<Granularity, string> = {
  'per-tensor': 'pertensor',
  'per-channel': 'perchan',
  'group-128': 'g128',
}

/** Must match core/sweep.py make_grid ids. */
export const configId = (s: Selection) =>
  `b${s.bits}-${s.symmetric ? 'sym' : 'asym'}-${SHORT[s.granularity]}-c${s.clip}`

export const qrange = (bits: number, symmetric: boolean): [number, number] =>
  symmetric ? [-(2 ** (bits - 1) - 1), 2 ** (bits - 1) - 1] : [0, 2 ** bits - 1]

async function getJson<T>(name: string): Promise<T> {
  const res = await fetch(`${import.meta.env.BASE_URL}data/${name}`)
  if (!res.ok) throw new Error(`${name}: HTTP ${res.status}`)
  return res.json() as Promise<T>
}

export async function loadData(): Promise<Data> {
  const [layers, cfg, whist] = await Promise.all([
    getJson<{ gpt2: { layers: LayerInfo[] } }>('layers.json'),
    getJson<{ prompts: string[]; configs: Config[] }>('configs.json'),
    getJson<Data['whist']>('whist.json'),
  ])
  return {
    layers: layers.gpt2.layers,
    prompts: cfg.prompts,
    configs: cfg.configs,
    byId: new Map(cfg.configs.map((c) => [c.id, c])),
    whist,
  }
}
