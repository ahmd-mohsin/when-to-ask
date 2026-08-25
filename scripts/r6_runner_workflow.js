export const meta = {
  name: 'r6-llm-cells',
  description: 'Run R6a/R6b judgment chunks via Fable subagents (028 T1, registry-blind)',
  phases: [
    { title: 'R6a', detail: 'single-run introspection chunks' },
    { title: 'R6b', detail: 'ensemble comparison chunks' },
  ],
}

// args: { chunkDir: string, r6a: ["payload_r6a_chunk_00.json", ...], r6b: [...] }
const DIR = args.chunkDir

const R6A_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { judgments: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: {
      item_id: { type: 'string' },
      ask: { type: 'boolean' },
      unresolved_decision: { type: 'string' },
      confidence: { type: 'integer', minimum: 1, maximum: 5 },
    }, required: ['item_id', 'ask', 'unresolved_decision', 'confidence'] } } },
  required: ['judgments'],
}

const R6B_SCHEMA = {
  type: 'object', additionalProperties: false,
  properties: { judgments: { type: 'array', items: {
    type: 'object', additionalProperties: false,
    properties: {
      item_id: { type: 'string' },
      different: { type: 'boolean' },
      confidence: { type: 'integer', minimum: 1, maximum: 5 },
    }, required: ['item_id', 'different', 'confidence'] } } },
  required: ['judgments'],
}

const GUARD = `INTEGRITY RULES (pre-registered protocol, decisions/028):
- Use the Read tool EXACTLY ONCE, on the payload file named below. Do not
  open ANY other file, do not run shell commands, do not search the
  repository. Your judgment must come only from the payload contents.
- Judge every item independently and honestly. Do not let one item's answer
  influence another's. Do not skip items.
- You are measuring, not helping: there is no "right" base rate; answer
  each item on its own evidence.`

function chunkPrompt(file, kind) {
  const what = kind === 'r6a'
    ? `For each item, mentally render the prompt_template with that item's
"instruction" and "prefix" fields and answer it: ask (boolean),
unresolved_decision (one sentence if ask=true, else ""), confidence (1-5).`
    : `For each item, mentally render the prompt_template with that item's
"excerpt_a" and "excerpt_b" fields and answer it: different (boolean),
confidence (1-5).`
  return `${GUARD}

Read the file: ${DIR}/${file}

It contains {prompt_template, items}. ${what}

Return judgments for ALL items, in the same order as the file, via structured output.`
}

phase('R6a')
const r6aResults = await parallel((args.r6a ?? []).map(f => () =>
  agent(chunkPrompt(f, 'r6a'), { label: f, phase: 'R6a', model: 'fable', schema: R6A_SCHEMA })
    .then(r => ({ file: f, judgments: r?.judgments ?? null }))))

phase('R6b')
const r6bResults = await parallel((args.r6b ?? []).map(f => () =>
  agent(chunkPrompt(f, 'r6b'), { label: f, phase: 'R6b', model: 'fable', schema: R6B_SCHEMA })
    .then(r => ({ file: f, judgments: r?.judgments ?? null }))))

const ok = r => r && r.judgments
log(`r6a: ${r6aResults.filter(ok).length}/${(args.r6a ?? []).length} chunks; ` +
    `r6b: ${r6bResults.filter(ok).length}/${(args.r6b ?? []).length} chunks`)
return { r6a: r6aResults.filter(Boolean), r6b: r6bResults.filter(Boolean) }
