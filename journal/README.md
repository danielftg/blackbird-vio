# journal/ — the research behind the code

```
journal/
├── notes/   reasoning, research, evaluation — worked out first
└── paper/   the formal write-up of the current state of src/ — written second
```

Order of motion: an idea gets worked out in `notes/` (especially `notes/design/`),
written up formally in `paper/`, then `src/` is built or changed to match `paper/`.
When something changes about how `src/` works, `paper/` changes first — `paper/` is
meant to always describe the current state of `src/`, not where it's headed next.

## `notes/`

Working notes, not written for an audience. Source, then our synthesis of it,
then the current blueprint — this is the reasoning, research, and evaluation
that `paper/` gets written from:

- **`survey/`** — papers and other sources read, with review notes and searchable extracts.
- **`synth/`** — our own synthesis of the survey into a coherent picture of the theory and
  the problem. Start here: [`notes/synth/00-vision.md`](notes/synth/00-vision.md) (what
  we're building and why) and [`notes/synth/01-problem.md`](notes/synth/01-problem.md)
  (the problem being solved).
- **`design/`** — the next blueprint: what we intend `src/` and `paper/` to become,
  as currently decided. Ahead of `paper/` until it's written up and built.
- **`scratch/`** — working notes for whatever task is active right now.

## `paper/`

The polished, outward-facing version of the system as it currently stands —
Lie algebra, the EKF, the solver, the measurement model, results. Read
[`paper/main.pdf`](paper/main.pdf) for the full picture in one document. Lags
`notes/design/` by however long the next build takes.

## What's public

This repository is public, but most of the research process behind it isn't: only
`paper/main.pdf` and `notes/synth/` are version-controlled and pushed to GitHub.
`notes/survey/`, `notes/design/`, and `notes/scratch/` are local working material,
kept off the public remote.
