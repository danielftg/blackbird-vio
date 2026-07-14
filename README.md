# blackbird-vio

**Work in progress.** We're building a tactical search-and-pursuit drone — short-to-medium
range, low/medium altitude relative to the ground surface, indoor or outdoor, needing
nothing more than a free flight corridor. It doesn't interact with its environment
(can't open a door, land somewhere non-trivial, etc.). This repo is the current stage
of that build; what "current" specifically means keeps changing — the full picture is
in [journal/paper/main.pdf](journal/paper/main.pdf).

## Map of this repo

```
blackbird-vio/
├── src/       the running code — whatever's currently built
└── journal/   the research behind it
    ├── notes/   reasoning, research, evaluation — worked out first
    └── paper/   the formal write-up of the current state of src/ — written second
```

Order of motion: an idea gets worked out in `notes/`, written up formally in `paper/`,
then `src/` is built or changed to match `paper/`. When something changes about how
`src/` works, `paper/` changes first.

## Getting started

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Then see **[src/README.md](src/README.md)** for what's currently in `src/` and how to run it.

## Going deeper

| Question | Where |
|---|---|
| What's actually in `src/` right now, and how do I run it? | [src/README.md](src/README.md) |
| What are we building, and why this way? | [journal/README.md](journal/README.md) |
| What's the current formal write-up? | [journal/paper/main.pdf](journal/paper/main.pdf) |

## License

MIT — see [LICENSE](LICENSE).
