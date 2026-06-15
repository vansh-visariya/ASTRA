# Contributing to ASTRA

Thank you for contributing! ASTRA is a distributed federated learning platform, and we welcome improvements of all kinds — bug fixes, features, documentation, or examples.

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Git

### Development Setup

```bash
# Clone and set up the backend
git clone https://github.com/vansh-visariya/ASTRA.git
cd ASTRA
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
pre-commit install

# Set up the dashboard
cd dashboard
npm install
cd ..
```

### Copy Environment Variables

```bash
cp .env.example .env
# Edit .env — generate a real SECRET_KEY for local development
```

### Run the Stack

```bash
# Terminal 1 — Backend server
make run-server

# Terminal 2 — Dashboard
cd dashboard && npm run dev
```

Backend at `http://localhost:8000` (Swagger docs at `/docs`), dashboard at `http://localhost:3000`.

## How to Contribute

### 1. Find or Create an Issue

- Check [open issues](https://github.com/vansh-visariya/ASTRA/issues) for something you'd like to work on
- If you found a bug or have a feature idea, open a new issue first to discuss it

### 2. Branch Naming

Follow this convention:

| Type | Pattern | Example |
|------|---------|---------|
| Feature | `feat/description` | `feat/add-gpu-support` |
| Bug fix | `fix/description` | `fix/auth-token-expiry` |
| Documentation | `docs/description` | `docs/api-examples` |
| Refactor | `refactor/description` | `refactor/aggregation-pipeline` |

### 3. Make Changes

```bash
git checkout -b feat/my-feature
# ... make changes ...
make lint     # ruff check
make test     # pytest
make typecheck  # mypy
```

All three checks must pass. The CI pipeline runs the same checks on every push.

### 4. Testing

- Write tests for new features
- Update existing tests if you change behavior
- Place tests in `tests/` following the existing naming pattern
- Run: `pytest tests/ -v`

### 5. Commit Messages

Use clear, descriptive commit messages:

```
feat: add GPU support for training groups

Adds CUDA device selection in FLClient and AsyncServer.
Clients can specify --device cuda:0 to use GPU acceleration.
```

Optional conventional commit prefixes: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.

### 6. Open a Pull Request

- Push your branch and open a PR against `main`
- Describe what you changed and why
- Link any related issues
- The CI pipeline will run automatically — make sure it passes
- Request a review from a maintainer

## Code Style

This project uses:

| Tool | Purpose | Config |
|------|---------|--------|
| **Ruff** | Linting + formatting | `pyproject.toml` |
| **mypy** | Static type checking | `pyproject.toml` |
| **pre-commit** | Git hooks | `.pre-commit-config.yaml` |

### Key Conventions

- All Python source lives under `src/astra/` and is importable as `astra.*`
- Use type hints on function signatures
- Follow existing module structure — `core/` for FL algorithms, `app/` for API layer, `infra/` for transport
- Config values go through `astra.core.config.load_config()`, not hardcoded
- Prefer registry patterns over `if/elif` chains (see `astra.infra.registry`)

## Project Architecture

For architecture details, data flow diagrams, and known gotchas, see [`AGENTS.md`](AGENTS.md).

Quick overview:

```
src/astra/
├── core/       # FL algorithms (pure Python, no web deps)
├── app/        # FastAPI API layer, orchestration, DB
├── infra/      # WebSocket, auth, model registry
└── client/     # Standalone FL client CLI
```

## Questions?

- Open a [GitHub Discussion](https://github.com/vansh-visariya/ASTRA/discussions) for questions
- Report bugs via [GitHub Issues](https://github.com/vansh-visariya/ASTRA/issues)
- Check [`AGENTS.md`](AGENTS.md) for detailed codebase notes
