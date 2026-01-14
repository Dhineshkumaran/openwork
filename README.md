# openwork

A tactical agent interface for [deepagentsjs](https://github.com/langchain-ai/deepagentsjs) - an opinionated harness for building deep agents with filesystem capabilities, planning, and subagent delegation.

![openwork screenshot](docs/screenshot.png)

## Features

- **Chat Interface** - Stream conversations with your AI agent in real-time
- **TODO Tracking** - Visual task list showing agent's planning progress
- **Filesystem Browser** - See files the agent reads, writes, and edits
- **Subagent Monitoring** - Track spawned subagents and their status
- **Human-in-the-Loop** - Approve, edit, or reject sensitive tool calls
- **Multi-Model Support** - Use Claude, GPT-4, Gemini, or local models
- **Thread Persistence** - SQLite-backed conversation history

## Installation

### npm (recommended)

```bash
# Run directly
npx openwork

# Or install globally
npm install -g openwork
openwork
```

Requires Node.js 18+. Electron is installed automatically as a dependency.

### From Source

```bash
git clone https://github.com/langchain-ai/openwork.git
cd openwork
pnpm install
pnpm run dev
```

## Configuration

### API Keys

openwork supports multiple LLM providers. Set your API keys via:

1. **Environment Variables** (recommended)
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-..."
   export OPENAI_API_KEY="sk-..."
   export GOOGLE_API_KEY="..."
   ```

2. **In-App Settings** - Click the settings icon and enter your API keys securely.

### Supported Models

| Provider | Models |
|----------|--------|
| Anthropic | Claude Sonnet 4, Claude 3.5 Sonnet, Claude 3.5 Haiku |
| OpenAI | GPT-4o, GPT-4o Mini |
| Google | Gemini 2.0 Flash |

## Architecture

openwork is built with:

- **Electron** - Cross-platform desktop framework
- **React** - UI components with tactical/SCADA-inspired design
- **deepagentsjs** - Agent harness with planning, filesystem, and subagents
- **LangGraph** - State machine for agent orchestration
- **SQLite** - Local persistence for threads and checkpoints

```
┌─────────────────────────────────────────────────────────────┐
│                     Electron Main Process                    │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │ IPC Handlers│  │   SQLite    │  │   DeepAgentsJS      │ │
│  │  - agent    │  │  - threads  │  │   - createAgent     │ │
│  │  - threads  │  │  - runs     │  │   - checkpointer    │ │
│  │  - models   │  │  - assists  │  │   - middleware      │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                         IPC Bridge
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Electron Renderer Process                 │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌─────────────────────┐  ┌───────────────┐  │
│  │ Sidebar  │  │    Chat Interface   │  │  Right Panel  │  │
│  │ - Threads│  │  - Messages         │  │  - TODOs      │  │
│  │ - Model  │  │  - Tool Renderers   │  │  - Files      │  │
│  │ - Config │  │  - Streaming        │  │  - Subagents  │  │
│  └──────────┘  └─────────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Development

```bash
# Install dependencies
pnpm install

# Start development server
pnpm run dev

# Build for production
pnpm run build
```

## Releases

To publish a new release:

1. Create a git tag: `git tag v0.2.0`
2. Push the tag: `git push origin v0.2.0`
3. GitHub Actions will:
   - Build the application
   - Publish to npm
   - Create a GitHub release

## Design System

openwork uses a tactical/SCADA-inspired design system optimized for:

- **Information density** - Dense layouts for monitoring agent activity
- **Status at a glance** - Color-coded status indicators (nominal, warning, critical)
- **Dark mode only** - Reduced eye strain for extended sessions
- **Monospace typography** - JetBrains Mono for data and code

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.
