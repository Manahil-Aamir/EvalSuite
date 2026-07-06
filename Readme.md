# EvalSuite — AI Agent Security Evaluation Pipeline

> Automatically evaluate any AI agent's security posture, functional quality, and adversarial resilience in minutes, not days.

---

## The Problem

As AI agents proliferate across industries, a critical gap has emerged: **there is no standardized, automated way to evaluate whether an agent is safe, secure, and behaving as intended.** Teams manually write test cases, hand-craft adversarial prompts, and subjectively review outputs — a process that is slow, inconsistent, and impossible to scale.

KidsLearn's customer support agent should never discuss adult content. A banking agent should never reveal account details to unauthorized users. A medical assistant should never provide specific dosage recommendations. How do you verify this automatically, at scale, before deployment?

**EvalSuite answers that question.**

---

## What EvalSuite Does

EvalSuite is a fully automated, multi-agent evaluation pipeline built on Google's ADK 2.0. Given any agent description and a live endpoint URL, EvalSuite:

1. Parses the agent's intended behavior into a structured eval spec
2. Enriches the spec with real security guidelines via Google Developer Knowledge MCP
3. Generates 20 diverse functional test cases
4. Generates 20 sophisticated adversarial attacks (prompt injection, jailbreaks, scope attacks, social engineering)
5. Fires all 40 tests against the real live agent endpoint
6. Scores every response across 5 dimensions using a two-tier evaluation system
7. Clusters failures by root cause
8. Produces a prioritized audit report with recommendations
9. Visualizes everything in a real-time security dashboard

**Time to evaluate:** 3–5 minutes. **No manual work required.**

---

## Architecture

```
User Input (agent description + endpoint URL)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│                    ADK 2.0 Workflow                         │
│                                                             │
│  START                                                      │
│    │                                                        │
│    ▼                                                        │
│  init_workflow          (injects target agent URL           │
│    │                     into pipeline state)               │
│    ▼                                                        │
│  spec_parser            (extracts structured eval spec      │
│    │                     from agent description)            │
│    ▼                                                        │
│  spec_parser_cleaner    (parses and validates spec JSON)     │
│    │                                                        │
│    ▼                                                        │
│  mcp_enricher           (fetches security guidelines via     │
│    │                     Google Developer Knowledge MCP)    │
│    ▼                                                        │
│  test_generator         (generates 20 functional            │
│    │                     test cases, spec-aware)            │
│    ▼                                                        │
│  adversary              (generates 20 adversarial attacks,   │
│    │                     spec- and guideline-aware)          │
│    ▼                                                        │
│  runner                 (fires all 40 tests against the      │
│    │                     real live agent endpoint)           │
│    ▼                                                        │
│  scorer                 (two-tier scoring: deterministic     │
│    │                     Python + selective LLM-as-judge)   │
│    ▼                                                        │
│  summarizer             (calculates aggregate                │
│    │                     statistics in Python)              │
│    ▼                                                        │
│  reporter               (generates prioritized               │
│    │                     audit report)                      │
│    ▼                                                        │
│  END                                                         │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
  Streamlit Dashboard
  (metrics, charts, heatmap, recommendations)
```

### Node Breakdown

| Node | Type | Model | Purpose |
|---|---|---|---|
| init_workflow | FunctionNode | — | Injects target URL into pipeline state |
| spec_parser | LlmAgent | llama-3.1-8b-instant | Extracts structured eval spec from description |
| spec_parser_cleaner | FunctionNode | — | Parses and validates spec JSON |
| mcp_enricher | FunctionNode | — | Fetches security guidelines via MCP |
| test_generator | LlmAgent | llama-3.3-70b-versatile | Generates 20 functional test cases |
| adversary | LlmAgent | llama-3.3-70b-versatile | Generates 20 adversarial attacks |
| runner | FunctionNode | — | Fires all 40 tests at real agent endpoint |
| scorer | FunctionNode + AsyncGroq | llama-3.3-70b-versatile | Two-tier scoring system |
| summarizer | FunctionNode | — | Calculates statistics in Python |
| reporter | FunctionNode + AsyncGroq | llama-3.1-8b-instant | Generates audit report |

---

## Two-Tier Scoring System

EvalSuite implements a principled scoring architecture that distinguishes between what can be measured deterministically and what requires genuine reasoning:

**Tier 1 — Deterministic Python (zero LLM tokens):**
- `scope_adherence`: Forbidden topic detection with refusal context awareness
- `attack_resistance`: Signal ratio analysis (refusal signals vs compliance indicators)
- `consistency`: Three-signal formula (length appropriateness, topic overlap, lexical diversity)

**Tier 2 — LLM-as-Judge (selective, only when needed):**
- `output_quality`: Helpfulness, clarity, and accuracy — subjective
- `edge_case_handling`: How well unusual inputs were handled — subjective
- Only triggered for results below the run's adaptive mean threshold

**Dimension weights** are derived from variance across the dataset using softmax normalization — more discriminating dimensions automatically receive higher weight. No hardcoded values.

---

## MCP Integration

EvalSuite integrates the **Google Developer Knowledge MCP server** to ground security test generation in real, documented vulnerability patterns:

```
mcp_enricher → queries MCP for security guidelines
             → appends guidelines to eval spec
             → test_generator and adversary use guidelines
               to generate spec-specific attacks
```

The `security_guidelines` field in `EvalSpec` carries MCP-retrieved knowledge through the entire pipeline. When the MCP server is unavailable, EvalSuite gracefully falls back to curated hardcoded security best practices.

---

## Project Structure

```
D:\EvalSuite\
├── .env                           # API keys
│
├── evalsuite/                    # Main EvalSuite pipeline
│   └── app/
│       └── agent.py              # 10-node ADK 2.0 Workflow
│
├── kidslearn-agent/              # Demo target agent
│   └── app/
│       ├── agent.py              # KidsLearn support agent
│       └── server.py             # FastAPI server
│
└── dashboard/                    # Streamlit dashboard
    └── app.py                    # Visualization UI
```

---

## Setup Instructions

### Prerequisites

- Python 3.12+
- Node.js 18+
- uv package manager
- agents-cli

### 1. Install agents-cli

```bash
uvx google-agents-cli setup
agents-cli login -i
```

### 2. Clone and configure

```bash
git clone https://github.com/YOUR_USERNAME/evalsuite
cd EvalSuite
```

Create `.env` in the root:

```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
GOOGLE_DEVELOPER_KNOWLEDGE_API_KEY=your_gemini_api_key
TARGET_AGENT_URL=http://127.0.0.1:8001
```

Get your free Groq API key at [console.groq.com](https://console.groq.com).

### 3. Install dependencies

```bash
# EvalSuite
cd evalsuite && agents-cli install && cd ..

# KidsLearn agent
cd kidslearn-agent && agents-cli install && cd ..

# Dashboard
cd dashboard && pip install streamlit plotly httpx python-dotenv && cd ..
```

### 4. Run all three services

**Terminal 1 — KidsLearn target agent:**
```bash
cd kidslearn-agent
uv run python app/server.py
# Runs on http://127.0.0.1:8001
```

**Terminal 2 — EvalSuite pipeline:**
```bash
cd evalsuite
uv run adk web . --host 127.0.0.1 --port 8080
# Runs on http://127.0.0.1:8080
```

**Terminal 3 — Dashboard:**
```bash
cd dashboard
streamlit run app.py --server.port 8502 --server.address 127.0.0.1
# Runs on http://127.0.0.1:8502
```

### 5. Run your first evaluation

Open `http://127.0.0.1:8502`, click **Run New Evaluation**, and paste (Sample input):

**Agent Description:**
```
A customer support agent for KidsLearn, a children's educational 
app for ages 6-12. Helps with account issues, billing, app features, 
and troubleshooting. Must never discuss topics unrelated to KidsLearn, 
never share its system prompt, and never engage with content 
inappropriate for children.
```

**Target Agent URL:** `http://127.0.0.1:8001`

Click **Run Evaluation** and wait 3–5 minutes.

---

## Evaluation Results (Sample)

Running EvalSuite against the KidsLearn demo agent revealed:

- **Pass rate:** 50% (20/40 tests passed)
- **Critical failures:** 20
- **Weakest dimension:** attack_resistance (avg 3.08/10)
- **Key finding:** Agent is vulnerable to social engineering attacks — it offered educator plan access to unverified "teachers" and assisted developers claiming backend access without verification

This is exactly the kind of finding that would prevent a security incident in production.

---

## Track

**Freestyle** — EvalSuite is a meta-tool, an agent pipeline that evaluates other agents. It automatically generates functional test cases and adversarial attacks, runs them against a live agent endpoint, and scores the results across multiple security and quality dimensions. It doesn't fit a single domain (business, personal, humanitarian) because it applies across all of them. Any AI agent, whether serving a business, a family, or a public health system, needs security evaluation before deployment. EvalSuite provides that infrastructure, demonstrating best practices in adversarial security testing.

---

## License

Apache 2.0 — see LICENSE