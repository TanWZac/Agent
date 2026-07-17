# Free Claude Code Setup Guide (GitHub)

This guide helps you set up a free-claude-code style project from GitHub with two provider options:

1. Azure AI Foundry (OpenAI-compatible endpoint)
2. OpenRouter using NVIDIA Nemotron 3 Ultra

## 1. Clone and Install

```bash
git clone https://github.com/<owner>/free-claude-code.git
cd free-claude-code

# If Python-based
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# If Node-based (use this instead of pip when package.json exists)
npm install
```

## 2. Pick One LLM Provider

Many free-claude-code projects support OpenAI-compatible settings. If your project uses different env names, map the same values to its expected variables.

### Option A: Azure AI Foundry (OpenAI-compatible)

Use this when you have an Azure endpoint and deployment already created.

```dotenv
# Provider switch (if your project has one)
LLM_PROVIDER=azure_openai

# Azure
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=<your-deployment-name>
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_API_KEY=<you-add-this>

# Optional generation settings
OPENAI_TEMPERATURE=0
```

Notes:
- Use the Azure OpenAI-style endpoint, not the AI Foundry project URL path.
- Deployment must match the model deployment name in Azure.

### Option B: OpenRouter with Nemotron 3 Ultra

Use this for a free model route on OpenRouter.

```dotenv
# For OpenAI-compatible clients
OPENAI_BASE_URL=https://openrouter.ai/api/v1
OPENAI_API_KEY=<you-add-this>
OPENAI_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free

# Optional but recommended by OpenRouter
OPENROUTER_HTTP_REFERER=https://github.com/<your-user>/<your-repo>
OPENROUTER_APP_TITLE=free-claude-code

# Optional generation settings
OPENAI_TEMPERATURE=0
```

If your code supports custom headers, send:
- HTTP-Referer: value of OPENROUTER_HTTP_REFERER
- X-OpenRouter-Title: value of OPENROUTER_APP_TITLE

## 3. Verify Connectivity

### OpenRouter quick check

```bash
curl https://openrouter.ai/api/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "messages": [{"role": "user", "content": "Reply with: ok"}]
  }'
```

### Azure quick check

```bash
curl "${AZURE_OPENAI_ENDPOINT}openai/deployments/${AZURE_OPENAI_DEPLOYMENT}/chat/completions?api-version=${AZURE_OPENAI_API_VERSION}" \
  -H "Content-Type: application/json" \
  -H "api-key: ${AZURE_OPENAI_API_KEY}" \
  -d '{
    "messages": [{"role": "user", "content": "Reply with: ok"}],
    "temperature": 0
  }'
```

## 4. Run the App

Use the command expected by your repo:

```bash
# Common Python patterns
python main.py
python -m app

# Common Node patterns
npm run dev
npm start
```

## 5. Troubleshooting

- 401 from OpenRouter:
  - Check OPENAI_API_KEY and confirm billing or free-tier availability.
- 404 model on OpenRouter:
  - Confirm the slug is exactly nvidia/nemotron-3-ultra-550b-a55b:free.
- 404 from Azure:
  - Usually endpoint/deployment mismatch. Re-check AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_DEPLOYMENT.
- Project ignores env vars:
  - Check its README for exact variable names and map the same values.

## 6. Recommended Fallbacks

If Nemotron 3 Ultra free route is temporarily unavailable, try:

1. nvidia/nemotron-3-super-120b-a12b:free
2. nvidia/nemotron-3-nano-30b-a3b:free
