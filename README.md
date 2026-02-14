# RAG Module

**Portable Knowledge Base for AI Agents**

A drop-in RAG (Retrieval Augmented Generation) system that lets AI agents remember everything. Ingest URLs, PDFs, YouTube videos, tweets, and text — then query with semantic search.

Built for [AI Installer](https://ai-installer-dashboard.vercel.app) client deployments.

## Features

- 🔗 **Multi-source ingestion**: URLs, PDFs, YouTube transcripts, Twitter/X, plain text
- 🧠 **Semantic search**: Find relevant content by meaning, not just keywords
- 🔄 **Auto-deduplication**: URL normalization + content hashing
- 📦 **Portable**: Single SQLite database, no external dependencies
- ⚡ **Free embeddings**: Uses Gemini (free tier) with OpenAI fallback
- 🔒 **Concurrency safe**: Lock file prevents parallel ingestion issues

## Quick Start

### Installation

```bash
npm install
```

### Environment Setup

```bash
# .env file
GEMINI_API_KEY=your-gemini-key    # Free at https://aistudio.google.com/
OPENAI_API_KEY=your-openai-key    # Optional fallback
```

### CLI Usage

```bash
# Ingest content
node src/cli.js ingest https://example.com/article
node src/cli.js ingest ./document.pdf --tags research,ai
node src/cli.js ingest "Some important text to remember"

# Search
node src/cli.js search "machine learning basics"

# Query (returns LLM-ready prompt with sources)
node src/cli.js query "What is RAG?"

# List sources
node src/cli.js list --type video --limit 20

# Stats
node src/cli.js stats

# Delete a source
node src/cli.js delete 5
```

### Programmatic Usage

```javascript
import { RAG } from './src/index.js';

// Initialize
const rag = new RAG({
  dbPath: './knowledge.db',
  geminiKey: process.env.GEMINI_API_KEY,
  // Optional config
  chunkSize: 800,
  overlap: 200
});

// Ingest content
await rag.ingest('https://example.com/article');
await rag.ingest('./document.pdf', { tags: ['research'] });

// Search for relevant chunks
const results = await rag.search('machine learning', { topK: 5 });

// Query with LLM-ready prompt
const { results, prompt, context } = await rag.query('What is RAG?');
// Pass `prompt` to your LLM for a grounded answer

// List all sources
const sources = rag.list({ limit: 50, sourceType: 'article' });

// Get stats
const stats = rag.stats();

// Cleanup
rag.close();
```

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Extractor  │────▶│   Chunker   │────▶│  Embedder   │
│  (content)  │     │  (800 char) │     │  (Gemini)   │
└─────────────┘     └─────────────┘     └─────────────┘
                                               │
                                               ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Search    │◀────│  SQLite DB  │◀────│   Store     │
│  (cosine)   │     │  (sources,  │     │  (chunks)   │
└─────────────┘     │   chunks)   │     └─────────────┘
                    └─────────────┘
```

## Content Extraction

| Source Type | Detection | Extractor |
|-------------|-----------|-----------|
| Article | HTTP(S) URLs | Readability |
| YouTube | youtube.com, youtu.be | Transcript API |
| Twitter/X | twitter.com, x.com | FxTwitter API |
| PDF | .pdf files | pdf-parse |
| Text | .txt, .md, plain | Direct |

## Deduplication

Two-layer deduplication prevents storing the same content twice:

1. **URL normalization**: Strips tracking params (utm_*, fbclid, etc.), normalizes domains
2. **Content hashing**: SHA-256 hash of extracted content

## Embedding Providers

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| Gemini | text-embedding-004 | 768 | Free |
| OpenAI | text-embedding-3-small | 1536 | $0.02/1M tokens |

Gemini is preferred (free tier). OpenAI is automatic fallback.

## Configuration

```javascript
const rag = new RAG({
  // Database
  dbPath: './rag.db',           // SQLite database path
  
  // Embeddings
  geminiKey: '...',             // Or GEMINI_API_KEY env var
  openaiKey: '...',             // Or OPENAI_API_KEY env var
  preferredProvider: 'gemini',  // 'gemini' or 'openai'
  
  // Chunking
  chunkSize: 800,               // Characters per chunk
  overlap: 200                  // Overlap between chunks
});
```

## Client Installation

For AI Installer deployments, include this module in the agent setup:

```bash
# In client's agent directory
npm install /path/to/rag-module

# Or from git
npm install github:ai-installer/rag-module
```

Then wire it into the agent's tools:

```javascript
// agent-tools.js
import { RAG } from 'rag-module';

const rag = new RAG({ dbPath: './client-knowledge.db' });

export const tools = {
  save_to_knowledge_base: async (url) => {
    return await rag.ingest(url);
  },
  
  search_knowledge_base: async (query) => {
    return await rag.search(query, { topK: 5 });
  },
  
  ask_knowledge_base: async (question) => {
    const { prompt } = await rag.query(question);
    // Pass to LLM...
  }
};
```

## License

MIT
