# RAG Module

**Portable Knowledge Base for AI Agents**

A drop-in RAG (Retrieval Augmented Generation) system that lets AI agents remember everything. Ingest URLs, PDFs, YouTube videos, tweets, and text — then query with semantic search.

Built for [AI Installer](https://ai-installer-dashboard.vercel.app) client deployments.

## Features

- 🔗 **Multi-source ingestion**: URLs, PDFs, YouTube transcripts, Twitter/X, plain text
- 🧠 **Semantic search**: Find relevant content by meaning, not just keywords
- 🔄 **Auto-deduplication**: URL normalization + content hashing
- 📦 **Dual backends**: SQLite (local) or Supabase (cloud with pgvector)
- ⚡ **Free embeddings**: Uses Gemini (free tier) with OpenAI fallback
- 🔒 **Multi-tenant**: Isolate knowledge per agent with `RAG_AGENT_ID`

## Quick Start

### Installation

```bash
npm install
```

### Environment Setup

```bash
# .env file

# Required: Embeddings
GEMINI_API_KEY=your-gemini-key    # Free at https://aistudio.google.com/

# Optional: Fallback embeddings
OPENAI_API_KEY=your-openai-key

# Optional: Use Supabase instead of SQLite
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Optional: Multi-tenant (isolate per agent)
RAG_AGENT_ID=my-agent-name
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

## Storage Backends

### SQLite (Default)

Zero config, stores everything in a local `./rag.db` file. Perfect for single-agent setups.

```bash
# Just works - no env vars needed
node src/cli.js ingest https://example.com
```

### Supabase (Cloud)

For multi-agent deployments, cloud access, or when you need pgvector's native similarity search.

#### 1. Create Supabase Project

Go to [supabase.com](https://supabase.com) and create a free project.

#### 2. Run the SQL Schema

In your Supabase SQL editor, run:

```sql
-- Enable pgvector extension
create extension if not exists vector;

-- Sources table
create table rag_sources (
  id serial primary key,
  agent_id text not null default 'default',
  url text,
  url_normalized text,
  title text,
  source_type text not null,
  summary text,
  raw_content text,
  content_hash text,
  tags text[] default '{}',
  metadata jsonb default '{}',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Chunks table with vector embeddings
create table rag_chunks (
  id serial primary key,
  agent_id text not null default 'default',
  source_id integer references rag_sources(id) on delete cascade,
  chunk_index integer not null,
  content text not null,
  embedding vector(768),  -- Gemini embeddings are 768 dimensions
  embedding_provider text,
  embedding_model text,
  created_at timestamptz default now()
);

-- Indexes for performance
create index idx_sources_agent on rag_sources(agent_id);
create index idx_sources_hash on rag_sources(content_hash);
create index idx_sources_url on rag_sources(url_normalized);
create index idx_chunks_agent on rag_chunks(agent_id);
create index idx_chunks_source on rag_chunks(source_id);

-- Vector similarity search index (IVFFlat for speed)
create index idx_chunks_embedding on rag_chunks 
  using ivfflat (embedding vector_cosine_ops) with (lists = 100);

-- Optional: RPC function for server-side similarity search
create or replace function match_rag_chunks(
  query_embedding vector(768),
  match_count int default 5,
  filter_agent_id text default 'default'
)
returns table (
  id int,
  source_id int,
  chunk_index int,
  content text,
  similarity float,
  title text,
  url text,
  source_type text
)
language plpgsql
as $$
begin
  return query
  select
    c.id,
    c.source_id,
    c.chunk_index,
    c.content,
    1 - (c.embedding <=> query_embedding) as similarity,
    s.title,
    s.url,
    s.source_type
  from rag_chunks c
  join rag_sources s on c.source_id = s.id
  where c.agent_id = filter_agent_id
    and c.embedding is not null
  order by c.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

#### 3. Configure Environment

```bash
# .env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
GEMINI_API_KEY=your-gemini-key

# Optional: isolate this agent's data
RAG_AGENT_ID=iris
```

#### 4. Use It

```bash
# Automatically uses Supabase when env vars are set
node src/cli.js stats
# → "Using Supabase database"
```

## Multi-Tenant Setup

For AI Installer deployments with multiple client agents:

```bash
# Each agent gets isolated knowledge
RAG_AGENT_ID=client_acme node src/cli.js ingest https://acme.com/docs
RAG_AGENT_ID=client_globex node src/cli.js ingest https://globex.com/faq

# Searches only return that agent's data
RAG_AGENT_ID=client_acme node src/cli.js search "pricing"
```

All agents share the same Supabase project, but data is isolated by `agent_id`.

## Programmatic Usage

```javascript
import { RAG } from './src/index.js';

// Initialize (auto-detects SQLite vs Supabase)
const rag = new RAG({
  // Optional: override detection
  dbPath: './knowledge.db',        // SQLite path
  supabaseUrl: '...',              // Or use env vars
  supabaseKey: '...',
  
  // Chunking config
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
const sources = await rag.list({ limit: 50, sourceType: 'article' });

// Get stats
const stats = await rag.stats();

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
                    ┌──────────────────────────┤
                    │                          │
              ┌─────▼─────┐             ┌──────▼──────┐
              │  SQLite   │             │  Supabase   │
              │  (local)  │     OR      │  (pgvector) │
              └───────────┘             └─────────────┘
```

## Content Extraction

| Source Type | Detection | Extractor |
|-------------|-----------|-----------|
| Article | HTTP(S) URLs | Readability |
| YouTube | youtube.com, youtu.be | Transcript API |
| Twitter/X | twitter.com, x.com | FxTwitter API |
| PDF | .pdf files | pdf-parse |
| Text | .txt, .md, plain | Direct |

## Embedding Providers

| Provider | Model | Dimensions | Cost |
|----------|-------|------------|------|
| Gemini | gemini-embedding-001 | 768 | Free (1500/day) |
| OpenAI | text-embedding-3-small | 1536 | $0.02/1M tokens |

Gemini is preferred (free tier). OpenAI is automatic fallback.

## Configuration

| Env Variable | Description | Default |
|--------------|-------------|---------|
| `GEMINI_API_KEY` | Gemini embedding API key | - |
| `OPENAI_API_KEY` | OpenAI fallback API key | - |
| `SUPABASE_URL` | Supabase project URL | - |
| `SUPABASE_KEY` | Supabase anon key | - |
| `RAG_AGENT_ID` | Multi-tenant agent ID | `default` |
| `RAG_DB_PATH` | SQLite database path | `./rag.db` |

## License

MIT
