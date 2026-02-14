#!/usr/bin/env node
/**
 * RAG CLI - Command line interface for the knowledge base
 */

import { RAG } from './rag.js';
import { readFileSync } from 'fs';
import { resolve } from 'path';

// Load environment variables from .env if exists
try {
  const envPath = resolve(process.cwd(), '.env');
  const envContent = readFileSync(envPath, 'utf-8');
  envContent.split('\n').forEach(line => {
    const [key, ...valueParts] = line.split('=');
    if (key && valueParts.length) {
      process.env[key.trim()] = valueParts.join('=').trim().replace(/^["']|["']$/g, '');
    }
  });
} catch { /* no .env file */ }

const args = process.argv.slice(2);
const command = args[0];

function printUsage() {
  console.log(`
RAG Knowledge Base CLI

Usage:
  rag ingest <url|file|text> [--tags tag1,tag2] [--db path]
  rag search <query> [--top K] [--db path]
  rag query <question> [--top K] [--db path]
  rag list [--type article|video|pdf|text] [--limit N] [--db path]
  rag stats [--db path]
  rag delete <id> [--db path]

Environment Variables:
  GEMINI_API_KEY    - Google Gemini API key (free embeddings)
  GOOGLE_API_KEY    - Alternative to GEMINI_API_KEY
  OPENAI_API_KEY    - OpenAI API key (fallback embeddings)
  RAG_DB_PATH       - Default database path

Examples:
  rag ingest https://example.com/article
  rag ingest ./document.pdf --tags research,ai
  rag search "machine learning basics"
  rag query "What is RAG?"
  rag list --type video --limit 20
`);
}

function parseArgs(args) {
  const opts = {};
  let positional = [];
  
  for (let i = 0; i < args.length; i++) {
    if (args[i].startsWith('--')) {
      const key = args[i].slice(2);
      const value = args[i + 1] && !args[i + 1].startsWith('--') ? args[++i] : true;
      opts[key] = value;
    } else {
      positional.push(args[i]);
    }
  }
  
  return { opts, positional };
}

async function main() {
  if (!command || command === 'help' || command === '--help') {
    printUsage();
    process.exit(0);
  }

  const { opts, positional } = parseArgs(args.slice(1));
  const dbPath = opts.db || process.env.RAG_DB_PATH || './rag.db';
  
  const rag = new RAG({ dbPath });

  try {
    switch (command) {
      case 'ingest': {
        const input = positional[0];
        if (!input) {
          console.error('Error: No input provided');
          process.exit(1);
        }
        const tags = opts.tags ? opts.tags.split(',').map(t => t.trim()) : [];
        const result = await rag.ingest(input, { tags });
        console.log('\nResult:', JSON.stringify(result, null, 2));
        break;
      }

      case 'search': {
        const query = positional.join(' ');
        if (!query) {
          console.error('Error: No query provided');
          process.exit(1);
        }
        const topK = parseInt(opts.top) || 5;
        const results = await rag.search(query, { topK });
        console.log('\nResults:');
        results.forEach((r, i) => {
          console.log(`\n${i + 1}. ${r.title} (${(r.similarity * 100).toFixed(1)}% match)`);
          console.log(`   Type: ${r.sourceType} | URL: ${r.url || 'N/A'}`);
          console.log(`   ${r.content.slice(0, 200)}...`);
        });
        break;
      }

      case 'query': {
        const question = positional.join(' ');
        if (!question) {
          console.error('Error: No question provided');
          process.exit(1);
        }
        const topK = parseInt(opts.top) || 5;
        const { results, prompt } = await rag.query(question, { topK });
        
        console.log('\nRelevant Sources:');
        results.forEach((r, i) => {
          console.log(`  ${i + 1}. ${r.title} (${(r.similarity * 100).toFixed(1)}%)`);
        });
        
        console.log('\n--- LLM Prompt ---');
        console.log(prompt);
        break;
      }

      case 'list': {
        const limit = parseInt(opts.limit) || 20;
        const sourceType = opts.type;
        const sources = rag.list({ limit, sourceType });
        
        console.log(`\nSources (${sources.length}):`);
        sources.forEach(s => {
          console.log(`  [${s.id}] ${s.source_type.padEnd(8)} ${s.title?.slice(0, 50) || s.url?.slice(0, 50)}`);
        });
        break;
      }

      case 'stats': {
        const stats = rag.stats();
        console.log('\nKnowledge Base Stats:');
        console.log(`  Total Sources: ${stats.totalSources}`);
        console.log(`  Total Chunks: ${stats.totalChunks}`);
        console.log('  By Type:');
        Object.entries(stats.byType).forEach(([type, count]) => {
          console.log(`    ${type}: ${count}`);
        });
        break;
      }

      case 'delete': {
        const id = parseInt(positional[0]);
        if (!id) {
          console.error('Error: No source ID provided');
          process.exit(1);
        }
        const result = rag.delete(id);
        console.log(result.changes ? `Deleted source ${id}` : `Source ${id} not found`);
        break;
      }

      default:
        console.error(`Unknown command: ${command}`);
        printUsage();
        process.exit(1);
    }
  } finally {
    rag.close();
  }
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});
