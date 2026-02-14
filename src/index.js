/**
 * RAG Module - Portable Knowledge Base for AI Agents
 * 
 * @example
 * import { RAG } from 'rag-module';
 * 
 * const rag = new RAG({ dbPath: './knowledge.db' });
 * 
 * // Ingest content
 * await rag.ingest('https://example.com/article');
 * await rag.ingest('./document.pdf');
 * 
 * // Search
 * const results = await rag.search('machine learning');
 * 
 * // Query with LLM prompt
 * const { results, prompt } = await rag.query('What is RAG?');
 */

export { RAG } from './rag.js';
export { RagDatabase } from './db.js';
export { EmbeddingProvider, cosineSimilarity } from './embeddings.js';
export { extract, detectSourceType } from './extractors.js';
export { chunkText, createChunks } from './chunker.js';

// Default export
export { RAG as default } from './rag.js';
