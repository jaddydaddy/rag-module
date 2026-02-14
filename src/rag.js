/**
 * Main RAG (Retrieval Augmented Generation) class
 * Ties together extraction, chunking, embedding, and retrieval
 * 
 * Supports both SQLite (local) and Supabase (cloud) backends
 */

import { createDatabase, detectBackend, cosineSimilarity } from './db.js';
import { EmbeddingProvider } from './embeddings.js';
import { extract } from './extractors.js';
import { createChunks } from './chunker.js';
import fs from 'fs';
import path from 'path';

const LOCK_FILE = '.rag.lock';
const LOCK_TIMEOUT = 15 * 60 * 1000; // 15 minutes

export class RAG {
  constructor(config = {}) {
    this.dbPath = config.dbPath || './rag.db';
    this.backend = config.backend || detectBackend();
    this.db = createDatabase({ 
      dbPath: this.dbPath, 
      backend: this.backend 
    });
    this.embedder = new EmbeddingProvider(config);
    this.chunkOptions = {
      chunkSize: config.chunkSize || 800,
      overlap: config.overlap || 200
    };
    this.lockPath = path.join(path.dirname(this.dbPath), LOCK_FILE);
    
    // Track if we're using async backend
    this.isAsync = this.backend === 'supabase';
  }

  /**
   * Acquire lock for ingestion (local SQLite only)
   */
  acquireLock() {
    if (this.isAsync) return; // Supabase handles concurrency
    
    if (fs.existsSync(this.lockPath)) {
      const stats = fs.statSync(this.lockPath);
      const age = Date.now() - stats.mtimeMs;
      if (age < LOCK_TIMEOUT) {
        throw new Error('Another ingestion is in progress');
      }
      fs.unlinkSync(this.lockPath);
    }
    fs.writeFileSync(this.lockPath, process.pid.toString());
  }

  /**
   * Release lock
   */
  releaseLock() {
    if (this.isAsync) return;
    
    if (fs.existsSync(this.lockPath)) {
      fs.unlinkSync(this.lockPath);
    }
  }

  /**
   * Ingest content into the knowledge base
   */
  async ingest(input, options = {}) {
    this.acquireLock();
    
    try {
      // Extract content
      console.log(`Extracting content from: ${input.slice(0, 100)}...`);
      const extracted = await extract(input, options.sourceType);
      
      // Check for duplicates
      const existing = await Promise.resolve(this.db.sourceExists(extracted.url, extracted.content));
      if (existing) {
        console.log(`Duplicate detected (source ID: ${existing.id}), skipping`);
        return { status: 'duplicate', sourceId: existing.id };
      }
      
      // Create chunks
      console.log(`Chunking content (${extracted.content.length} chars)...`);
      const chunks = createChunks(extracted.content, this.chunkOptions);
      console.log(`Created ${chunks.length} chunks`);
      
      // Generate embeddings
      console.log('Generating embeddings...');
      const embedResults = await this.embedder.embedBatch(chunks.map(c => c.content));
      
      // Merge chunks with embeddings
      const chunksWithEmbeddings = chunks.map((chunk, i) => ({
        ...chunk,
        embedding: embedResults[i].embedding,
        provider: embedResults[i].provider,
        model: embedResults[i].model
      }));
      
      // Store source
      const sourceId = await Promise.resolve(this.db.insertSource({
        url: extracted.url,
        title: extracted.title,
        sourceType: extracted.sourceType,
        summary: extracted.excerpt,
        rawContent: extracted.content,
        tags: options.tags || [],
        metadata: options.metadata || {}
      }));
      
      // Store chunks
      await Promise.resolve(this.db.insertChunks(sourceId, chunksWithEmbeddings));
      
      console.log(`Successfully ingested: ${extracted.title} (ID: ${sourceId})`);
      
      return {
        status: 'success',
        sourceId,
        title: extracted.title,
        sourceType: extracted.sourceType,
        chunkCount: chunks.length,
        backend: this.backend
      };
      
    } finally {
      this.releaseLock();
    }
  }

  /**
   * Search the knowledge base
   */
  async search(query, options = {}) {
    const topK = options.topK || 10;
    const maxCharsPerResult = options.maxCharsPerResult || 2500;
    const dedupeBySource = options.dedupeBySource !== false;
    
    // Embed the query
    console.log('Embedding query...');
    const { embedding: queryEmbedding } = await this.embedder.embed(query);
    
    // Vector search (uses pgvector for Supabase, in-memory for SQLite)
    console.log(`Searching with ${this.backend} backend...`);
    let results;
    
    if (this.db.vectorSearch) {
      // Use native vector search
      results = await this.db.vectorSearch(queryEmbedding, topK * 3); // Get more for deduplication
    } else {
      // Fallback: get all chunks and compute similarity
      const chunks = await Promise.resolve(this.db.getAllChunksWithEmbeddings());
      console.log(`Searching ${chunks.length} chunks...`);
      
      results = chunks.map(chunk => ({
        ...chunk,
        similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
      }));
      
      results.sort((a, b) => b.similarity - a.similarity);
    }
    
    // Deduplicate by source if requested
    let filtered = results;
    if (dedupeBySource) {
      const seenSources = new Set();
      filtered = results.filter(r => {
        if (seenSources.has(r.source_id)) return false;
        seenSources.add(r.source_id);
        return true;
      });
    }
    
    // Take top K and sanitize
    const topResults = filtered.slice(0, topK).map(r => ({
      sourceId: r.source_id,
      title: r.title,
      url: r.url,
      sourceType: r.source_type,
      content: r.content.slice(0, maxCharsPerResult),
      similarity: r.similarity,
      chunkIndex: r.chunk_index
    }));
    
    return topResults;
  }

  /**
   * Query with LLM-ready prompt
   * Returns search results + prompt for LLM
   */
  async query(question, options = {}) {
    const results = await this.search(question, options);
    
    if (results.length === 0) {
      return {
        answer: null,
        results: [],
        prompt: null
      };
    }
    
    // Build context from results
    const context = results.map((r, i) => 
      `[Source ${i + 1}: ${r.title}${r.url ? ` (${r.url})` : ''}]\n${r.content}`
    ).join('\n\n---\n\n');
    
    // Build prompt for LLM
    const prompt = `Answer the following question using ONLY the provided context. Cite which sources you used (e.g., "According to Source 1...").

If the context doesn't contain enough information to answer, say so.

Context:
${context}

Question: ${question}

Answer:`;

    return {
      results,
      prompt,
      context,
      backend: this.backend
    };
  }

  /**
   * List all sources
   */
  async list(options = {}) {
    return Promise.resolve(this.db.listSources(options));
  }

  /**
   * Get a specific source
   */
  async getSource(id) {
    return Promise.resolve(this.db.getSource(id));
  }

  /**
   * Delete a source
   */
  async delete(id) {
    return Promise.resolve(this.db.deleteSource(id));
  }

  /**
   * Get stats
   */
  async stats() {
    return Promise.resolve(this.db.getStats());
  }

  /**
   * Close database connection
   */
  close() {
    this.db.close();
  }
}

export default RAG;
