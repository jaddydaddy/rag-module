/**
 * Main RAG (Retrieval Augmented Generation) class
 * Ties together extraction, chunking, embedding, and retrieval
 * Supports both SQLite (local) and Supabase (cloud) backends
 */

import { createDatabase, getDatabaseType } from './db-factory.js';
import { EmbeddingProvider, cosineSimilarity } from './embeddings.js';
import { extract, detectSourceType } from './extractors.js';
import { createChunks } from './chunker.js';
import fs from 'fs';
import path from 'path';

const LOCK_FILE = '.rag.lock';
const LOCK_TIMEOUT = 15 * 60 * 1000; // 15 minutes

export class RAG {
  constructor(config = {}) {
    this.config = config;
    this.dbPath = config.dbPath || './rag.db';
    this.db = createDatabase(config);
    this.embedder = new EmbeddingProvider(config);
    this.chunkOptions = {
      chunkSize: config.chunkSize || 800,
      overlap: config.overlap || 200
    };
    this.lockPath = path.join(path.dirname(this.dbPath), LOCK_FILE);
    this.isSupabase = getDatabaseType() === 'supabase';
  }

  /**
   * Helper to handle both sync (SQLite) and async (Supabase) methods
   */
  async dbCall(method, ...args) {
    const result = this.db[method](...args);
    return result instanceof Promise ? await result : result;
  }

  /**
   * Acquire lock for ingestion (SQLite only)
   */
  acquireLock() {
    if (this.isSupabase) return; // Supabase handles concurrency
    
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
    if (this.isSupabase) return;
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
      const existing = await this.dbCall('sourceExists', extracted.url, extracted.content);
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
      const sourceId = await this.dbCall('insertSource', {
        url: extracted.url,
        title: extracted.title,
        sourceType: extracted.sourceType,
        summary: extracted.excerpt,
        rawContent: extracted.content,
        tags: options.tags || [],
        metadata: options.metadata || {}
      });
      
      // Store chunks
      await this.dbCall('insertChunks', sourceId, chunksWithEmbeddings);
      
      console.log(`Successfully ingested: ${extracted.title} (ID: ${sourceId})`);
      
      return {
        status: 'success',
        sourceId,
        title: extracted.title,
        sourceType: extracted.sourceType,
        chunkCount: chunks.length
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
    
    // Try server-side search for Supabase
    if (this.isSupabase && this.db.searchSimilar) {
      const serverResults = await this.db.searchSimilar(queryEmbedding, topK);
      if (serverResults) {
        return serverResults.map(r => ({
          sourceId: r.source_id,
          title: r.title,
          url: r.url,
          sourceType: r.source_type,
          content: r.content?.slice(0, maxCharsPerResult),
          similarity: r.similarity,
          chunkIndex: r.chunk_index
        }));
      }
    }
    
    // Fall back to client-side search
    const chunks = await this.dbCall('getAllChunksWithEmbeddings');
    console.log(`Searching ${chunks.length} chunks...`);
    
    // Calculate similarities
    const results = chunks.map(chunk => ({
      ...chunk,
      similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
    }));
    
    // Sort by similarity
    results.sort((a, b) => b.similarity - a.similarity);
    
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
   * Query with LLM-generated answer
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
      context
    };
  }

  /**
   * List all sources
   */
  async list(options = {}) {
    return await this.dbCall('listSources', options);
  }

  /**
   * Get a specific source
   */
  async getSource(id) {
    return await this.dbCall('getSource', id);
  }

  /**
   * Delete a source
   */
  async delete(id) {
    return await this.dbCall('deleteSource', id);
  }

  /**
   * Get stats
   */
  async stats() {
    return await this.dbCall('getStats');
  }

  /**
   * Close database connection
   */
  close() {
    this.db.close();
  }
}

export default RAG;
