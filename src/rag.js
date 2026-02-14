/**
 * Main RAG (Retrieval Augmented Generation) class
 * Ties together extraction, chunking, embedding, and retrieval
 */

import { RagDatabase } from './db.js';
import { EmbeddingProvider, cosineSimilarity } from './embeddings.js';
import { extract, detectSourceType } from './extractors.js';
import { createChunks } from './chunker.js';
import fs from 'fs';
import path from 'path';

const LOCK_FILE = '.rag.lock';
const LOCK_TIMEOUT = 15 * 60 * 1000; // 15 minutes

export class RAG {
  constructor(config = {}) {
    this.dbPath = config.dbPath || './rag.db';
    this.db = new RagDatabase(this.dbPath);
    this.embedder = new EmbeddingProvider(config);
    this.chunkOptions = {
      chunkSize: config.chunkSize || 800,
      overlap: config.overlap || 200
    };
    this.lockPath = path.join(path.dirname(this.dbPath), LOCK_FILE);
  }

  /**
   * Acquire lock for ingestion
   */
  acquireLock() {
    if (fs.existsSync(this.lockPath)) {
      const stats = fs.statSync(this.lockPath);
      const age = Date.now() - stats.mtimeMs;
      if (age < LOCK_TIMEOUT) {
        throw new Error('Another ingestion is in progress');
      }
      // Stale lock, remove it
      fs.unlinkSync(this.lockPath);
    }
    fs.writeFileSync(this.lockPath, process.pid.toString());
  }

  /**
   * Release lock
   */
  releaseLock() {
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
      const existing = this.db.sourceExists(extracted.url, extracted.content);
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
      const sourceId = this.db.insertSource({
        url: extracted.url,
        title: extracted.title,
        sourceType: extracted.sourceType,
        summary: extracted.excerpt,
        rawContent: extracted.content,
        tags: options.tags || [],
        metadata: options.metadata || {}
      });
      
      // Store chunks
      this.db.insertChunks(sourceId, chunksWithEmbeddings);
      
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
    
    // Get all chunks with embeddings
    const chunks = this.db.getAllChunksWithEmbeddings();
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
  list(options = {}) {
    return this.db.listSources(options);
  }

  /**
   * Get a specific source
   */
  getSource(id) {
    return this.db.getSource(id);
  }

  /**
   * Delete a source
   */
  delete(id) {
    return this.db.deleteSource(id);
  }

  /**
   * Get stats
   */
  stats() {
    return this.db.getStats();
  }

  /**
   * Close database connection
   */
  close() {
    this.db.close();
  }
}

export default RAG;
