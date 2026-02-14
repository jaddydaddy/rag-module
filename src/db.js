/**
 * Database abstraction for RAG knowledge base
 * Supports SQLite (local) and Supabase (cloud with pgvector)
 * 
 * Backend selection:
 * - If SUPABASE_URL and SUPABASE_KEY are set → Supabase
 * - Otherwise → SQLite (default)
 */

import Database from 'better-sqlite3';
import { createClient } from '@supabase/supabase-js';
import { createHash } from 'crypto';
import path from 'path';
import fs from 'fs';

/**
 * Detect which backend to use
 */
export function detectBackend() {
  const supabaseUrl = process.env.SUPABASE_URL;
  const supabaseKey = process.env.SUPABASE_KEY || process.env.SUPABASE_ANON_KEY;
  
  if (supabaseUrl && supabaseKey) {
    return 'supabase';
  }
  return 'sqlite';
}

/**
 * SQLite implementation (original)
 */
class SQLiteDatabase {
  constructor(dbPath = './rag.db') {
    const dir = path.dirname(dbPath);
    if (dir && !fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.backend = 'sqlite';
    this.init();
  }

  init() {
    this.db.exec(`
      CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        url TEXT,
        url_normalized TEXT UNIQUE,
        title TEXT,
        source_type TEXT NOT NULL,
        summary TEXT,
        raw_content TEXT,
        content_hash TEXT UNIQUE,
        tags TEXT DEFAULT '[]',
        metadata TEXT DEFAULT '{}',
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )
    `);

    this.db.exec(`
      CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        content TEXT NOT NULL,
        embedding BLOB,
        embedding_dim INTEGER,
        embedding_provider TEXT,
        embedding_model TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
      )
    `);

    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
      CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
      CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);
      CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url_normalized);
    `);
  }

  normalizeUrl(url) {
    if (!url) return null;
    try {
      const parsed = new URL(url);
      const trackingParams = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 
        'fbclid', 'igshid', 'ref', 's', 't', 'si', 'feature'];
      trackingParams.forEach(p => parsed.searchParams.delete(p));
      if (parsed.hostname === 'twitter.com') parsed.hostname = 'x.com';
      parsed.hostname = parsed.hostname.replace(/^www\./, '');
      let normalized = parsed.origin + parsed.pathname.replace(/\/$/, '') + parsed.search;
      return normalized.toLowerCase();
    } catch {
      return url.toLowerCase();
    }
  }

  hashContent(content) {
    return createHash('sha256').update(content).digest('hex');
  }

  sourceExists(url, content) {
    const normalizedUrl = this.normalizeUrl(url);
    const contentHash = this.hashContent(content);
    
    const byUrl = normalizedUrl ? 
      this.db.prepare('SELECT id FROM sources WHERE url_normalized = ?').get(normalizedUrl) : null;
    const byHash = this.db.prepare('SELECT id FROM sources WHERE content_hash = ?').get(contentHash);
    
    return byUrl || byHash;
  }

  insertSource({ url, title, sourceType, summary, rawContent, tags = [], metadata = {} }) {
    const stmt = this.db.prepare(`
      INSERT INTO sources (url, url_normalized, title, source_type, summary, raw_content, content_hash, tags, metadata)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    `);
    
    const result = stmt.run(
      url,
      this.normalizeUrl(url),
      title,
      sourceType,
      summary,
      rawContent,
      this.hashContent(rawContent),
      JSON.stringify(tags),
      JSON.stringify(metadata)
    );
    
    return result.lastInsertRowid;
  }

  insertChunks(sourceId, chunks) {
    const stmt = this.db.prepare(`
      INSERT INTO chunks (source_id, chunk_index, content, embedding, embedding_dim, embedding_provider, embedding_model)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);

    const insertMany = this.db.transaction((chunks) => {
      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        stmt.run(
          sourceId,
          i,
          chunk.content,
          chunk.embedding ? Buffer.from(new Float32Array(chunk.embedding).buffer) : null,
          chunk.embedding?.length || null,
          chunk.provider || null,
          chunk.model || null
        );
      }
    });

    insertMany(chunks);
  }

  getAllChunksWithEmbeddings() {
    const rows = this.db.prepare(`
      SELECT c.id, c.source_id, c.chunk_index, c.content, c.embedding, c.embedding_dim,
             s.title, s.url, s.source_type
      FROM chunks c
      JOIN sources s ON c.source_id = s.id
      WHERE c.embedding IS NOT NULL
    `).all();

    return rows.map(row => ({
      ...row,
      embedding: row.embedding ? Array.from(new Float32Array(row.embedding.buffer)) : null
    }));
  }

  /**
   * Vector similarity search (SQLite - in-memory calculation)
   * Returns chunks sorted by similarity
   */
  async vectorSearch(queryEmbedding, topK = 10) {
    // SQLite doesn't have native vector search, so we fetch all and compute in JS
    const chunks = this.getAllChunksWithEmbeddings();
    
    const results = chunks.map(chunk => ({
      ...chunk,
      similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
    }));
    
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  getSource(id) {
    return this.db.prepare('SELECT * FROM sources WHERE id = ?').get(id);
  }

  listSources({ limit = 50, offset = 0, sourceType = null } = {}) {
    let query = 'SELECT id, url, title, source_type, created_at FROM sources';
    const params = [];
    
    if (sourceType) {
      query += ' WHERE source_type = ?';
      params.push(sourceType);
    }
    
    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?';
    params.push(limit, offset);
    
    return this.db.prepare(query).all(...params);
  }

  deleteSource(id) {
    return this.db.prepare('DELETE FROM sources WHERE id = ?').run(id);
  }

  getStats() {
    const sources = this.db.prepare('SELECT COUNT(*) as count FROM sources').get();
    const chunks = this.db.prepare('SELECT COUNT(*) as count FROM chunks').get();
    const byType = this.db.prepare(`
      SELECT source_type, COUNT(*) as count 
      FROM sources 
      GROUP BY source_type
    `).all();
    
    return {
      totalSources: sources.count,
      totalChunks: chunks.count,
      byType: Object.fromEntries(byType.map(r => [r.source_type, r.count])),
      backend: 'sqlite'
    };
  }

  close() {
    this.db.close();
  }
}

/**
 * Supabase implementation (cloud with pgvector)
 */
class SupabaseDatabase {
  constructor() {
    const supabaseUrl = process.env.SUPABASE_URL;
    const supabaseKey = process.env.SUPABASE_KEY || process.env.SUPABASE_ANON_KEY;
    
    if (!supabaseUrl || !supabaseKey) {
      throw new Error('SUPABASE_URL and SUPABASE_KEY required for Supabase backend');
    }
    
    this.client = createClient(supabaseUrl, supabaseKey);
    this.backend = 'supabase';
  }

  normalizeUrl(url) {
    if (!url) return null;
    try {
      const parsed = new URL(url);
      const trackingParams = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 
        'fbclid', 'igshid', 'ref', 's', 't', 'si', 'feature'];
      trackingParams.forEach(p => parsed.searchParams.delete(p));
      if (parsed.hostname === 'twitter.com') parsed.hostname = 'x.com';
      parsed.hostname = parsed.hostname.replace(/^www\./, '');
      let normalized = parsed.origin + parsed.pathname.replace(/\/$/, '') + parsed.search;
      return normalized.toLowerCase();
    } catch {
      return url.toLowerCase();
    }
  }

  hashContent(content) {
    return createHash('sha256').update(content).digest('hex');
  }

  async sourceExists(url, content) {
    const normalizedUrl = this.normalizeUrl(url);
    const contentHash = this.hashContent(content);
    
    // Check by URL
    if (normalizedUrl) {
      const { data: byUrl } = await this.client
        .from('rag_sources')
        .select('id')
        .eq('url_normalized', normalizedUrl)
        .limit(1)
        .single();
      if (byUrl) return byUrl;
    }
    
    // Check by content hash
    const { data: byHash } = await this.client
      .from('rag_sources')
      .select('id')
      .eq('content_hash', contentHash)
      .limit(1)
      .single();
    
    return byHash;
  }

  async insertSource({ url, title, sourceType, summary, rawContent, tags = [], metadata = {} }) {
    const { data, error } = await this.client
      .from('rag_sources')
      .insert({
        url,
        url_normalized: this.normalizeUrl(url),
        title,
        source_type: sourceType,
        summary,
        raw_content: rawContent,
        content_hash: this.hashContent(rawContent),
        tags,
        metadata
      })
      .select('id')
      .single();
    
    if (error) throw new Error(`Supabase insert error: ${error.message}`);
    return data.id;
  }

  async insertChunks(sourceId, chunks) {
    const rows = chunks.map((chunk, i) => ({
      source_id: sourceId,
      chunk_index: i,
      content: chunk.content,
      embedding: chunk.embedding ? `[${chunk.embedding.join(',')}]` : null,
      embedding_provider: chunk.provider || null,
      embedding_model: chunk.model || null
    }));
    
    const { error } = await this.client
      .from('rag_chunks')
      .insert(rows);
    
    if (error) throw new Error(`Supabase chunk insert error: ${error.message}`);
  }

  async getAllChunksWithEmbeddings() {
    const { data, error } = await this.client
      .from('rag_chunks')
      .select(`
        id,
        source_id,
        chunk_index,
        content,
        embedding,
        rag_sources!inner (
          title,
          url,
          source_type
        )
      `)
      .not('embedding', 'is', null);
    
    if (error) throw new Error(`Supabase query error: ${error.message}`);
    
    return data.map(row => ({
      id: row.id,
      source_id: row.source_id,
      chunk_index: row.chunk_index,
      content: row.content,
      embedding: row.embedding,
      title: row.rag_sources.title,
      url: row.rag_sources.url,
      source_type: row.rag_sources.source_type
    }));
  }

  /**
   * Vector similarity search using pgvector
   * Uses Supabase RPC function for efficient vector search
   */
  async vectorSearch(queryEmbedding, topK = 10) {
    // Use RPC function for vector similarity search
    const { data, error } = await this.client.rpc('match_rag_chunks', {
      query_embedding: queryEmbedding,
      match_count: topK
    });
    
    if (error) {
      // Fallback to manual search if RPC not available
      console.warn('Vector search RPC not available, using fallback:', error.message);
      return this.vectorSearchFallback(queryEmbedding, topK);
    }
    
    return data.map(row => ({
      id: row.id,
      source_id: row.source_id,
      chunk_index: row.chunk_index,
      content: row.content,
      title: row.title,
      url: row.url,
      source_type: row.source_type,
      similarity: row.similarity
    }));
  }

  /**
   * Fallback vector search (fetches all chunks, computes similarity in JS)
   */
  async vectorSearchFallback(queryEmbedding, topK = 10) {
    const chunks = await this.getAllChunksWithEmbeddings();
    
    const results = chunks.map(chunk => ({
      ...chunk,
      similarity: cosineSimilarity(queryEmbedding, chunk.embedding)
    }));
    
    results.sort((a, b) => b.similarity - a.similarity);
    return results.slice(0, topK);
  }

  async getSource(id) {
    const { data, error } = await this.client
      .from('rag_sources')
      .select('*')
      .eq('id', id)
      .single();
    
    if (error) return null;
    return data;
  }

  async listSources({ limit = 50, offset = 0, sourceType = null } = {}) {
    let query = this.client
      .from('rag_sources')
      .select('id, url, title, source_type, created_at')
      .order('created_at', { ascending: false })
      .range(offset, offset + limit - 1);
    
    if (sourceType) {
      query = query.eq('source_type', sourceType);
    }
    
    const { data, error } = await query;
    if (error) throw new Error(`Supabase list error: ${error.message}`);
    return data;
  }

  async deleteSource(id) {
    const { data, error } = await this.client
      .from('rag_sources')
      .delete()
      .eq('id', id);
    
    if (error) throw new Error(`Supabase delete error: ${error.message}`);
    return { changes: 1 };
  }

  async getStats() {
    const { count: sourceCount } = await this.client
      .from('rag_sources')
      .select('*', { count: 'exact', head: true });
    
    const { count: chunkCount } = await this.client
      .from('rag_chunks')
      .select('*', { count: 'exact', head: true });
    
    const { data: byType } = await this.client
      .from('rag_sources')
      .select('source_type')
      .then(({ data }) => {
        const counts = {};
        data?.forEach(row => {
          counts[row.source_type] = (counts[row.source_type] || 0) + 1;
        });
        return { data: counts };
      });
    
    return {
      totalSources: sourceCount || 0,
      totalChunks: chunkCount || 0,
      byType: byType || {},
      backend: 'supabase'
    };
  }

  close() {
    // Supabase client doesn't need explicit close
  }
}

/**
 * Cosine similarity between two vectors
 */
function cosineSimilarity(a, b) {
  if (!a || !b || a.length !== b.length) return 0;
  
  let dotProduct = 0;
  let normA = 0;
  let normB = 0;
  
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  
  const denominator = Math.sqrt(normA) * Math.sqrt(normB);
  return denominator === 0 ? 0 : dotProduct / denominator;
}

/**
 * Factory function - returns appropriate database implementation
 */
export function createDatabase(config = {}) {
  const backend = config.backend || detectBackend();
  
  if (backend === 'supabase') {
    console.log('Using Supabase backend (pgvector)');
    return new SupabaseDatabase();
  } else {
    console.log(`Using SQLite database: ${config.dbPath || './rag.db'}`);
    return new SQLiteDatabase(config.dbPath);
  }
}

// Export classes for direct use
export { SQLiteDatabase, SupabaseDatabase, cosineSimilarity };

// Default export for backwards compatibility
export class RagDatabase extends SQLiteDatabase {}
export default RagDatabase;
