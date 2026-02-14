/**
 * SQLite database for RAG knowledge base
 * Stores sources and their embedded chunks
 */

import Database from 'better-sqlite3';
import { createHash } from 'crypto';
import path from 'path';
import fs from 'fs';

export class RagDatabase {
  constructor(dbPath = './rag.db') {
    // Ensure directory exists
    const dir = path.dirname(dbPath);
    if (dir && !fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    
    this.db = new Database(dbPath);
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('foreign_keys = ON');
    this.init();
  }

  init() {
    // Sources table - stores original content
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

    // Chunks table - stores embedded pieces
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

    // Create indexes
    this.db.exec(`
      CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source_id);
      CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(source_type);
      CREATE INDEX IF NOT EXISTS idx_sources_hash ON sources(content_hash);
      CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(url_normalized);
    `);
  }

  /**
   * Normalize URL for deduplication
   */
  normalizeUrl(url) {
    if (!url) return null;
    try {
      const parsed = new URL(url);
      // Remove tracking params
      const trackingParams = ['utm_source', 'utm_medium', 'utm_campaign', 'utm_content', 
        'fbclid', 'igshid', 'ref', 's', 't', 'si', 'feature'];
      trackingParams.forEach(p => parsed.searchParams.delete(p));
      // Normalize twitter -> x
      if (parsed.hostname === 'twitter.com') parsed.hostname = 'x.com';
      // Remove www
      parsed.hostname = parsed.hostname.replace(/^www\./, '');
      // Remove trailing slash and fragment
      let normalized = parsed.origin + parsed.pathname.replace(/\/$/, '') + parsed.search;
      return normalized.toLowerCase();
    } catch {
      return url.toLowerCase();
    }
  }

  /**
   * Hash content for deduplication
   */
  hashContent(content) {
    return createHash('sha256').update(content).digest('hex');
  }

  /**
   * Check if source already exists by URL or content hash
   */
  sourceExists(url, content) {
    const normalizedUrl = this.normalizeUrl(url);
    const contentHash = this.hashContent(content);
    
    const byUrl = normalizedUrl ? 
      this.db.prepare('SELECT id FROM sources WHERE url_normalized = ?').get(normalizedUrl) : null;
    const byHash = this.db.prepare('SELECT id FROM sources WHERE content_hash = ?').get(contentHash);
    
    return byUrl || byHash;
  }

  /**
   * Insert a new source
   */
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

  /**
   * Insert chunks for a source
   */
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

  /**
   * Get all chunks with embeddings for similarity search
   */
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
   * Get source by ID
   */
  getSource(id) {
    return this.db.prepare('SELECT * FROM sources WHERE id = ?').get(id);
  }

  /**
   * List all sources
   */
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

  /**
   * Delete a source and its chunks
   */
  deleteSource(id) {
    return this.db.prepare('DELETE FROM sources WHERE id = ?').run(id);
  }

  /**
   * Get stats
   */
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
      byType: Object.fromEntries(byType.map(r => [r.source_type, r.count]))
    };
  }

  close() {
    this.db.close();
  }
}

export default RagDatabase;
