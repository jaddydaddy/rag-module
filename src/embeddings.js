/**
 * Embedding providers for RAG
 * Supports Gemini (free) and OpenAI (fallback)
 */

const GEMINI_EMBEDDING_MODEL = 'text-embedding-004';
const OPENAI_EMBEDDING_MODEL = 'text-embedding-3-small';

const MAX_RETRIES = 3;
const RETRY_DELAYS = [1000, 2000, 4000];
const BATCH_SIZE = 10;
const BATCH_DELAY = 200;
const MAX_INPUT_CHARS = 8000;

// Simple LRU cache
class EmbeddingCache {
  constructor(maxSize = 1000) {
    this.cache = new Map();
    this.maxSize = maxSize;
  }

  get(key) {
    if (this.cache.has(key)) {
      const value = this.cache.get(key);
      // Move to end (most recently used)
      this.cache.delete(key);
      this.cache.set(key, value);
      return value;
    }
    return null;
  }

  set(key, value) {
    if (this.cache.size >= this.maxSize) {
      // Delete oldest entry
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  hash(text, provider) {
    // Simple hash for cache key
    let hash = 0;
    const str = `${provider}:${text}`;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }
    return hash.toString();
  }
}

const cache = new EmbeddingCache();

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function retryWithBackoff(fn, retries = MAX_RETRIES) {
  for (let i = 0; i < retries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (i === retries - 1) throw error;
      console.warn(`Retry ${i + 1}/${retries} after error:`, error.message);
      await sleep(RETRY_DELAYS[i] || RETRY_DELAYS[RETRY_DELAYS.length - 1]);
    }
  }
}

/**
 * Generate embedding using Gemini
 */
async function embedWithGemini(text, apiKey) {
  const truncated = text.slice(0, MAX_INPUT_CHARS);
  
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_EMBEDDING_MODEL}:embedContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: `models/${GEMINI_EMBEDDING_MODEL}`,
        content: { parts: [{ text: truncated }] }
      })
    }
  );

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`Gemini API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.embedding?.values || [];
}

/**
 * Generate embedding using OpenAI
 */
async function embedWithOpenAI(text, apiKey) {
  const truncated = text.slice(0, MAX_INPUT_CHARS);
  
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${apiKey}`
    },
    body: JSON.stringify({
      model: OPENAI_EMBEDDING_MODEL,
      input: truncated
    })
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`OpenAI API error: ${response.status} - ${error}`);
  }

  const data = await response.json();
  return data.data?.[0]?.embedding || [];
}

/**
 * Main embedding class with provider fallback
 */
export class EmbeddingProvider {
  constructor(config = {}) {
    this.geminiKey = config.geminiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    this.openaiKey = config.openaiKey || process.env.OPENAI_API_KEY;
    this.preferredProvider = config.preferredProvider || 'gemini';
  }

  /**
   * Embed a single text
   */
  async embed(text) {
    const cacheKey = cache.hash(text, this.preferredProvider);
    const cached = cache.get(cacheKey);
    if (cached) return cached;

    let embedding;
    let provider;
    let model;

    // Try preferred provider first
    if (this.preferredProvider === 'gemini' && this.geminiKey) {
      try {
        embedding = await retryWithBackoff(() => embedWithGemini(text, this.geminiKey));
        provider = 'gemini';
        model = GEMINI_EMBEDDING_MODEL;
      } catch (error) {
        console.warn('Gemini embedding failed, trying OpenAI:', error.message);
      }
    }

    // Fallback to OpenAI
    if (!embedding && this.openaiKey) {
      embedding = await retryWithBackoff(() => embedWithOpenAI(text, this.openaiKey));
      provider = 'openai';
      model = OPENAI_EMBEDDING_MODEL;
    }

    // Try Gemini as last resort if OpenAI was preferred
    if (!embedding && this.preferredProvider === 'openai' && this.geminiKey) {
      embedding = await retryWithBackoff(() => embedWithGemini(text, this.geminiKey));
      provider = 'gemini';
      model = GEMINI_EMBEDDING_MODEL;
    }

    if (!embedding || embedding.length === 0) {
      throw new Error('No embedding provider available or all providers failed');
    }

    const result = { embedding, provider, model };
    cache.set(cacheKey, result);
    return result;
  }

  /**
   * Embed multiple texts in batches
   */
  async embedBatch(texts) {
    const results = [];
    
    for (let i = 0; i < texts.length; i += BATCH_SIZE) {
      const batch = texts.slice(i, i + BATCH_SIZE);
      const batchResults = await Promise.all(batch.map(text => this.embed(text)));
      results.push(...batchResults);
      
      // Delay between batches
      if (i + BATCH_SIZE < texts.length) {
        await sleep(BATCH_DELAY);
      }
    }
    
    return results;
  }
}

/**
 * Cosine similarity between two vectors
 */
export function cosineSimilarity(a, b) {
  if (a.length !== b.length) return 0;
  
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

export default EmbeddingProvider;
