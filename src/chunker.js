/**
 * Text chunking for RAG
 * Splits content into overlapping chunks for embedding
 */

const DEFAULT_CHUNK_SIZE = 800;
const DEFAULT_OVERLAP = 200;
const MIN_CHUNK_SIZE = 100;

/**
 * Split text into sentences
 */
function splitSentences(text) {
  // Split on sentence boundaries
  return text.split(/(?<=[.!?])\s+/).filter(s => s.trim());
}

/**
 * Chunk text with overlap
 */
export function chunkText(text, options = {}) {
  const chunkSize = options.chunkSize || DEFAULT_CHUNK_SIZE;
  const overlap = options.overlap || DEFAULT_OVERLAP;
  const minChunkSize = options.minChunkSize || MIN_CHUNK_SIZE;
  
  // Clean text
  const cleaned = text
    .replace(/\s+/g, ' ')
    .trim();
  
  if (cleaned.length <= chunkSize) {
    return [cleaned];
  }
  
  const sentences = splitSentences(cleaned);
  const chunks = [];
  let currentChunk = '';
  let currentSentences = [];
  
  for (const sentence of sentences) {
    // If adding this sentence would exceed chunk size
    if (currentChunk.length + sentence.length + 1 > chunkSize) {
      if (currentChunk.length >= minChunkSize) {
        chunks.push(currentChunk.trim());
      }
      
      // Calculate overlap - take sentences from end of current chunk
      let overlapText = '';
      let overlapSentences = [];
      for (let i = currentSentences.length - 1; i >= 0; i--) {
        const testOverlap = currentSentences[i] + ' ' + overlapText;
        if (testOverlap.length <= overlap) {
          overlapText = testOverlap;
          overlapSentences.unshift(currentSentences[i]);
        } else {
          break;
        }
      }
      
      // Start new chunk with overlap
      currentChunk = overlapText + sentence;
      currentSentences = [...overlapSentences, sentence];
    } else {
      currentChunk += (currentChunk ? ' ' : '') + sentence;
      currentSentences.push(sentence);
    }
  }
  
  // Add final chunk
  if (currentChunk.length >= minChunkSize) {
    chunks.push(currentChunk.trim());
  } else if (chunks.length > 0 && currentChunk.length > 0) {
    // Append small remainder to last chunk
    chunks[chunks.length - 1] += ' ' + currentChunk.trim();
  } else if (currentChunk.length > 0) {
    chunks.push(currentChunk.trim());
  }
  
  return chunks;
}

/**
 * Create chunks with metadata
 */
export function createChunks(text, options = {}) {
  const rawChunks = chunkText(text, options);
  
  return rawChunks.map((content, index) => ({
    content,
    index,
    charStart: text.indexOf(content.slice(0, 50)),
    charLength: content.length
  }));
}

export default { chunkText, createChunks };
