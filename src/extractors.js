/**
 * Content extractors for different source types
 * Handles URLs, YouTube, PDFs, and plain text
 */

import { JSDOM } from 'jsdom';
import { Readability } from '@mozilla/readability';
import { YoutubeTranscript } from 'youtube-transcript';
import pdf from 'pdf-parse';
import fs from 'fs';
import path from 'path';

const MAX_CONTENT_LENGTH = 200000;
const MIN_CONTENT_LENGTH = 20;
const MIN_ARTICLE_LENGTH = 500;
const ERROR_SIGNALS = ['access denied', 'captcha', 'please enable javascript', 'cloudflare', '404', 'sign in', 'blocked', 'rate limit'];

/**
 * Detect source type from URL or file path
 */
export function detectSourceType(input) {
  const lower = input.toLowerCase();
  
  // YouTube
  if (lower.includes('youtube.com/watch') || lower.includes('youtu.be/')) {
    return 'video';
  }
  
  // Twitter/X
  if (lower.includes('twitter.com/') || lower.includes('x.com/')) {
    return 'tweet';
  }
  
  // PDF
  if (lower.endsWith('.pdf')) {
    return 'pdf';
  }
  
  // Local files
  if (lower.endsWith('.txt') || lower.endsWith('.md')) {
    return 'text';
  }
  
  // URLs default to article
  if (lower.startsWith('http://') || lower.startsWith('https://')) {
    return 'article';
  }
  
  // Plain text input
  return 'text';
}

/**
 * Extract YouTube video ID from URL
 */
function extractYouTubeId(url) {
  const patterns = [
    /(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([^&\n?#]+)/,
    /youtube\.com\/v\/([^&\n?#]+)/
  ];
  
  for (const pattern of patterns) {
    const match = url.match(pattern);
    if (match) return match[1];
  }
  return null;
}

/**
 * Validate extracted content quality
 */
function validateContent(content, sourceType) {
  if (!content || content.length < MIN_CONTENT_LENGTH) {
    throw new Error(`Content too short (${content?.length || 0} chars)`);
  }
  
  // Check for error pages
  const lowerContent = content.toLowerCase();
  const errorCount = ERROR_SIGNALS.filter(signal => lowerContent.includes(signal)).length;
  if (errorCount >= 2) {
    throw new Error('Content appears to be an error page');
  }
  
  // For articles, check for real prose
  if (sourceType === 'article' && content.length < MIN_ARTICLE_LENGTH) {
    throw new Error(`Article content too short (${content.length} chars)`);
  }
  
  // Check for prose vs navigation menus (for articles)
  if (sourceType === 'article') {
    const lines = content.split('\n').filter(l => l.trim());
    const longLines = lines.filter(l => l.length > 80);
    if (lines.length > 5 && longLines.length / lines.length < 0.15) {
      throw new Error('Content appears to be navigation/menus rather than prose');
    }
  }
  
  return true;
}

/**
 * Extract content from a web article
 */
async function extractArticle(url) {
  const response = await fetch(url, {
    headers: {
      'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
      'Accept': 'text/html,application/xhtml+xml'
    }
  });
  
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
  }
  
  const html = await response.text();
  const dom = new JSDOM(html, { url });
  const reader = new Readability(dom.window.document);
  const article = reader.parse();
  
  if (!article || !article.textContent) {
    throw new Error('Failed to extract article content');
  }
  
  return {
    title: article.title || '',
    content: article.textContent.slice(0, MAX_CONTENT_LENGTH),
    excerpt: article.excerpt || ''
  };
}

/**
 * Extract content from YouTube video transcript
 */
async function extractYouTube(url) {
  const videoId = extractYouTubeId(url);
  if (!videoId) {
    throw new Error('Could not extract YouTube video ID');
  }
  
  try {
    const transcript = await YoutubeTranscript.fetchTranscript(videoId);
    const content = transcript.map(t => t.text).join(' ');
    
    // Try to get video title
    let title = `YouTube Video ${videoId}`;
    try {
      const oembedUrl = `https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v=${videoId}&format=json`;
      const metaResponse = await fetch(oembedUrl);
      if (metaResponse.ok) {
        const meta = await metaResponse.json();
        title = meta.title || title;
      }
    } catch { /* ignore */ }
    
    return {
      title,
      content: content.slice(0, MAX_CONTENT_LENGTH),
      excerpt: content.slice(0, 200)
    };
  } catch (error) {
    throw new Error(`YouTube transcript extraction failed: ${error.message}`);
  }
}

/**
 * Extract content from Twitter/X
 * Uses FxTwitter API (no auth needed)
 */
async function extractTweet(url) {
  // Extract tweet ID
  const match = url.match(/(?:twitter\.com|x\.com)\/\w+\/status\/(\d+)/);
  if (!match) {
    throw new Error('Could not extract tweet ID');
  }
  
  const tweetId = match[1];
  const username = url.match(/(?:twitter\.com|x\.com)\/(\w+)\/status/)?.[1] || 'user';
  
  // Try FxTwitter API
  try {
    const response = await fetch(`https://api.fxtwitter.com/${username}/status/${tweetId}`);
    if (response.ok) {
      const data = await response.json();
      const tweet = data.tweet;
      return {
        title: `Tweet by @${tweet.author?.screen_name || username}`,
        content: tweet.text || '',
        excerpt: tweet.text?.slice(0, 200) || ''
      };
    }
  } catch { /* try fallback */ }
  
  throw new Error('Could not extract tweet content');
}

/**
 * Extract content from PDF file
 */
async function extractPDF(filePath) {
  const absolutePath = path.resolve(filePath);
  if (!fs.existsSync(absolutePath)) {
    throw new Error(`PDF file not found: ${filePath}`);
  }
  
  const dataBuffer = fs.readFileSync(absolutePath);
  const data = await pdf(dataBuffer);
  
  return {
    title: path.basename(filePath, '.pdf'),
    content: data.text.slice(0, MAX_CONTENT_LENGTH),
    excerpt: data.text.slice(0, 200)
  };
}

/**
 * Handle plain text or local file
 */
async function extractText(input) {
  // Check if it's a file path
  if (fs.existsSync(input)) {
    const content = fs.readFileSync(input, 'utf-8');
    return {
      title: path.basename(input),
      content: content.slice(0, MAX_CONTENT_LENGTH),
      excerpt: content.slice(0, 200)
    };
  }
  
  // Plain text input
  return {
    title: input.slice(0, 50) + (input.length > 50 ? '...' : ''),
    content: input.slice(0, MAX_CONTENT_LENGTH),
    excerpt: input.slice(0, 200)
  };
}

/**
 * Main extraction function
 */
export async function extract(input, sourceType = null) {
  const type = sourceType || detectSourceType(input);
  
  let result;
  switch (type) {
    case 'article':
      result = await extractArticle(input);
      break;
    case 'video':
      result = await extractYouTube(input);
      break;
    case 'tweet':
      result = await extractTweet(input);
      break;
    case 'pdf':
      result = await extractPDF(input);
      break;
    case 'text':
    default:
      result = await extractText(input);
      break;
  }
  
  // Validate content
  validateContent(result.content, type);
  
  return {
    ...result,
    sourceType: type,
    url: type !== 'text' ? input : null
  };
}

export default { extract, detectSourceType };
