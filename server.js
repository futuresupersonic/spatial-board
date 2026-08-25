// Spatial board — backend
//
// Three jobs, all thin wrappers around OpenAI so the API key never touches
// the browser:
//   POST /chat        real answers for RESPOND() (streamed, replaces the
//                      canned CANNED[] array in the client)
//   POST /transcribe   Whisper transcription for the dictation mic
//   POST /session      ephemeral Realtime token for the live-voice orb
//
// Run:
//   cp .env.example .env   # fill in OPENAI_API_KEY
//   npm install
//   npm start
//   open http://localhost:8787/board.html

require('dotenv').config();
const path = require('path');
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const { toFile } = require('openai/uploads');

const PORT = process.env.PORT || 8787;
const CHAT_MODEL = process.env.CHAT_MODEL || 'gpt-4o';
const TRANSCRIBE_MODEL = process.env.TRANSCRIBE_MODEL || 'whisper-1';
// OpenAI's Realtime API went GA in August 2025 — the old preview alias
// ('gpt-4o-realtime-preview') is on its way out. 'gpt-realtime' is the
// current always-latest GA alias for the speech-to-speech model.
const REALTIME_MODEL = process.env.REALTIME_MODEL || 'gpt-realtime';

if (!process.env.OPENAI_API_KEY) {
  console.warn(
    '\n[board] WARNING: OPENAI_API_KEY is not set. /chat, /transcribe and ' +
    '/session will return 500s until you copy .env.example to .env and add a key.\n'
  );
}

// A placeholder key lets the server boot (and serve the static board) even
// with no .env yet; real calls to OpenAI will fail with a clear 500 until
// OPENAI_API_KEY is actually set.
const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY || 'sk-missing-configure-.env' });

const SYSTEM_PROMPT = process.env.SYSTEM_PROMPT || [
  'You are the model answering into a spatial notes board. Each exchange becomes',
  'one card the person can place and rearrange in space, so answers should stand',
  'on their own without leaning on "as I said above" — there is no "above."',
  '',
  'Be direct and substantive. Take a real position when asked for one instead of',
  'hedging into a list of considerations. Push back when something is wrong or',
  'underspecified rather than agreeing by default. Skip preamble and filler',
  '("Great question!", "I hope this helps") and go straight into the answer.',
  'Match length to the question — a yes/no gets a short answer, a hard design',
  'question earns a real one.',
  '',
  'If the message includes fetched web page content below, treat it as reference',
  'material the person wants you to use — read, summarize, critique, or connect',
  'it to what they asked, as a real reader of that page would.'
].join('\n');

// ------------------------------------------------------------ link reading --
// Very small, dependency-free "read this page" helper: fetch a URL the user
// pasted into a question, strip it down to plain text, and hand that to the
// model as extra context. Not a full readability parser — good enough for
// blog posts / articles (e.g. a Substack piece) without adding a dependency.
const URL_RE = /https?:\/\/[^\s<>"')\]]+/gi;
function decodeEntities(s) {
  return s
    .replace(/&nbsp;/gi, ' ').replace(/&amp;/gi, '&').replace(/&lt;/gi, '<')
    .replace(/&gt;/gi, '>').replace(/&quot;/gi, '"').replace(/&#0?39;/gi, "'")
    .replace(/&mdash;/gi, '—').replace(/&ndash;/gi, '–').replace(/&rsquo;/gi, '’')
    .replace(/&lsquo;/gi, '‘').replace(/&rdquo;/gi, '”').replace(/&ldquo;/gi, '“');
}
async function fetchPageText(url) {
  try {
    const r = await fetch(url, {
      headers: { 'User-Agent': 'Mozilla/5.0 (compatible; SpatialBoard/1.0; +https://spatial-board.onrender.com)' },
      redirect: 'follow'
    });
    if (!r.ok) return { url, error: 'HTTP ' + r.status };
    const ct = r.headers.get('content-type') || '';
    if (ct && ct.indexOf('html') === -1 && ct.indexOf('text') === -1) {
      return { url, error: 'not a readable page (' + ct + ')' };
    }
    let html = await r.text();
    if (html.length > 500000) html = html.slice(0, 500000);
    const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
    const title = titleMatch ? decodeEntities(titleMatch[1]).replace(/\s+/g, ' ').trim() : '';
    let body = html
      .replace(/<script[\s\S]*?<\/script>/gi, ' ')
      .replace(/<style[\s\S]*?<\/style>/gi, ' ')
      .replace(/<!--[\s\S]*?-->/g, ' ')
      .replace(/<(br|\/p|\/div|\/li|\/h[1-6])\s*\/?>/gi, '\n')
      .replace(/<[^>]+>/g, ' ');
    body = decodeEntities(body).replace(/[ \t]+/g, ' ').replace(/\n[ \t]*/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
    if (body.length > 8000) body = body.slice(0, 8000) + '\n…[truncated]';
    return { url, title, text: body };
  } catch (err) {
    return { url, error: err.message || String(err) };
  }
}

const app = express();
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => res.redirect('/board.html'));

// ---------------------------------------------------------------- /chat ----
// body: { question: string, history: [{ q: string, a: string }] }
// streams plain text chunks (the running "sofar" text is NOT re-sent —
// each chunk is a delta, matching how the client's onToken accumulates).
app.post('/chat', express.json({ limit: '2mb' }), async (req, res) => {
  const { question, history } = req.body || {};
  if (!question || typeof question !== 'string') {
    return res.status(400).json({ error: 'missing "question" string' });
  }
  const messages = [{ role: 'system', content: SYSTEM_PROMPT }];
  (Array.isArray(history) ? history : []).slice(-20).forEach((turn) => {
    if (turn && turn.q) messages.push({ role: 'user', content: String(turn.q) });
    if (turn && turn.a) messages.push({ role: 'assistant', content: String(turn.a) });
  });

  // If the question contains a link, fetch it and hand the page content to
  // the model as reference context — e.g. "what do you think of this
  // Substack post, and what should I write next?"
  const urls = Array.from(new Set(question.match(URL_RE) || [])).slice(0, 2);
  if (urls.length) {
    const pages = await Promise.all(urls.map(fetchPageText));
    const chunks = pages.map((p) => {
      if (p.error) return '[Could not read ' + p.url + ': ' + p.error + ']';
      return '[Page: ' + p.url + (p.title ? ' — "' + p.title + '"' : '') + ']\n' + p.text;
    });
    messages.push({ role: 'system', content: 'Fetched web page content:\n\n' + chunks.join('\n\n---\n\n') });
  }

  messages.push({ role: 'user', content: question });

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');

  try {
    const stream = await openai.chat.completions.create({
      model: CHAT_MODEL,
      messages,
      stream: true
    });
    for await (const part of stream) {
      const delta = part.choices?.[0]?.delta?.content;
      if (delta) res.write(delta);
    }
    res.end();
  } catch (err) {
    console.error('[board] /chat error:', err.message || err);
    // headers may already be sent if the stream started; if not, send a real status
    if (!res.headersSent) {
      res.status(500).json({ error: 'chat failed: ' + (err.message || String(err)) });
    } else {
      res.end('\n\n[error: ' + (err.message || String(err)) + ']');
    }
  }
});

// ------------------------------------------------------------ /transcribe --
// body: raw audio bytes, Content-Type identifies the codec (webm/mp4/ogg)
app.post('/transcribe', express.raw({ type: '*/*', limit: '25mb' }), async (req, res) => {
  if (!Buffer.isBuffer(req.body) || !req.body.length) {
    return res.status(400).send('empty body');
  }
  const mime = req.headers['content-type'] || 'audio/webm';
  const ext = mime.indexOf('mp4') >= 0 ? 'mp4' : mime.indexOf('ogg') >= 0 ? 'ogg' : 'webm';
  try {
    const file = await toFile(req.body, `audio.${ext}`, { type: mime });
    const result = await openai.audio.transcriptions.create({
      file,
      model: TRANSCRIBE_MODEL
    });
    res.type('text/plain').send(result.text || '');
  } catch (err) {
    console.error('[board] /transcribe error:', err.message || err);
    res.status(500).send('transcription failed: ' + (err.message || String(err)));
  }
});

// ---------------------------------------------------------------- /session --
// mints a short-lived client secret so the browser can talk to the Realtime
// API directly over WebRTC without ever seeing OPENAI_API_KEY.
//
// OpenAI's Realtime API went GA in August 2025 and the request/response shape
// changed from the original beta: the endpoint moved from
// /v1/realtime/sessions to /v1/realtime/client_secrets, and the model/voice
// now live nested under a "session" object instead of top-level fields. The
// response shape also shifted (some accounts get the token as a top-level
// "value", others nested under "client_secret.value") — this handles both so
// a future minor API tweak doesn't silently break voice again.
app.post('/session', express.json({ limit: '10kb' }), async (req, res) => {
  try {
    // A tester can set a preferred voice in Settings; that overrides the
    // server's default (REALTIME_VOICE env, or 'marin') for just their call.
    const requestedVoice = req.body && typeof req.body.voice === 'string' ? req.body.voice.trim() : '';
    const voice = requestedVoice || process.env.REALTIME_VOICE || 'marin';
    const r = await fetch('https://api.openai.com/v1/realtime/client_secrets', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer ' + process.env.OPENAI_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        session: {
          type: 'realtime',
          model: REALTIME_MODEL,
          audio: { output: { voice: voice } }
        }
      })
    });
    const raw = await r.text();
    let data;
    try { data = JSON.parse(raw); } catch (e) { throw new Error('non-JSON response (' + r.status + '): ' + raw.slice(0, 200)); }
    if (!r.ok) throw new Error(data?.error?.message || ('realtime session request failed (' + r.status + ')'));
    const token = data.value || (data.client_secret && data.client_secret.value);
    if (!token) throw new Error('no client secret in response: ' + raw.slice(0, 200));
    // normalize to the shape the client expects regardless of which shape
    // OpenAI actually returned
    res.json({ client_secret: { value: token }, model: REALTIME_MODEL, raw: data });
  } catch (err) {
    console.error('[board] /session error:', err.message || err);
    res.status(500).json({ error: 'session failed: ' + (err.message || String(err)) });
  }
});

app.listen(PORT, () => {
  console.log(`[board] listening on http://localhost:${PORT}/board.html`);
});
