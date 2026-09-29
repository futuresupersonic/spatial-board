// Spatial board — backend
//
// Four jobs, all thin wrappers around OpenAI so the API key never touches
// the browser:
//   POST /chat        real answers for RESPOND() (streamed, replaces the
//                      canned CANNED[] array in the client)
//   POST /transcribe   Whisper transcription for the dictation mic
//   POST /session      ephemeral Realtime token for the live-voice orb
//   POST /tts          real speech for "Read out loud" (replaces the
//                      browser's robotic built-in speechSynthesis voice)
//
// Added: context-aware answers. POST /chat now also accepts { package } — the
// board's structured context (follow-up lineage, linked cards with their bridge
// words, quote) — and answers from that. The old { question, history } shape
// still works untouched. Also new: GET /health, POST /context/render (debug),
// and an optional BOARD_PASSWORD gate.
//
// Run:
//   cp .env.example .env   # fill in OPENAI_API_KEY
//   npm install
//   npm start
//   open http://localhost:8787/board.html

require('dotenv').config();
const path = require('path');
const crypto = require('crypto');
const express = require('express');
const cors = require('cors');
const OpenAI = require('openai');
const { toFile } = require('openai/uploads');

const PORT = process.env.PORT || 8787;
const CHAT_MODEL = process.env.CHAT_MODEL || 'gpt-4o';
const TRANSCRIBE_MODEL = process.env.TRANSCRIBE_MODEL || 'whisper-1';
// Same family of natural voices as the live-conversation orb, just used for
// one-shot "read this card aloud" playback instead of a two-way call.
const TTS_MODEL = process.env.TTS_MODEL || 'gpt-4o-mini-tts';
// OpenAI's Realtime API went GA in August 2025 — the old preview alias
// ('gpt-4o-realtime-preview') is on its way out. 'gpt-realtime' is the
// current always-latest GA alias for the speech-to-speech model.
const REALTIME_MODEL = process.env.REALTIME_MODEL || 'gpt-realtime';
const CONTEXT_DEBUG = process.env.CONTEXT_DEBUG !== '0';   // print each rendered prompt in the logs; set 0 to silence
const BOARD_PASSWORD = process.env.BOARD_PASSWORD || '';    // optional: if set, the whole site asks for it
const BUILD = 'context-v1';

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

// ===================== context package -> prompt =====================
// The board sends a structured "context package" (see buildContextPackage in
// board-v4.html): current turn, lineage, prior follow-ups, explicit links with
// their bridge words, recent history (fallback only), and a small topology.
// This section is the ONLY place the package becomes text for the model, so
// /chat and /context/render can never disagree about what was sent.

const CONTEXT_SYSTEM_PROMPT = [
  'You are answering inside a spatial knowledge board. The user builds structure on the board by hand,',
  'and that structure reaches you as a context package. Treat the package as a map of what matters for this question.',
  '',
  '- CURRENT TURN is what you must answer. If a QUOTED PASSAGE is present, the question is about that passage first;',
  '  use its source card only to understand it.',
  '- LINEAGE is the chain of earlier exchanges this question is a follow-up to, oldest first, ending with the parent',
  '  card the user chose to continue from. Treat it as the conversation so far for this thread.',
  '- EARLIER FOLLOW-UPS are questions already asked under the same parent, in the order they were asked.',
  '- LINKED CONTEXT are exchanges the user explicitly connected and marked as context. Each has a relation label the user',
  '  wrote (a "bridge word"). Use that label as the meaning of the connection. If a connection has no label, the user did not',
  '  say how the cards relate, so do not invent a relationship.',
  '- RECENT CONVERSATION appears only when the question is not anchored to a card. It is a fallback, not a topic.',
  '- BOARD STRUCTURE shows how the included pieces relate, and how many other cards exist but were left out.',
  '  Do not assume anything about cards that were left out.',
  '',
  'Rules: answer the current question directly. Do not summarize the package back to the user. Card labels such as "5-1" are',
  'identifiers the user sees on the board; you may use them when it helps clarity. If the package does not contain what you',
  'need, say so instead of guessing.'
].join('\n');

function clipS(s, n) {
  s = String(s == null ? '' : s);
  return s.length > n ? s.slice(0, n) + ' …[truncated]' : s;
}
function arr(a, max) { return Array.isArray(a) ? a.slice(0, max) : []; }

// Defensive shape check. The client is trusted (it's your own board) but a
// malformed or oversized body should fail with a clear message, not a crash.
function sanitizePackage(p) {
  if (!p || p.v !== 1) throw new Error('package.v must be 1');
  if (!p.current || typeof p.current.question !== 'string' || !p.current.question.trim())
    throw new Error('package.current.question is required');
  const card = (c) => ({
    label: clipS(c && c.label, 40), role: clipS(c && c.role, 40),
    q: clipS(c && c.q, 4000), a: clipS(c && c.a, 8000),
    hop: c && c.hop, via: c && c.via, detached: !!(c && c.detached), interrupted: !!(c && c.interrupted)
  });
  const q = p.current.quote;
  return {
    v: 1,
    current: {
      question: clipS(p.current.question, 8000),
      followUpFrom: p.current.followUpFrom ? clipS(p.current.followUpFrom, 40) : null,
      quote: q && q.text ? { text: clipS(q.text, 6000), fromCard: q.fromCard ? clipS(q.fromCard, 40) : null } : null
    },
    focus: p.focus || { kind: 'none' },
    lineage: arr(p.lineage, 24).map(card),
    priorFollowUps: arr(p.priorFollowUps, 24).map(card),
    links: arr(p.links, 24).map(card),
    recent: arr(p.recent, 12).map(card),
    structure: {
      nodes: arr(p.structure && p.structure.nodes, 80),
      edges: arr(p.structure && p.structure.edges, 120)
    },
    meta: p.meta || {}
  };
}

function renderCard(c) {
  const tags = [];
  if (c.detached) tags.push('detached follow-up');
  if (c.interrupted) tags.push('answer was interrupted');
  return 'Card ' + c.label + (tags.length ? ' (' + tags.join(', ') + ')' : '') +
         '\n  Asked: ' + c.q + '\n  Answer: ' + c.a;
}

function renderMessages(pkg) {
  const out = [];
  const cur = pkg.current;
  out.push('=== CURRENT TURN ===');
  out.push('Question: ' + cur.question);
  if (cur.quote) {
    out.push('');
    out.push('Quoted passage' + (cur.quote.fromCard ? ' (from card ' + cur.quote.fromCard + ')' : '') +
             ' — the question is about this:');
    out.push('"""' + cur.quote.text + '"""');
  }
  out.push(cur.followUpFrom
    ? 'This question is a follow-up from card ' + cur.followUpFrom + '.'
    : 'This question is not anchored to any card.');

  if (pkg.lineage.length) {
    out.push('', '=== LINEAGE (oldest first; the last entry is the card being followed up) ===');
    pkg.lineage.forEach((c) => {
      out.push('[' + (c.role === 'parent' ? 'parent' : 'ancestor') + '] ' + renderCard(c));
    });
  }
  if (pkg.priorFollowUps.length) {
    out.push('', '=== EARLIER FOLLOW-UPS UNDER CARD ' + cur.followUpFrom + ' (in the order they were asked) ===');
    pkg.priorFollowUps.forEach((c) => out.push(renderCard(c)));
  }
  if (pkg.links.length) {
    out.push('', '=== LINKED CONTEXT (connected by the user and marked as context) ===');
    pkg.links.forEach((c) => {
      const v = c.via || {};
      const hops = c.hop > 1 ? ' (' + c.hop + ' links away from card ' + cur.followUpFrom + ')' : '';
      const rel = v.word
        ? 'connected to card ' + v.from + ' with the relation: "' + v.word + '"' + hops
        : 'connected to card ' + v.from + ' — the user gave no label for this connection' + hops;
      out.push('[linked] ' + renderCard(c));
      out.push('  Relation: ' + rel);
    });
  }
  if (pkg.recent.length) {
    out.push('', '=== RECENT CONVERSATION (fallback; oldest first) ===');
    pkg.recent.forEach((c) => out.push(renderCard(c)));
  }

  const st = pkg.structure;
  if (st.nodes.length || st.edges.length) {
    out.push('', '=== BOARD STRUCTURE ===');
    if (st.nodes.length) out.push('Included: ' + st.nodes.map((n) => n.label + ' (' + n.role + ')').join(', '));
    st.edges.forEach((e) => {
      if (e.type === 'followup') out.push('  ' + e.from + ' → ' + e.to + '   follow-up');
      else out.push('  ' + e.from + ' ↔ ' + e.to + '   link' + (e.word ? ' "' + e.word + '"' : ' (no label)') + ', marked as context');
    });
    const m = pkg.meta || {};
    if (typeof m.boardCards === 'number')
      out.push('The board has ' + m.boardCards + ' cards; ' + (m.includedCards || 0) +
               ' are included above. The rest were left out and should not be assumed.');
  }
  return [
    { role: 'system', content: CONTEXT_SYSTEM_PROMPT },
    { role: 'user', content: out.join('\n') }
  ];
}

function logPrompt(pkg, messages) {
  if (!CONTEXT_DEBUG) return;
  const bar = '─'.repeat(72);
  console.log('\n' + bar);
  console.log('[/chat] ' + new Date().toLocaleTimeString() + '  model=' + CHAT_MODEL +
              '  anchored=' + !!pkg.current.followUpFrom + '  quote=' + !!pkg.current.quote +
              '  lineage=' + pkg.lineage.length + ' prior=' + pkg.priorFollowUps.length +
              ' links=' + pkg.links.length + ' recent=' + pkg.recent.length);
  console.log(bar);
  console.log(messages[1].content);
  console.log(bar);
}


// answers a { package } request: the package becomes the prompt (see renderMessages)
async function contextChat(req, res) {
  let pkg;
  try { pkg = sanitizePackage(req.body.package); }
  catch (e) { return res.status(400).json({ error: 'bad context package: ' + e.message }); }
  const messages = renderMessages(pkg);
  logPrompt(pkg, messages);

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('X-Accel-Buffering', 'no');
  try {
    const stream = await openai.chat.completions.create({ model: CHAT_MODEL, messages, stream: true });
    res.on('close', () => { try { stream.controller && stream.controller.abort(); } catch (e) {} });   // user left: stop paying
    for await (const part of stream) {
      const delta = part.choices?.[0]?.delta?.content;
      if (delta) res.write(delta);
    }
    res.end();
  } catch (err) {
    console.error('[board] /chat (context) error:', err.message || err);
    if (!res.headersSent) {
      const status = (err && (err.status || err.statusCode)) || 500;
      res.status(status).type('text/plain').send('model API said ' + status + ': ' + clipS(err.message || String(err), 500));
    } else {
      res.end();
    }
  }
}

const app = express();

// optional password gate: off unless BOARD_PASSWORD is set (Render -> Environment).
// Any username works. /health stays open so Render's checks keep passing.
function samePassword(a, b) {
  const x = Buffer.from(String(a)), y = Buffer.from(String(b));
  return x.length === y.length && crypto.timingSafeEqual(x, y);
}
if (BOARD_PASSWORD) {
  app.use((req, res, next) => {
    if (req.path === '/health') return next();
    const m = /^Basic (.+)$/.exec(req.headers.authorization || '');
    if (m) {
      const pass = Buffer.from(m[1], 'base64').toString('utf8').split(':').slice(1).join(':');
      if (samePassword(pass, BOARD_PASSWORD)) return next();
    }
    res.set('WWW-Authenticate', 'Basic realm="Spatial Board"').status(401).send('Password required');
  });
}
app.use(cors());
app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => res.redirect('/board.html'));

// is this the new server, and is the key set? (never reveals the key)
app.get('/health', (req, res) => {
  res.json({ ok: true, build: BUILD, model: CHAT_MODEL, keyConfigured: !!process.env.OPENAI_API_KEY, passwordGate: !!BOARD_PASSWORD });
});
// dry run for the board's debug panel: the exact prompt a package produces, no model call
app.post('/context/render', express.json({ limit: '2mb' }), (req, res) => {
  try { res.json({ messages: renderMessages(sanitizePackage(req.body && req.body.package)) }); }
  catch (e) { res.status(400).json({ error: 'bad context package: ' + e.message }); }
});

// ---------------------------------------------------------------- /chat ----
// body: { question: string, history: [{ q: string, a: string }] }
// streams plain text chunks (the running "sofar" text is NOT re-sent —
// each chunk is a delta, matching how the client's onToken accumulates).
app.post('/chat', express.json({ limit: '2mb' }), async (req, res) => {
  if (req.body && req.body.package) return contextChat(req, res);   // new: context-package requests
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

// ------------------------------------------------------------------ /tts --
// body: { text: string, voice?: string }
// Real speech instead of the browser's built-in speechSynthesis — that
// voice is a robotic system TTS engine, not a neural one, which is exactly
// why "Read out loud" sounded nothing like the voice in a Realtime call (or
// in ChatGPT/Claude's own apps). This is a one-shot request/response, not a
// stream: the client wants the whole clip before it starts playback anyway,
// so there's no benefit to chunking it and it keeps the client simpler.
app.post('/tts', express.json({ limit: '200kb' }), async (req, res) => {
  const { text, voice } = req.body || {};
  if (!text || typeof text !== 'string' || !text.trim()) {
    return res.status(400).json({ error: 'missing "text" string' });
  }
  const requestedVoice = typeof voice === 'string' ? voice.trim() : '';
  try {
    const speech = await openai.audio.speech.create({
      model: TTS_MODEL,
      voice: requestedVoice || process.env.REALTIME_VOICE || 'marin',
      input: text.slice(0, 4000),   // same practical cap as a card's answer length
      format: 'mp3'
    });
    const buf = Buffer.from(await speech.arrayBuffer());
    res.set('Content-Type', 'audio/mpeg');
    res.send(buf);
  } catch (err) {
    console.error('[board] /tts error:', err.message || err);
    res.status(500).json({ error: 'tts failed: ' + (err.message || String(err)) });
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
          audio: {
            output: { voice: voice },
            // Without this, the API never sends back a transcript of what
            // the PERSON said — only the model's own spoken replies come
            // through for free. That silently broke "save this live
            // conversation as a card": every card landed with a real
            // answer but a blank question, since the client's liveQ never
            // had anything to fill it with.
            input: { transcription: { model: 'gpt-4o-mini-transcribe' } }
          }
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
