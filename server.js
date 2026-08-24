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
const REALTIME_MODEL = process.env.REALTIME_MODEL || 'gpt-4o-realtime-preview';

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
  'question earns a real one.'
].join('\n');

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
// mints a short-lived token so the browser can talk to the Realtime API
// directly over WebRTC without ever seeing OPENAI_API_KEY.
app.post('/session', async (req, res) => {
  try {
    const r = await fetch('https://api.openai.com/v1/realtime/sessions', {
      method: 'POST',
      headers: {
        Authorization: 'Bearer ' + process.env.OPENAI_API_KEY,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: REALTIME_MODEL,
        voice: process.env.REALTIME_VOICE || 'alloy'
      })
    });
    const raw = await r.text();
    let data;
    try { data = JSON.parse(raw); } catch (e) { throw new Error('non-JSON response (' + r.status + '): ' + raw.slice(0, 200)); }
    if (!r.ok) throw new Error(data?.error?.message || 'realtime session request failed');
    res.json(data);
  } catch (err) {
    console.error('[board] /session error:', err.message || err);
    res.status(500).json({ error: 'session failed: ' + (err.message || String(err)) });
  }
});

app.listen(PORT, () => {
  console.log(`[board] listening on http://localhost:${PORT}/board.html`);
});
