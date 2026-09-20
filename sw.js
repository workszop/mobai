/* KlockiAI Mobile service worker: offline app shell + CDN libraries.
   The MobileNet base model is not cached here; the app stores it in IndexedDB. */
const CACHE = 'klocki-mobile-v4';
const SHELL = [
  './',
  './index.html',
  './site.webmanifest',
  './favicon.svg',
  './edulab-mark-ink.png',
  './favicon-96x96.png',
  './favicon.ico',
  './apple-touch-icon.png',
  './web-app-manifest-192x192.png',
  './web-app-manifest-512x512.png',
  'https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js',
  'https://cdn.jsdelivr.net/npm/lucide@1.47.0/dist/umd/lucide.min.js',
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE)
      .then(c => Promise.allSettled(SHELL.map(u => c.add(u))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  const isShell = url.origin === self.location.origin;
  const isCdn = url.hostname === 'cdn.jsdelivr.net';
  const isFont = url.hostname === 'fonts.googleapis.com' || url.hostname === 'fonts.gstatic.com';
  if (!isShell && !isCdn && !isFont) return; // model downloads etc. go straight to the network

  if (isShell && (req.mode === 'navigate' || url.pathname.endsWith('index.html') || url.pathname.endsWith('/'))) {
    // Network first for the page itself, so updates land; cache is the offline fallback.
    event.respondWith(
      fetch(req).then(res => { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); return res; })
        .catch(() => caches.match(req).then(r => r || caches.match('./index.html')))
    );
    return;
  }
  // Cache first for static assets, pinned CDN libraries and web fonts (CSS + woff2).
  event.respondWith(
    caches.match(req).then(hit => hit || fetch(req).then(res => {
      if (res && (res.ok || res.type === 'opaque')) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }))
  );
});
