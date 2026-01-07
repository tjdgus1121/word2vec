// Service Worker for Word2Vec Game
// 오프라인 캐싱 및 성능 최적화

const CACHE_NAME = 'word2vec-game-v1';
const CDN_CACHE = 'word2vec-cdn-v1';

// 캐시할 파일 목록
const STATIC_ASSETS = [
    './',
    './index.html'
];

// Service Worker 설치
self.addEventListener('install', (event) => {
    console.log('[SW] Installing...');
    
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => {
                console.log('[SW] Caching static assets');
                return cache.addAll(STATIC_ASSETS);
            })
            .then(() => self.skipWaiting())
    );
});

// Service Worker 활성화
self.addEventListener('activate', (event) => {
    console.log('[SW] Activating...');
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            return Promise.all(
                cacheNames.map((cacheName) => {
                    if (cacheName !== CACHE_NAME && cacheName !== CDN_CACHE) {
                        console.log('[SW] Deleting old cache:', cacheName);
                        return caches.delete(cacheName);
                    }
                })
            );
        }).then(() => self.clients.claim())
    );
});

// Fetch 이벤트 처리
self.addEventListener('fetch', (event) => {
    const url = new URL(event.request.url);
    
    // jsDelivr CDN 요청 처리 (Word2Vec JSON)
    if (url.hostname === 'cdn.jsdelivr.net') {
        event.respondWith(
            caches.open(CDN_CACHE).then((cache) => {
                return cache.match(event.request).then((cachedResponse) => {
                    if (cachedResponse) {
                        console.log('[SW] Returning cached CDN data');
                        return cachedResponse;
                    }
                    
                    console.log('[SW] Fetching from CDN and caching...');
                    return fetch(event.request).then((response) => {
                        // 유효한 응답만 캐시
                        if (response && response.status === 200) {
                            cache.put(event.request, response.clone());
                        }
                        return response;
                    }).catch((error) => {
                        console.error('[SW] CDN fetch failed:', error);
                        throw error;
                    });
                });
            })
        );
        return;
    }
    
    // 정적 자산 처리 (Cache First 전략)
    event.respondWith(
        caches.match(event.request).then((cachedResponse) => {
            if (cachedResponse) {
                return cachedResponse;
            }
            
            return fetch(event.request).then((response) => {
                // HTML, CSS, JS 파일은 캐시
                if (event.request.destination === 'document' || 
                    event.request.destination === 'script' ||
                    event.request.destination === 'style') {
                    
                    caches.open(CACHE_NAME).then((cache) => {
                        cache.put(event.request, response.clone());
                    });
                }
                
                return response;
            });
        })
    );
});

// 메시지 처리 (캐시 수동 삭제 등)
self.addEventListener('message', (event) => {
    if (event.data.action === 'clearCache') {
        event.waitUntil(
            caches.keys().then((cacheNames) => {
                return Promise.all(
                    cacheNames.map((cacheName) => caches.delete(cacheName))
                );
            }).then(() => {
                event.ports[0].postMessage({success: true});
            })
        );
    }
    
    if (event.data.action === 'skipWaiting') {
        self.skipWaiting();
    }
});
