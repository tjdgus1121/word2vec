/**
 * morpheme_worker.js
 * Web Worker — TF.js 형태소 분석 전담 (메인 스레드 블로킹 없음)
 * 메시지 프로토콜:
 *   받기: { type:'init' } | { type:'analyze', sentences:[...] }
 *   보내기: { type:'ready', charCount, tagCount }
 *          | { type:'progress', index, total, text }
 *          | { type:'done', allRaw }
 *          | { type:'error', message }
 */

importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.20.0/dist/tf.min.js');

const MAX_LEN = 128;
let model    = null;
let char2idx = null;
let idx2tag  = null;

// ── 초기화 ─────────────────────────────────────────
async function init() {
    const [m, c, i] = await Promise.all([
        tf.loadLayersModel('tfjs_model/model.json'),
        fetch('char2idx.json').then(r => r.json()),
        fetch('idx2tag.json').then(r => r.json())
    ]);
    model    = m;
    char2idx = c;
    idx2tag  = i;
    self.postMessage({
        type: 'ready',
        charCount: Object.keys(c).length,
        tagCount:  Object.keys(i).length
    });
}

// ── 한 문장(어절 묶음) 배치 추론 ───────────────────
function analyzeText(text) {
    const rawEojeols = text.split(/\s+/).filter(w => w);
    const items = [];
    for (const eojeol of rawEojeols) {
        const cleaned = eojeol.replace(/[.,!?;:'"()\[\]…·]/g, '');
        if (cleaned) items.push(Array.from(cleaned));
    }
    if (items.length === 0) return [];

    const batchData = items.map(chars => {
        const ids = chars.map(c => char2idx[c] ?? 1);
        return [...ids, ...new Array(MAX_LEN).fill(0)].slice(0, MAX_LEN);
    });

    const inputT    = tf.tensor2d(batchData, [items.length, MAX_LEN], 'int32');
    const outputT   = model.predict(inputT);
    const allTagIds = outputT.argMax(-1).arraySync();
    inputT.dispose();
    outputT.dispose();

    const result = [];
    for (let i = 0; i < items.length; i++) {
        const morphs = decodeBIO(items[i], allTagIds[i]);
        if (morphs.length > 0) result.push({ morphs });
    }
    return result;
}

// ── BIO 디코딩 ─────────────────────────────────────
function decodeBIO(chars, tagIds) {
    const morphs = [];
    let curLex = '', curTag = '';
    chars.forEach((char, i) => {
        const tag = idx2tag[String(tagIds[i])] ?? 'O';
        if (tag.startsWith('B-')) {
            if (curLex) morphs.push({ lex: curLex, tag: curTag });
            curLex = char;
            curTag = tag.slice(2);
        } else if (tag.startsWith('I-') && curLex) {
            curLex += char;
        } else {
            if (curLex) morphs.push({ lex: curLex, tag: curTag });
            curLex = char;
            curTag = tag === 'O' ? 'NNG' : tag.replace(/^[BI]-/, '');
        }
    });
    if (curLex) morphs.push({ lex: curLex, tag: curTag });
    return morphs;
}

// ── 메시지 핸들러 ───────────────────────────────────
self.onmessage = async function(e) {
    const { type, data } = e.data;

    if (type === 'init') {
        try {
            await init();
        } catch (err) {
            self.postMessage({ type: 'error', message: err.message ?? String(err) });
        }

    } else if (type === 'analyze') {
        const sentences = data.sentences;
        const allRaw    = [];
        for (let i = 0; i < sentences.length; i++) {
            try {
                allRaw.push(analyzeText(sentences[i]));
            } catch {
                allRaw.push([]);
            }
            self.postMessage({ type: 'progress', index: i + 1, total: sentences.length, text: sentences[i] });
        }
        self.postMessage({ type: 'done', allRaw });
    }
};
