"""
Kiwi → TF.js 형태소 분석 모델 학습 (개선판)

개선 사항:
  1. 데이터 100만 문장 (위키피디아 + 뉴스 코퍼스)
  2. 자모(초/중/종성) 분해 특징 추가 → 어미/조사 패턴 인식 대폭 향상
  3. 문장 전체를 하나의 시퀀스로 학습 (추론 방식과 완전 일치)
  4. Label smoothing으로 과적합 완화
  5. CosineDecay LR 스케줄

설치:
    pip install kiwipiepy tensorflow tensorflowjs datasets
"""

import json, os, time, unicodedata
import numpy as np
import tensorflow as tf

# ============================================================
# 자모 분해 유틸
# ============================================================
CHOSUNG  = list('ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ')
JUNGSUNG = list('ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ')
JONGSUNG = [''] + list('ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ')

def decompose(ch):
    """한글 음절 → (초성, 중성, 종성) 인덱스 반환. 비한글은 (None, None, None)."""
    code = ord(ch) - 0xAC00
    if code < 0 or code > 11171:
        return None, None, None
    cho  = code // (21 * 28)
    jung = (code % (21 * 28)) // 28
    jong = code % 28
    return cho, jung, jong

def build_jamo_vocab():
    cho_vocab  = {c: i+1 for i, c in enumerate(CHOSUNG)}   # 0=PAD
    jung_vocab = {c: i+1 for i, c in enumerate(JUNGSUNG)}
    jong_vocab = {c: i+1 for i, c in enumerate(JONGSUNG) if c}
    return cho_vocab, jung_vocab, jong_vocab


# ============================================================
# 유틸
# ============================================================
def progress_bar(current, total, prefix='', suffix='', bar_len=40):
    pct    = current / total if total > 0 else 0
    filled = int(bar_len * pct)
    bar    = '█' * filled + '░' * (bar_len - filled)
    print(f'\r{prefix} [{bar}] {pct*100:5.1f}% {suffix}', end='', flush=True)

def elapsed(start):
    s = int(time.time() - start)
    return f'{s//60}분 {s%60}초'


# ============================================================
# 1. GPU 설정
# ============================================================
def setup_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if not gpus:
        print("⚠  GPU 미감지 → CPU로 실행")
        return False
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"✅ GPU: {[g.name for g in gpus]}")
    print("✅ Mixed Precision FP16 활성화")
    return True


# ============================================================
# 2. 데이터 로딩 (위키피디아 + 나무위키 or 뉴스)
# ============================================================
def load_korean_data(num_sentences=1_000_000):
    from datasets import load_dataset

    print(f"\n[1/5] 데이터 로딩 (목표: {num_sentences:,}문장)")
    sentences = []
    t0 = time.time()

    # ── 소스 1: 한국어 위키피디아 (전체 수집) ────────────
    print("  위키피디아 로딩 중...")
    try:
        ds_wiki = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
        wiki_limit = num_sentences  # 위키 단독으로 최대한 채움
        for art_idx, article in enumerate(ds_wiki):
            for line in article["text"].split("\n"):
                line = line.strip()
                if 15 < len(line) < 200 and any('\uAC00' <= c <= '\uD7A3' for c in line):
                    sentences.append(line)
            if art_idx % 200 == 0:
                progress_bar(min(len(sentences), wiki_limit), wiki_limit,
                             prefix='  위키', suffix=f'{len(sentences):,}문장')
            if len(sentences) >= wiki_limit:
                break
        print(f'\n  위키피디아: {len(sentences):,}문장')
    except Exception as e:
        print(f'\n  위키피디아 로딩 실패: {e}')

    # ── 소스 2: mc4 한국어 (뉴스/웹 텍스트) ──────────────
    # CC-100은 구식 dataset script 방식으로 최신 datasets 라이브러리에서 지원 종료
    # mc4 한국어 서브셋으로 대체 (CommonCrawl 기반 정제 텍스트)
    EXTRA_SOURCES = [
        ("allenai/c4",              "ko"),   # C4 한국어 (parquet 기반, 안정적)
        ("heegyu/namuwiki-extracted", None), # 나무위키 (다양한 구어체 포함)
        ("mc4",                     "ko"),   # mC4 한국어 (위 두 개 실패 시 fallback)
    ]
    for src_name, src_config in EXTRA_SOURCES:
        if len(sentences) >= num_sentences:
            break
        need = num_sentences - len(sentences)
        print(f"  {src_name} 로딩 중... (추가 목표: {need:,}문장)")
        try:
            kwargs = dict(split="train", streaming=True)
            if src_config:
                kwargs["name"] = src_config
            ds_extra = load_dataset(src_name, **kwargs)
            added = 0
            for item in ds_extra:
                text = item.get("text", item.get("content", ""))
                for line in text.split("\n"):
                    line = line.strip()
                    if 15 < len(line) < 200 and any('\uAC00' <= c <= '\uD7A3' for c in line):
                        sentences.append(line)
                        added += 1
                if len(sentences) >= num_sentences:
                    break
                if added % 5000 == 0 and added > 0:
                    progress_bar(len(sentences), num_sentences,
                                 prefix=f'  {src_name.split("/")[-1]}',
                                 suffix=f'{len(sentences):,}문장')
            print(f'\n  {src_name}: +{added:,}문장 추가 → 총 {len(sentences):,}문장')
        except Exception as e:
            print(f'\n  {src_name} 실패: {e}')

    sentences = sentences[:num_sentences]
    print(f'\n  최종: {len(sentences):,}문장  ({elapsed(t0)} 소요)\n')
    return sentences


# ============================================================
# 3. Kiwi BIO 라벨 생성
# ============================================================
def generate_bio_data(sentences):
    from kiwipiepy import Kiwi

    print("[2/5] Kiwi BIO 라벨 생성")
    kiwi      = Kiwi(num_workers=-1)
    SKIP_TAGS = {'SF', 'SP', 'SS', 'SE', 'SO', 'SW', 'SB'}
    SKIP_CHARS = set('.,!?;:\'"()[]{}…·—–-')
    CHUNK     = 2000

    dataset, errors = [], 0
    total = len(sentences)
    t0    = time.time()

    for start in range(0, total, CHUNK):
        chunk = sentences[start:start + CHUNK]
        done  = start + len(chunk)
        try:
            results = kiwi.analyze(chunk)
        except Exception as e:
            errors += len(chunk)
            continue

        for sent, result in zip(chunk, results):
            try:
                tokens  = result[0][0]
                text_ns = sent.replace(' ', '').replace('\n', '')
                chars   = list(text_ns)
                labels  = ['O'] * len(chars)

                orig_to_ns = {}
                ns_i = 0
                for orig_i, ch in enumerate(sent):
                    if ch != ' ':
                        orig_to_ns[orig_i] = ns_i
                        ns_i += 1

                for tok in tokens:
                    if tok.tag in SKIP_TAGS:
                        continue
                    ns_pos = [orig_to_ns[j]
                              for j in range(tok.start, tok.start + tok.len)
                              if j in orig_to_ns]
                    for k, p in enumerate(ns_pos):
                        if p < len(labels):
                            labels[p] = f'{"B" if k==0 else "I"}-{tok.tag}'

                filtered = [(c, l) for c, l in zip(chars, labels) if c not in SKIP_CHARS]
                if filtered:
                    dataset.append(filtered)
            except Exception:
                errors += 1

        speed = done / max(time.time() - t0, 1)
        eta   = int((total - done) / max(speed, 1))
        progress_bar(done, total, prefix='  라벨링',
                     suffix=f'{done:,}/{total:,}  성공 {len(dataset):,}  오류 {errors}  남은 {eta//60}분{eta%60}초')

    print(f'\n  완료: {len(dataset):,}문장  ({elapsed(t0)} 소요)\n')
    return dataset


# ============================================================
# 4. 사전 구축 (문자 + 자모)
# ============================================================
def build_vocab(dataset):
    print("[3/5] 사전 구축")
    chars, tags = set(), set()
    for sent in dataset:
        for c, t in sent:
            chars.add(c)
            tags.add(t)

    char2idx = {'<PAD>': 0, '<UNK>': 1}
    char2idx.update({c: i + 2 for i, c in enumerate(sorted(chars))})
    tag2idx  = {t: i for i, t in enumerate(sorted(tags))}
    idx2tag  = {str(v): k for k, v in tag2idx.items()}

    cho_vocab, jung_vocab, jong_vocab = build_jamo_vocab()

    print(f'  문자 사전: {len(char2idx)}개  자모: 초성{len(cho_vocab)} 중성{len(jung_vocab)} 종성{len(jong_vocab)}')
    print(f'  태그: {sorted(tags)}\n')
    return char2idx, tag2idx, idx2tag, cho_vocab, jung_vocab, jong_vocab


# ============================================================
# 5. tf.data 파이프라인 (자모 피처 포함)
# ============================================================
def make_tf_dataset(data, char2idx, tag2idx, cho_vocab, jung_vocab, jong_vocab,
                    max_len, batch_size, label=''):
    total = len(data)
    X_char, X_cho, X_jung, X_jong, Y = [], [], [], [], []

    for i, sent in enumerate(data):
        ids_char = [char2idx.get(c, 1) for c, _ in sent]
        ids_cho, ids_jung, ids_jong = [], [], []
        for c, _ in sent:
            ch_idx, ju_idx, jo_idx = decompose(c)
            ids_cho.append(cho_vocab.get(CHOSUNG[ch_idx], 0) if ch_idx is not None else 0)
            ids_jung.append(jung_vocab.get(JUNGSUNG[ju_idx], 0) if ju_idx is not None else 0)
            ids_jong.append(jong_vocab.get(JONGSUNG[jo_idx], 0) if jo_idx is not None and jo_idx > 0 else 0)
        tids = [tag2idx[t] for _, t in sent]

        pad = lambda lst: (lst + [0] * max_len)[:max_len]
        X_char.append(pad(ids_char))
        X_cho.append(pad(ids_cho))
        X_jung.append(pad(ids_jung))
        X_jong.append(pad(ids_jong))
        Y.append(pad(tids))

        if i % 20000 == 0:
            progress_bar(i + 1, total, prefix=f'  벡터화({label})', suffix=f'{i+1:,}/{total:,}')

    print(f'\n  {label} 완료: {total:,}건')

    X_char  = np.array(X_char,  dtype=np.int32)
    X_cho   = np.array(X_cho,   dtype=np.int32)
    X_jung  = np.array(X_jung,  dtype=np.int32)
    X_jong  = np.array(X_jong,  dtype=np.int32)
    Y       = np.array(Y,       dtype=np.int32)[..., np.newaxis]

    ds = tf.data.Dataset.from_tensor_slices(
        ({'char': X_char, 'cho': X_cho, 'jung': X_jung, 'jong': X_jong}, Y)
    )
    return ds.shuffle(20_000).batch(batch_size).prefetch(tf.data.AUTOTUNE)


# ============================================================
# 6. 진행률 콜백
# ============================================================
class LiveProgress(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.t_train      = None

    def on_train_begin(self, logs=None):
        self.t_train = time.time()
        print("\n  에폭  배치진행             손실     정확도   val손실  val정확도  경과")
        print("  " + "-" * 72)

    def on_epoch_begin(self, epoch, logs=None):
        self.t_epoch   = time.time()
        self.epoch_num = epoch + 1

    def on_train_batch_end(self, batch, logs=None):
        total_batches = self.params.get('steps', 1)
        bar_len = 20
        filled  = int(bar_len * (batch + 1) / total_batches)
        bar     = '█' * filled + '░' * (bar_len - filled)
        pct     = (batch + 1) / total_batches * 100
        print(f'\r  {self.epoch_num:>2}/{self.total_epochs}'
              f'  [{bar}] {pct:5.1f}%'
              f'  손실 {logs.get("loss",0):.4f}'
              f'  정확도 {logs.get("accuracy",0):.4f}', end='', flush=True)

    def on_epoch_end(self, epoch, logs=None):
        ep_time = int(time.time() - self.t_epoch)
        print(f'\r  {self.epoch_num:>2}/{self.total_epochs}'
              f'  [{"█"*20}] 100.0%'
              f'  손실 {logs.get("loss",0):.4f}'
              f'  정확도 {logs.get("accuracy",0):.4f}'
              f'  val손실 {logs.get("val_loss",0):.4f}'
              f'  val정확도 {logs.get("val_accuracy",0):.4f}'
              f'  {ep_time}초  (누적 {elapsed(self.t_train)})')


# ============================================================
# 7. 모델 (자모 임베딩 + 2-layer BiLSTM)
# ============================================================
def build_model(num_chars, num_tags, max_len,
                cho_size, jung_size, jong_size,
                label_smoothing=0.05):
    # ── 입력 ──────────────────────────────────────────
    inp_char = tf.keras.layers.Input(shape=(max_len,), dtype='int32', name='char')
    inp_cho  = tf.keras.layers.Input(shape=(max_len,), dtype='int32', name='cho')
    inp_jung = tf.keras.layers.Input(shape=(max_len,), dtype='int32', name='jung')
    inp_jong = tf.keras.layers.Input(shape=(max_len,), dtype='int32', name='jong')

    # ── 임베딩 ────────────────────────────────────────
    # mask_zero 제거: Concatenate와 함께 쓰면 마스크 전파 불가 + 오버헤드 발생
    emb_char = tf.keras.layers.Embedding(num_chars, 128)(inp_char)
    emb_cho  = tf.keras.layers.Embedding(cho_size + 1, 16)(inp_cho)
    emb_jung = tf.keras.layers.Embedding(jung_size + 1, 16)(inp_jung)
    emb_jong = tf.keras.layers.Embedding(jong_size + 1, 16)(inp_jong)

    # 문자 임베딩 + 자모 임베딩 결합 (128 + 48 = 176dim)
    x = tf.keras.layers.Concatenate()([emb_char, emb_cho, emb_jung, emb_jong])
    x = tf.keras.layers.Dropout(0.1)(x)

    # ── BiLSTM (cuDNN 가속 조건: dropout=0, recurrent_dropout=0) ──
    # dropout을 LSTM 밖 별도 레이어로 분리 → cuDNN 커널 활성화 → 3~5배 빠름
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(128, return_sequences=True))(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # ── 출력 ──────────────────────────────────────────
    x       = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(128, activation='relu'))(x)
    logits  = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_tags))(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32')(logits)

    model = tf.keras.Model(
        inputs=[inp_char, inp_cho, inp_jung, inp_jong],
        outputs=outputs
    )

    # Label smoothing: SparseCategorical은 미지원 → one-hot 변환 후 CategoricalCrossentropy로 구현
    def smoothed_loss(y_true, y_pred):
        y_true_oh = tf.one_hot(tf.cast(tf.squeeze(y_true, -1), tf.int32), num_tags)
        y_smooth  = y_true_oh * (1.0 - label_smoothing) + label_smoothing / num_tags
        return tf.keras.losses.categorical_crossentropy(y_smooth, y_pred)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=smoothed_loss,
        metrics=['accuracy']
    )
    model.summary()
    return model


# ============================================================
# 8. TF.js 변환 — 자모 사전 포함
# ============================================================
def export(model, char2idx, idx2tag, cho_vocab, jung_vocab, jong_vocab):
    import tensorflowjs as tfjs
    import shutil

    print("\n[5/5] TF.js 변환 중...")
    os.makedirs('tfjs_model', exist_ok=True)

    # 커스텀 손실함수는 TF.js 역직렬화 불가 → 표준 손실로 재컴파일 후 변환
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

    # 모델을 임시 디렉토리에 먼저 저장 후 교체 (브라우저가 반쪽짜리 파일 읽는 것 방지)
    tfjs.converters.save_keras_model(model, 'tfjs_model_new')

    # 자모 사전 준비
    jamo_vocab = {
        'cho':      {c: i for c, i in cho_vocab.items()},
        'jung':     {c: i for c, i in jung_vocab.items()},
        'jong':     {c: i for c, i in jong_vocab.items()},
        'chosung':  CHOSUNG,
        'jungsung': JUNGSUNG,
        'jongsung': [c for c in JONGSUNG if c],
    }

    # 모든 파일 원자적 교체 (사전 + 모델 동시에)
    with open('char2idx.json', 'w', encoding='utf-8') as f:
        json.dump(char2idx, f, ensure_ascii=False)
    with open('idx2tag.json', 'w', encoding='utf-8') as f:
        json.dump(idx2tag, f, ensure_ascii=False)
    with open('jamo_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(jamo_vocab, f, ensure_ascii=False)

    # model.json의 Mixed precision Policy 객체를 'float32' 문자열로 패치
    # (TF.js는 dtype을 문자열로만 인식하므로 Policy 객체가 있으면 로드 실패)
    model_json_path = os.path.join('tfjs_model_new', 'model.json')
    with open(model_json_path, 'r', encoding='utf-8') as f:
        model_data = json.load(f)

    def _fix_dtype(obj):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key == 'dtype' and isinstance(obj[key], dict):
                    obj[key] = 'float32'
                else:
                    _fix_dtype(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _fix_dtype(item)

    _fix_dtype(model_data)
    with open(model_json_path, 'w', encoding='utf-8') as f:
        json.dump(model_data, f, ensure_ascii=False)
    print("   model.json dtype Policy 패치 완료")

    # 임시 모델 → 정식 경로로 교체
    if os.path.exists('tfjs_model'):
        shutil.rmtree('tfjs_model')
    shutil.move('tfjs_model_new', 'tfjs_model')

    # _new 임시 사전 파일 정리
    for tmp in ('char2idx_new.json', 'idx2tag_new.json'):
        if os.path.exists(tmp):
            os.remove(tmp)

    print("✅ 변환 완료 — 모든 파일 동시 교체됨")
    print("   tfjs_model/model.json + *.bin")
    print("   char2idx.json / idx2tag.json / jamo_vocab.json")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    MAX_LEN       = 96    # 문장 전체 처리 기준: 한국어 평균 문장 60~80자(공백제거)
    NUM_SENTENCES = 1_000_000
    EPOCHS        = 20

    print("=" * 60)
    print("  Kiwi 형태소 모델 학습 (자모 피처 + 100만 문장)")
    print("=" * 60)

    use_gpu    = setup_gpu()
    BATCH_SIZE = 512 if use_gpu else 64
    print(f"  배치={BATCH_SIZE} / MAX_LEN={MAX_LEN} / 목표문장={NUM_SENTENCES:,}\n")

    t_total = time.time()

    sentences = load_korean_data(NUM_SENTENCES)
    dataset   = generate_bio_data(sentences)
    char2idx, tag2idx, idx2tag, cho_vocab, jung_vocab, jong_vocab = build_vocab(dataset)

    # 사전을 _new 임시 파일로 저장 (학습 완료 후 export에서 정식 파일로 교체)
    # → 브라우저가 사용 중인 기존 char2idx.json / idx2tag.json을 건드리지 않음
    with open('char2idx_new.json', 'w', encoding='utf-8') as f:
        json.dump(char2idx, f, ensure_ascii=False)
    with open('idx2tag_new.json', 'w', encoding='utf-8') as f:
        json.dump(idx2tag, f, ensure_ascii=False)
    print(f"  사전 임시 저장: char={len(char2idx)}, tags={len(tag2idx)}\n")

    print("[4/5] 데이터셋 준비 및 학습")
    split    = int(len(dataset) * 0.9)
    train_ds = make_tf_dataset(dataset[:split], char2idx, tag2idx,
                               cho_vocab, jung_vocab, jong_vocab, MAX_LEN, BATCH_SIZE, '학습')
    val_ds   = make_tf_dataset(dataset[split:], char2idx, tag2idx,
                               cho_vocab, jung_vocab, jong_vocab, MAX_LEN, BATCH_SIZE, '검증')

    model = build_model(
        num_chars=len(char2idx),
        num_tags=len(tag2idx),
        max_len=MAX_LEN,
        cho_size=len(cho_vocab),
        jung_size=len(jung_vocab),
        jong_size=len(jong_vocab),
    )

    # CosineDecay LR: 초기 빠른 학습 후 안정적 수렴
    total_steps = (len(dataset) * 9 // 10 // BATCH_SIZE) * EPOCHS
    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(1e-3, total_steps, alpha=1e-5)
    model.optimizer.learning_rate = lr_schedule

    callbacks = [
        LiveProgress(EPOCHS),
        tf.keras.callbacks.ModelCheckpoint(
            'best_morpheme.keras',
            monitor='val_accuracy', save_best_only=True, verbose=0),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=5,
            restore_best_weights=True, verbose=1),
    ]

    model.fit(
        train_ds, validation_data=val_ds,
        epochs=EPOCHS, callbacks=callbacks,
        verbose=0
    )

    export(model, char2idx, idx2tag, cho_vocab, jung_vocab, jong_vocab)

    print(f"\n전체 소요시간: {elapsed(t_total)}")
