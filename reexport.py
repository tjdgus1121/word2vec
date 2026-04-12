"""
재변환 스크립트
- 커스텀 손실함수(smoothed_loss) → TF.js 역직렬화 불가 → 표준 손실로 재컴파일
- Mixed precision Policy 객체 → TF.js가 dtype 문자열로만 인식 → float32로 패치
"""
import json, os, shutil
import tensorflow as tf
import tensorflowjs as tfjs

# ── 1. Mixed precision 끄고 모델 로드 ──────────────────
# float32 정책으로 전환해야 레이어 dtype이 문자열로 직렬화됨
tf.keras.mixed_precision.set_global_policy('float32')

print("📦 best_morpheme.keras 로드 중 (compile=False)...")
model = tf.keras.models.load_model('best_morpheme.keras', compile=False)

# 추론 전용 재컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.summary()

# ── 2. TF.js 변환 ──────────────────────────────────────
print("\n🔄 TF.js 변환 중...")
if os.path.exists('tfjs_model_new'):
    shutil.rmtree('tfjs_model_new')
os.makedirs('tfjs_model_new')
tfjs.converters.save_keras_model(model, 'tfjs_model_new')

# ── 3. model.json Policy 객체 패치 ─────────────────────
# Mixed precision이 남긴 {"class_name": "Policy", ...} dtype을 "float32"로 교체
print("\n🔧 model.json dtype Policy 패치 중...")

model_json_path = os.path.join('tfjs_model_new', 'model.json')
with open(model_json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

patched = 0

def fix_dtype(obj):
    global patched
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            val = obj[key]
            if key == 'dtype' and isinstance(val, dict):
                # Policy 객체 또는 기타 dict dtype → float32로 대체
                obj[key] = 'float32'
                patched += 1
            else:
                fix_dtype(val)
    elif isinstance(obj, list):
        for item in obj:
            fix_dtype(item)

fix_dtype(data)

with open(model_json_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)

print(f"   {patched}개 Policy 객체 → 'float32' 교체 완료")

# ── 4. 기존 모델 교체 ──────────────────────────────────
if os.path.exists('tfjs_model'):
    shutil.rmtree('tfjs_model')
shutil.move('tfjs_model_new', 'tfjs_model')

print("\n✅ 재변환 완료")
print("   브라우저에서 Ctrl+Shift+R 로 강력 새로고침하세요.")
