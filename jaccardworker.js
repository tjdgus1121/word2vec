// Cloudflare Workers - Gemini API 프록시
// 배포 방법: Cloudflare Workers 대시보드에서 새 Worker 생성 후 이 코드 붙여넣기

export default {
  async fetch(request, env) {
    // 허용된 도메인 (필요에 따라 추가)
    const allowedOrigins = [
      'https://tjdgus1121.github.io',
      'http://localhost:5500', 
      'http://127.0.0.1:5500',
      'http://localhost:3000',
      'http://127.0.0.1:3000',
      'http://localhost:5502',
      'http://127.0.0.1:5502'
    ];

    const origin = request.headers.get('Origin');
    // 요청 도메인이 허용 목록에 있으면 해당 도메인을, 아니면 와일드카드나 기본값 사용
    // 개발 편의성을 위해 Origin이 없거나(직접 접속) 목록에 없으면 요청된 Origin을 그대로 반환 (보안상 주의 필요하지만 학생용이므로 허용)
    const allowOrigin = origin || '*';

    // CORS 헤더 설정
    const corsHeaders = {
      'Access-Control-Allow-Origin': allowOrigin,
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
      'Access-Control-Max-Age': '86400',
    };

    // OPTIONS 요청 처리 (CORS preflight)
    if (request.method === 'OPTIONS') {
      return new Response(null, { 
        status: 204,
        headers: corsHeaders 
      });
    }

    // GET 요청 처리 (상태 확인용 - 405 에러 방지)
    if (request.method === 'GET') {
      return new Response(JSON.stringify({ 
        status: 'running', 
        message: 'Jaccard Sentiment Analysis Worker is active.',
        usage: 'Send a POST request with { "text": "your text" } to analyze sentiment.',
        version: '1.1.0'
      }), { 
        status: 200,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' } 
      });
    }

    // POST 요청 처리
    if (request.method === 'POST') {
      try {
        // 요청 본문 파싱
        const body = await request.json().catch(() => ({}));
        const { text } = body;

        if (!text || text.trim().length === 0) {
          return new Response(JSON.stringify({ error: '텍스트를 입력해주세요' }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        if (text.length > 100) {
          return new Response(JSON.stringify({ error: '문장이 너무 깁니다. 100글자 이내로 입력해주세요.' }), {
            status: 400,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        // 세부 감정 분석 요청 여부 확인
        const detailAnalysis = body.detailAnalysis || false;

        // Gemini API 키 (Cloudflare Workers 환경 변수에 설정되어 있어야 함)
        const GEMINI_API_KEY = env.GEMINI_API_KEY;

        if (!GEMINI_API_KEY) {
          return new Response(JSON.stringify({ error: 'Worker의 GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.' }), {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        const promptText = detailAnalysis 
          ? `다음 한국어 텍스트를 분석해주세요:

"${text}"

**요구사항:**
1. **핵심 키워드 추출**: 문장에서 실질적인 의미를 가진 **명사, 동사, 형용사, 부사**만 추출하세요.
   - **제외 대상**: 조사(은/는/이/가/을/를 등), 어미(-다/-요 등), 문장부호, 특수기호는 절대 포함하지 마세요.
   - 동사와 형용사는 기본형(예: '가고' -> '가다', '예쁜' -> '예쁘다')으로 변환하세요.
2. **문맥 기반 감성 분석**: 추출된 각 단어가 **이 문장 안에서** 어떤 감정으로 쓰였는지 분석하세요.
3. 세부 감정 태깅: [기쁨, 슬픔, 분노, 놀람, 두려움, 혐오, 중립] 중 하나 선택.

**출력 형식 (JSON Only):**
{
  "morphemes": [
    {"word": "시험", "pos": "명사", "sentiment": "neutral", "specific_emotion": "중립"},
    {"word": "합격하다", "pos": "동사", "sentiment": "positive", "specific_emotion": "기쁨"},
    {"word": "행복하다", "pos": "형용사", "sentiment": "positive", "specific_emotion": "기쁨"},
    {"word": "피곤하다", "pos": "형용사", "sentiment": "negative", "specific_emotion": "슬픔"}
  ],
  "overall_sentiment": "positive",
  "sentiment_scores": { "positive": 2, "neutral": 1, "negative": 1 },
  "specific_emotion_scores": { "기쁨": 2, "슬픔": 1, "분노": 0, "놀람": 0, "두려움": 0, "혐오": 0, "중립": 1 }
}
JSON만 출력하세요.`
          : `다음 한국어 텍스트를 분석해주세요:

"${text}"

**요구사항:**
1. **핵심 키워드 추출**: 문장에서 실질적인 의미를 가진 **명사, 동사, 형용사, 부사**만 추출하세요.
   - **제외 대상**: 조사(은/는/이/가/을/를 등), 어미(-다/-요 등), 문장부호.
   - 동사와 형용사는 기본형으로 변환하세요.
2. **감성 분석**: 각 단어의 감성을 긍정(positive), 중립(neutral), 부정(negative)으로 분류하세요.

**출력 형식 (JSON Only):**
{
  "morphemes": [
    {"word": "단어", "pos": "품사", "sentiment": "positive/neutral/negative"},
    ...
  ],
  "overall_sentiment": "positive",
  "sentiment_scores": { "positive": 0, "neutral": 0, "negative": 0 }
}
JSON만 출력하세요.`;

        // Gemini API 호출
        const geminiResponse = await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent?key=${GEMINI_API_KEY}`,
          {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              contents: [{
                parts: [{
                  text: promptText
                }]
              }],
              generationConfig: {
                temperature: 0.1,
                maxOutputTokens: 2048,
              }
            })
          }
        );

        if (!geminiResponse.ok) {
          const errorText = await geminiResponse.text();
          return new Response(JSON.stringify({ 
            error: 'Gemini API 호출 실패',
            details: errorText 
          }), {
            status: geminiResponse.status,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

        const geminiData = await geminiResponse.json();
        let responseText = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || '';
        
        // JSON 추출 (마크다운 코드 블록 제거)
        responseText = responseText.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
        
        try {
          const analysisResult = JSON.parse(responseText);
          return new Response(JSON.stringify(analysisResult), {
            status: 200,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        } catch (parseError) {
          console.error('JSON Parse Error:', parseError);
          return new Response(JSON.stringify({ 
            error: 'AI 응답 데이터 분석 실패',
            raw: responseText
          }), {
            status: 500,
            headers: { ...corsHeaders, 'Content-Type': 'application/json' }
          });
        }

      } catch (error) {
        return new Response(JSON.stringify({ 
          error: 'Worker 실행 중 오류 발생',
          details: error.message 
        }), {
          status: 500,
          headers: { ...corsHeaders, 'Content-Type': 'application/json' }
        });
      }
    }

    // 그 외 모든 요청
    return new Response('Method Not Allowed', { 
      status: 405, 
      headers: corsHeaders 
    });
  }
};