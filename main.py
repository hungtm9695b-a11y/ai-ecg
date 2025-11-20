import uvicorn
from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import base64
from io import BytesIO
from openai import OpenAI

client = OpenAI()

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# VISION PROMPT — Tối ưu cho ảnh ECG chụp giấy (loại 1)
# ============================================================
VISION_PROMPT = """
Bạn là chuyên gia tim mạch đọc ECG (ESC/ACC).

Ảnh ECG có thể bị:
- Nền đỏ do giấy
- Méo nhẹ do chụp điện thoại
- Bóng hoặc chói sáng
- Gập nếp

HÃY TẬP TRUNG vào:
- ST elevation (mm và vị trí)
- ST depression
- T-wave: âm, đối xứng, cao nhọn
- Q-wave (pathologic)
- Reciprocal changes
- Dấu hiệu nguy hiểm: Wellens, de Winter, STEMI equivalents

Trả về JSON DUY NHẤT:

{
  "st_elevation": { "co_khong": "...", "vi_tri": "..."},
  "st_depression": "...",
  "t_wave": "...",
  "q_wave": "...",
  "reciprocal": "...",
  "pattern_nguy_hiem": ["..."],
  "ket_luan_ecg": "..."
}
"""


# ============================================================
# CLINICAL PROMPT (Triệu chứng + Risk + HEAR)
# ============================================================
CLINICAL_PROMPT = """
Bạn là bác sĩ tim mạch theo ESC 2023.

Hãy phân tích dữ liệu:
- Tuổi: {age}
- Giới: {sex}
- Huyết áp: {sbp}/{dbp}
- Mạch: {hr}
- SpO2: {spo2}

Triệu chứng:
{trieuchung}

Yếu tố nguy cơ:
{nguyco}

HEAR score = {hear_score} (mức: {hear_level})
HEAR CHỈ HỖ TRỢ, không quyết định thay ESC.

Hãy phân loại:
- Mức độ triệu chứng: điển hình / không điển hình / ít gợi ý
- Đánh giá nguy cơ lâm sàng sơ bộ

Trả về JSON:

{
  "phan_loai_trieu_chung": "...",
  "nguy_co_lam_sang": "cao/trung_binh/thap"
}
"""


# ============================================================
# FUSION PROMPT — Ghép Clinical + ECG theo ESC 2023
# ============================================================
FUSION_PROMPT = """
Bạn là chuyên gia cấp cứu & tim mạch theo ESC/ACC 2023.

Dữ liệu Clinical:
{clinical_json}

Dữ liệu ECG:
{ecg_json}

Nhiệm vụ:
1) Phân loại mức NGUY CƠ CUỐI:
   - cao  
   - trung_binh  
   - thap

2) Chẩn đoán gợi ý (ngắn gọn)

3) Xuất đúng 2 câu khuyến cáo theo tiêu chuẩn ESC:
   - Nếu "cao": chuyển tuyến ngay, chuẩn bị can thiệp.
   - Nếu "trung_binh": theo dõi sát, làm troponin động học.
   - Nếu "thap": ngoại trú, stress-test nếu cần.

Trả về JSON DUY NHẤT:

{
  "muc_nguy_co": "cao/trung_binh/thap",
  "chan_doan_goi_y": "...",
  "khuyen_cao": ["câu1", "câu2"]
}
"""


# ============================================================
# API /analyze
# ============================================================
@app.post("/api/analyze")
async def analyze(
    age: int = Form(...),
    sex: str = Form(...),
    sbp: int = Form(...),
    dbp: int = Form(...),
    hr: int = Form(...),
    spo2: int = Form(...),

    hear_score: int = Form(...),
    hear_level: str = Form(...),

    sx: list[str] = Form(None),
    risk: list[str] = Form(None),

    ecg_file: UploadFile = File(...)
):
    try:
        # ---------------------------------------------
        # 1) Encode ECG image
        # ---------------------------------------------
        img_bytes = await ecg_file.read()
        ecg_base64 = base64.b64encode(img_bytes).decode()

        # ---------------------------------------------
        # 2) Tầng VISION
        # ---------------------------------------------
        vision = client.responses.create(
            model="gpt-4o-mini",
            reasoning={"effort": "medium"},
            input=[
                {"role": "user", "content": VISION_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{ecg_base64}"
                        }
                    ]
                }
            ]
        )

        vision_raw = vision.output_text
        try:
            import json
            vision_json = json.loads(vision_raw)
        except:
            vision_json = {"ket_luan_ecg": "ECG không rõ, ảnh nhiễu"}

        # ---------------------------------------------
        # 3) Tầng CLINICAL
        # ---------------------------------------------
        trieuchung_text = ", ".join(sx) if sx else "Không có"
        nguyco_text = ", ".join(risk) if risk else "Không có"

        clinical_prompt_filled = CLINICAL_PROMPT.format(
            age=age, sex=sex,
            sbp=sbp, dbp=dbp,
            hr=hr, spo2=spo2,
            trieuchung=trieuchung_text,
            nguyco=nguyco_text,
            hear_score=hear_score,
            hear_level=hear_level
        )

        clinical = client.responses.create(
            model="gpt-4o-mini",
            reasoning={"effort": "medium"},
            input=clinical_prompt_filled
        )

        clinical_raw = clinical.output_text
        try:
            clinical_json = json.loads(clinical_raw)
        except:
            clinical_json = {
                "phan_loai_trieu_chung": "khong_ro",
                "nguy_co_lam_sang": "trung_binh"
            }

        # ---------------------------------------------
        # 4) Tầng FUSION
        # ---------------------------------------------
        fusion_prompt_filled = FUSION_PROMPT.format(
            clinical_json=clinical_json,
            ecg_json=vision_json
        )

        fusion = client.responses.create(
            model="gpt-4o-mini",
            reasoning={"effort": "medium"},
            input=fusion_prompt_filled
        )

        import json
        fusion_json = json.loads(fusion.output_text)

        # ---------------------------------------------
        # 5) TRẢ VỀ FULL JSON
        # ---------------------------------------------
        return JSONResponse({
            "muc_nguy_co": fusion_json.get("muc_nguy_co", ""),
            "chan_doan_goi_y": fusion_json.get("chan_doan_goi_y", ""),
            "khuyen_cao": fusion_json.get("khuyen_cao", ["", ""]),
            "ecg": vision_json
        })

    except Exception as e:
        return JSONResponse({"error": str(e)})


# ============================================================
# RUN LOCAL
# ============================================================
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
