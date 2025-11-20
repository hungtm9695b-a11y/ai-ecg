from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64, json

client = OpenAI()
app = FastAPI()

# CORS để frontend GitHub Pages gọi được
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================
#   TẦNG 1 — VISION
# ======================
VISION_PROMPT = """
Bạn là chuyên gia điện tâm đồ. 
Chỉ được mô tả điện tâm đồ từ ảnh ECG.

Phải tìm:
- ST chênh lên (mm, vị trí)
- ST chênh xuống
- Sóng T âm/bất đối xứng
- Sóng Q bệnh lý
- Reciprocal depression
- Pattern nguy hiểm: De Winter, Wellens, Sgarbossa

Không được chẩn đoán ACS.

Trả đúng JSON:

{
  "st_elevation": {"co_khong":"", "vi_tri":"", "mien_do":""},
  "st_depression": "",
  "t_wave": "",
  "q_wave": "",
  "reciprocal": "",
  "pattern_nguy_hiem": [],
  "ket_luan_ecg": ""
}
"""
# ======================
#  TẦNG 2 — CLINICAL
# ======================
CLINICAL_PROMPT = """
Bạn là bác sĩ tim mạch theo ESC 2023 / ACC 2024.
Chỉ được phân tích TRIỆU CHỨNG + SINH HIỆU, KHÔNG sử dụng ECG.

Trả đúng JSON:

{
  "phan_loai_lam_sang": "cao | trung_binh | thap",
  "giai_thich_lam_sang": ""
}
"""
# ======================
#   TẦNG 3 — FUSION
# ======================
FUSION_PROMPT = """
Bạn là bác sĩ tim mạch cấp cứu theo ESC 2023.
Kết hợp vision_ecg + clinical.

QUY TẮC FUSION:
- Có ST chênh lên, pattern nguy hiểm → NGUY CƠ CAO
- ECG không đặc hiệu + lâm sàng nghi ngờ → TRUNG BÌNH
- ECG bình thường + lâm sàng thấp → THẤP
- Thiếu thông tin → nâng mức nguy cơ

Trả đúng JSON:

{
  "muc_nguy_co": "",
  "chan_doan_goi_y": "",
  "khuyen_cao": [],
  "giai_thich": ""
}
"""
# ======================
#  ENDPOINT CHÍNH
# ======================
@app.post("/api/analyze")
async def analyze(
    age: int = Form(...),
    sex: str = Form(...),
    sbp: int = Form(...),
    dbp: int = Form(...),
    hr: int = Form(...),
    spo2: int = Form(...),
    symptom_main: str = Form(...),
    duration: int = Form(...),
    radiation: str = Form(...),
    gt_nitrate: str = Form(...),
    risk: str = Form(...),
    ecg_file: UploadFile = File(...)
):

    # ---- Đọc ảnh ECG ----
    b64_image = base64.b64encode(await ecg_file.read()).decode()

    # ==========================
    #  CALL TẦNG 1 — VISION AI
    # ==========================
    vision_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Phân tích ECG."},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]
    )

    vision_json = json.loads(vision_response.choices[0].message.content)

    # ==========================
    #   CALL TẦNG 2 — CLINICAL
    # ==========================
    clinical_input = f"""
    Tuổi: {age}
    Giới: {sex}
    Huyết áp: {sbp}/{dbp}
    Mạch: {hr}
    SpO2: {spo2}
    Triệu chứng: {symptom_main}
    Thời gian đau: {duration}
    Lan: {radiation}
    Đáp ứng nitrate: {gt_nitrate}
    Nguy cơ: {risk}
    """

    clinical_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": CLINICAL_PROMPT},
            {"role": "user", "content": clinical_input}
        ]
    )

    clinical_json = json.loads(clinical_response.choices[0].message.content)

    # ==========================
    #   CALL TẦNG 3 — FUSION
    # ==========================
    fusion_input = f"""
    vision_ecg = {json.dumps(vision_json)}
    clinical = {json.dumps(clinical_json)}
    """

    fusion_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": FUSION_PROMPT},
            {"role": "user", "content": fusion_input}
        ]
    )

    fusion_json = json.loads(fusion_response.choices[0].message.content)

    return fusion_json
