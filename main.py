from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import base64, json

client = OpenAI()
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================
#   HELPER: SAFE JSON PARSER
# ============================
def safe_json_loads(text):
    """
    Parses JSON safely.
    Returns (success, data or error_message)
    """
    try:
        return True, json.loads(text)
    except Exception as e:
        return False, f"JSON ERROR: {str(e)} | Raw: {text[:200]}"


# ============================
#   PROMPTS (3 tầng)
# ============================

VISION_PROMPT = """
Bạn là chuyên gia điện tâm đồ. Hãy trả đúng JSON KHÔNG THÊM TỪ NÀO.

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

CLINICAL_PROMPT = """
Phân tích triệu chứng + sinh hiệu theo ESC 2023.
Chỉ trả JSON:

{
  "phan_loai_lam_sang": "cao | trung_binh | thap",
  "giai_thich_lam_sang": ""
}
"""

FUSION_PROMPT = """
Kết hợp vision + clinical, trả đúng JSON:

{
  "muc_nguy_co": "",
  "chan_doan_goi_y": "",
  "khuyen_cao": [],
  "giai_thich": ""
}
"""


# ============================
#   BACKEND ENDPOINT
# ============================

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

    # ======================
    # Tầng 1 — VISION
    # ======================
    b64_image = base64.b64encode(await ecg_file.read()).decode()

    vision_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": VISION_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Phân tích hình ECG sau:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]
    )

    vision_raw = vision_response.choices[0].message.content
    ok, vision_json = safe_json_loads(vision_raw)

    if not ok:
        return {
            "muc_nguy_co": "không_xác_định",
            "chan_doan_goi_y": "AI không đọc được ECG",
            "khuyen_cao": ["Chụp lại ảnh ECG rõ hơn", "Kiểm tra file ECG"],
            "giai_thich": f"Lỗi Vision: {vision_json}"
        }

    # ======================
    # Tầng 2 — CLINICAL
    # ======================
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

    clinical_raw = clinical_response.choices[0].message.content
    ok, clinical_json = safe_json_loads(clinical_raw)

    if not ok:
        return {
            "muc_nguy_co": "không_xác_định",
            "chan_doan_gợi_y": "AI không xử lý được lâm sàng",
            "khuyen_cao": ["Nhập triệu chứng lại", "Kiểm tra dữ liệu đầu vào"],
            "giai_thich": f"Lỗi Clinical: {clinical_json}"
        }

    # ======================
    # Tầng 3 — FUSION
    # ======================
    fusion_input = f"vision_ecg = {vision_json}\nclinical = {clinical_json}"

    fusion_response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": FUSION_PROMPT},
            {"role": "user", "content": fusion_input}
        ]
    )

    fusion_raw = fusion_response.choices[0].message.content
    ok, fusion_json = safe_json_loads(fusion_raw)

    if not ok:
        return {
            "muc_nguy_co": "không_xác_định",
            "chan_doan_goi_y": "AI không tổng hợp được dữ liệu",
            "khuyen_cao": ["Thử lại sau vài giây"],
            "giai_thich": f"Lỗi Fusion: {fusion_json}"
        }

    return fusion_json
