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
#   SAFE JSON LOADER
# ============================
def safe_json_loads(text):
    try:
        return True, json.loads(text)
    except Exception as e:
        return False, f"JSON ERROR: {str(e)} | RAW: {text[:200]}"


# ============================
#   PROMPTS
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
Bạn là bác sĩ tim mạch cấp cứu theo ESC 2023/ACC 2024.
Dựa trên vision_ecg + clinical, hãy phân tầng nguy cơ.

QUY TẮC:
- Nguy cơ cao: có ST chênh lên, reciprocal depression, pattern nguy hiểm (De Winter, Wellens, Sgarbossa), hoặc lâm sàng rất nghi ngờ.
- Nguy cơ trung bình: ECG không đặc hiệu + lâm sàng nghi ngờ.
- Nguy cơ thấp: ECG bình thường + triệu chứng gợi ý thấp.

CHỈ TRẢ JSON và PHẢI dùng đúng 2 câu khuyến cáo theo từng mức nguy cơ.

BỘ KHUYẾN CÁO CHUẨN:

1) Nguy cơ CAO:
- "Chuyển bệnh nhân đến cơ sở có khả năng can thiệp mạch vành khẩn cấp ngay lập tức."
- "Duy trì monitoring liên tục và chuẩn bị xử trí rối loạn huyết động hoặc loạn nhịp nguy hiểm trong quá trình vận chuyển."

2) Nguy cơ TRUNG BÌNH:
- "Theo dõi sát triệu chứng và điện tâm đồ, đồng thời lặp lại ECG trong vòng 15–30 phút."
- "Ưu tiên làm Troponin độ nhạy cao nếu có điều kiện, hoặc chuyển tuyến nếu triệu chứng tiến triển."

3) Nguy cơ THẤP:
- "Hướng dẫn bệnh nhân theo dõi triệu chứng và tái khám ngay nếu đau ngực tái phát hoặc tăng lên."
- "Cân nhắc làm xét nghiệm Troponin hoặc tham khảo ý kiến chuyên khoa nếu còn nghi ngờ lâm sàng."

MẪU JSON BẮT BUỘC:
{
  "muc_nguy_co": "",
  "chan_doan_goi_y": "",
  "khuyen_cao": ["...", "..."],
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
                    {"type": "text", "text": "Phân tích ECG sau đây:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ]
            }
        ]
    )

    raw_vision = vision_response.choices[0].message.content
    ok, vision_json = safe_json_loads(raw_vision)

    if not ok:
        return {
            "muc_nguy_co": "không_xác_định",
            "chan_doan_goi_y": "AI không đọc được ECG.",
            "khuyen_cao": [
                "Chụp lại ảnh ECG rõ hơn.",
                "Kiểm tra lại dây dẫn và chất lượng hình ảnh."
            ],
            "giai_thich": raw_vision
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

    raw_clinical = clinical_response.choices[0].message.content
    ok, clinical_json = safe_json_loads(raw_clinical)

    if not ok:
        return {
            "muc_nguy_co": "không_xác_định",
            "chan_doan_goi_y": "AI không xử lý được dữ liệu lâm sàng.",
            "khuyen_cao": [
                "Kiểm tra và nhập lại triệu chứng.",
                "Cân nhắc chuyển tuyến nếu nghi ngờ cao."
            ],
            "giai_thich": raw_clinical
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

    raw_fusion = fusion_response.choices[0].message.content
    ok, fusion_json = safe_json_loads(raw_fusion)

    if not ok:
        return {
            "muc_nguy_co": "không_xác_định",
            "chan_doan_goi_y": "AI không tổng hợp được kết quả.",
            "khuyen_cao": [
                "Theo dõi và lặp lại đánh giá.",
                "Chuyển tuyến nếu triệu chứng không cải thiện."
            ],
            "giai_thich": raw_fusion
        }

    return fusion_json
