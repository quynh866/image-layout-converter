import io
import json
import os

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

app = FastAPI()

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def ocr_with_boxes(image_bytes: bytes):
    # đọc ảnh từ bytes
    image_np = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    h, w, _ = img.shape
    data = pytesseract.image_to_data(img, output_type=Output.DICT, lang="eng")

    blocks = []
    for i, text in enumerate(data["text"]):
        text = text.strip()
        if not text:
            continue
        conf = int(data["conf"][i])
        if conf < 60:
            continue

        x = data["left"][i]
        y = data["top"][i]
        bw = data["width"][i]
        bh = data["height"][i]

        blocks.append({
            "text": text,
            "page": int(data["page_num"][i]),
            "block": int(data["block_num"][i]),
            "line": int(data["line_num"][i]),
            "x": x / w,
            "y": y / h,
            "w": bw / w,
            "h": bh / h,
        })
    return blocks

def blocks_to_html(blocks):
    # Gửi blocks sang GPT để sinh HTML bố cục
    prompt = f"""
Bạn là lập trình viên front-end.

Tôi có danh sách các block text với toạ độ chuẩn hoá (0-1) trên 1 trang,
mỗi block có: text, page, block, line, x, y, w, h.

Yêu cầu:
1. Nhóm các block thành layout hợp lý (header, sidebar, nội dung, vv).
2. Sinh 1 tài liệu HTML hoàn chỉnh, dùng position:absolute theo các toạ độ.
3. Gộp các từ trên cùng một dòng thành 1 <div> hoặc <span>.
4. Không giải thích thêm, chỉ trả về HTML.

Dữ liệu:
{json.dumps(blocks, ensure_ascii=False)}
"""

    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    html = completion.choices[0].message.content
    return html

@app.get("/")
async def root():
    return {"message": "Image → HTML converter is running"}

@app.post("/convert")
async def convert_image(file: UploadFile = File(...)):
    content = await file.read()
    blocks = ocr_with_boxes(content)
    html = blocks_to_html(blocks)
    # Trả trực tiếp HTML
    return HTMLResponse(content=html)

@app.post("/debug_blocks")
async def debug_blocks(file: UploadFile = File(...)):
    content = await file.read()
    blocks = ocr_with_boxes(content)
    return JSONResponse(blocks)
