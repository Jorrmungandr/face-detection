from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import uvicorn

from face_detector import check_face_requirements

app = FastAPI()

@app.post("/api/verify-face")
async def verify_face(file: UploadFile = File(...)):
    file_bytes = await file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)

    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="INVALID_IMAGE")

    polygon = check_face_requirements(frame)

    return {
        "polygon": polygon
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
