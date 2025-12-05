import os
import base64
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

# 1. Setup OpenAI
# This looks for the NEW key, not the Google one.
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL: OPENAI_API_KEY is missing.")
    
client = OpenAI(api_key=api_key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_video(video_path):
    """Extract frames for GPT-4o"""
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    
    fps = video.get(cv2.CAP_PROP_FPS)
    skip_frames = int(fps) # Take 1 frame per second
    
    curr_frame = 0
    while video.isOpened():
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
            
        # Resize to 512px to save money and time
        _, buffer = cv2.imencode(".jpg", cv2.resize(frame, (512, 512)))
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        curr_frame += skip_frames
        
        # Limit to 15 frames max
        if len(base64Frames) >= 15:
            break
            
    video.release()
    return base64Frames

@app.post("/analyze")
async def analyze_video(
    team_name: str = Form(...),
    player_number: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # A. Save Video
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # B. Process Frames
        print("Extracting frames...")
        frames = process_video(temp_filename)

        # C. Send to OpenAI
        print("Sending to GPT-4o...")
        prompt_text = (
            f"Analyze this basketball video for Team: {team_name}, Player: {player_number}. "
            "Identify 2 Strengths, 2 Weaknesses, and 1 Tactical Improvement. Concise."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames)
                ]}
            ],
            max_tokens=400
        )

        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)






