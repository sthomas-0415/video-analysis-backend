import os
import base64
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

app = FastAPI()

# 1. Setup OpenAI Client
# We grab the key you just saved in Render
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("CRITICAL: OPENAI_API_KEY is missing.")
client = OpenAI(api_key=api_key)

# 2. Connection Settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_video(video_path):
    """Extracts frames from video to send to GPT-4o"""
    video = cv2.VideoCapture(video_path)
    base64Frames = []
    
    # Get total frame count to calculate spacing
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    
    # If video is short, take every 30th frame. If long, take fewer.
    # We aim for ~10-15 frames to keep it fast and cheap.
    skip_frames = max(int(fps), 30) 
    
    curr_frame = 0
    while video.isOpened():
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
            
        # Resize to speed up upload (Standard practice for GPT-4o)
        _, buffer = cv2.imencode(".jpg", cv2.resize(frame, (512, 512)))
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        
        curr_frame += skip_frames
        
        # Hard limit: Max 20 frames to prevent timeouts
        if len(base64Frames) >= 20:
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
        # A. Save Video Locally
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # B. Extract Frames
        print("Extracting frames...")
        frames = process_video(temp_filename)
        
        if not frames:
            raise HTTPException(status_code=400, detail="Could not read video frames.")

        # C. Send to OpenAI (GPT-4o)
        print(f"Sending {len(frames)} frames to OpenAI...")
        
        prompt_text = (
            f"You are a basketball coach. Analyze these video frames for Team: {team_name}, "
            f"Player Number: {player_number}. "
            "Identify 2 Strengths, 2 Weaknesses, and 1 Tactical Improvement. "
            "Be concise."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt_text},
                    *map(lambda x: {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{x}"}}, frames)
                ]}
            ],
            max_tokens=500
        )

        # D. Return Result
        return {"analysis": response.choices[0].message.content}

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"OpenAI Error: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)




