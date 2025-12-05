import os
import time
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. Setup Google AI
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("CRITICAL ERROR: GOOGLE_API_KEY is missing.")
else:
    genai.configure(api_key=API_KEY)

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_video(
    team_name: str = Form(...),
    player_number: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # A. Save File
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # B. Upload to Google
        print("Uploading to Gemini...")
        video_file = genai.upload_file(path=temp_filename)
        
        # C. Wait for Processing
        print(f"Processing video: {video_file.name}")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Video processing failed on Google side.")

        # D. Generate Content (Using Standard Model Name)
        print("Generating analysis...")
        
        # We use the standard flash model. 
        # If this fails, the requirements.txt update didn't work.
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        prompt = (
            f"You are a sports analyst. Analyze this video for {team_name}, "
            f"player {player_number}. List strengths, weaknesses, and tactical advice."
        )
        
        response = model.generate_content([video_file, prompt])

        # E. Cleanup
        genai.delete_file(video_file.name)

        return {"analysis": response.text}

    except Exception as e:
        print(f"ERROR DETAILS: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
import os
import time
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. Setup Google AI
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("CRITICAL ERROR: GOOGLE_API_KEY is missing.")
else:
    genai.configure(api_key=API_KEY)

# 2. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze")
async def analyze_video(
    team_name: str = Form(...),
    player_number: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # A. Save File
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # B. Upload to Google
        print("Uploading to Gemini...")
        video_file = genai.upload_file(path=temp_filename)
        
        # C. Wait for Processing
        print(f"Processing video: {video_file.name}")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Video processing failed on Google side.")

        # D. Generate Content (Using Standard Model Name)
        print("Generating analysis...")
        
        # We use the standard flash model. 
        # If this fails, the requirements.txt update didn't work.
        model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        
        prompt = (
            f"You are a sports analyst. Analyze this video for {team_name}, "
            f"player {player_number}. List strengths, weaknesses, and tactical advice."
        )
        
        response = model.generate_content([video_file, prompt])

        # E. Cleanup
        genai.delete_file(video_file.name)

        return {"analysis": response.text}

    except Exception as e:
        print(f"ERROR DETAILS: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


