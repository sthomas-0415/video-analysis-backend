import os
import time
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. Setup & Key Cleaning (Fixes common errors)
raw_key = os.getenv("GOOGLE_API_KEY", "")
# Remove spaces, quotes, or newlines that might have been pasted by mistake
API_KEY = raw_key.strip().replace('"', '').replace("'", "")

if not API_KEY:
    print("CRITICAL: GOOGLE_API_KEY is missing.")
else:
    genai.configure(api_key=API_KEY)

# 2. CORS (Connection to Website)
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
        # A. Save Video
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # B. Upload to Google
        print("Uploading...")
        video_file = genai.upload_file(path=temp_filename)
        
        # C. Wait for Processing
        print("Processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Google failed to process video.")

        # D. Generate Analysis
        print("Generating...")
        # We use the standard model. 
        # If this 404s, it confirms the library was not updated (See Step 3).
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = (
            f"Analyze this basketball video for Team: {team_name}, Player: {player_number}. "
            "Identify strengths, weaknesses, and tactical improvements."
        )
        
        response = model.generate_content([video_file, prompt])

        # E. Cleanup
        genai.delete_file(video_file.name)

        return {"analysis": response.text}

    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"AI Error: {str(e)}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)





