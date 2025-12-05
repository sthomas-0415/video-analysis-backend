import os
import time
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# 1. Setup Application
# This line creates the "app" variable that Render is looking for.
app = FastAPI()

# 2. Configure Google AI
# We use the key from the Environment Variable set in Render
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("WARNING: GOOGLE_API_KEY not found. App will fail if analyzing.")
else:
    genai.configure(api_key=API_KEY)

# 3. Enable CORS (Allows your website to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "System Operational", "backend": "Active"}

@app.post("/analyze")
async def analyze_video(
    team_name: str = Form(...),
    player_number: str = Form(...),
    file: UploadFile = File(...)
):
    temp_filename = f"temp_{file.filename}"
    
    try:
        # Step A: Save video locally
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())

        # Step B: Upload to Google Gemini
        if not API_KEY:
            raise HTTPException(status_code=500, detail="Server Error: Missing API Key")

        print("Uploading file to Google...")
        video_file = genai.upload_file(path=temp_filename)
        
        # Step C: Wait for Processing
        print("Waiting for processing...")
        while video_file.state.name == "PROCESSING":
            time.sleep(2)
            video_file = genai.get_file(video_file.name)

        if video_file.state.name == "FAILED":
            raise HTTPException(status_code=500, detail="Video processing failed on Google side.")

        # Step D: Generate Analysis
        print("Generating analysis...")
        model = genai.GenerativeModel(model_name="gemini-1.5-flash-001")
        
        prompt = (
            f"You are a sports analyst. Analyze the first half of this video for the team '{team_name}'. "
            f"Focus specifically on player number '{player_number}'. "
            "Identify strengths, weaknesses, and key tactical improvements. Keep it concise."
        )
        
        response = model.generate_content([video_file, prompt])

        # Step E: Cleanup
        genai.delete_file(video_file.name)

        return {"analysis": response.text}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


