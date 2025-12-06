from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import base64
import os
import tempfile
import logging
import openai

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
async def root():
    return {"message": "Video Analysis API is running"}

@app.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    try:
        # Save video temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        logger.info(f"Processing video: {file.filename}")
        
        # Extract frames
        frames_base64 = []
        cap = cv2.VideoCapture(tmp_path)
        
        frame_count = 0
        max_frames = 3
        success = True
        
        while success and len(frames_base64) < max_frames:
            success, frame = cap.read()
            if not success:
                break
            
            # Take every 10th frame
            if frame_count % 10 == 0:
                try:
                    # Resize to save bandwidth
                    frame = cv2.resize(frame, (640, 360))
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_base64 = base64.b64encode(buffer).decode('utf-8')
                    frames_base64.append(frame_base64)
                except Exception as e:
                    logger.error(f"Error processing frame: {e}")
                    continue
            
            frame_count += 1
        
        cap.release()
        
        if not frames_base64:
            os.unlink(tmp_path)
            raise HTTPException(status_code=400, detail="Could not extract frames from video")
        
        logger.info(f"Extracted {len(frames_base64)} frames")
        
        # Prepare for GPT-4o
        content = [{"type": "text", "text": "Describe what's happening in these video frames:"}]
        
        for frame_b64 in frames_base64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
            })
        
        # Call GPT-4o
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            max_tokens=500
        )
        
        analysis = response.choices[0].message.content
        
        # Cleanup
        os.unlink(tmp_path)
        
        return JSONResponse(content={
            "status": "success",
            "analysis": analysis,
            "frames_processed": len(frames_base64),
            "filename": file.filename
        })
        
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit exceeded")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "video-analysis-api"}

@app.get("/test")
async def test():
    return {"test": "ok", "opencv_version": cv2.__version__}








