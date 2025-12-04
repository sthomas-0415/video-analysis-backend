import os, tempfile, base64, cv2, json, re
import gradio as gr

# ---------- SETTINGS ----------
FORCE_MOCK = False         # <- START IN MOCK MODE so the button always works.
print("MODE:", "REAL (OpenAI)" if not FORCE_MOCK else "MOCK")
MAX_FRAMES = 6             # tiny while debugging
BATCH = 3
MODEL = os.getenv("VISION_MODEL", "gpt-4o")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

PROMPT = (
    "You are a veteran sports video analyst. You will see frames sampled "
    "ONLY from the FIRST HALF (0–50%) of a game. The team of interest is identified "
    "by its jersey color (HEX) and a target player number.\n\n"
    "Return compact JSON with keys:\n"
    "  team_weaknesses: [strings]\n"
    "  player_weaknesses: [strings]\n"
    "  improvement_ideas: [strings]\n"
    "  evidence: [ {timestamp: string, note: string} ]\n"
    "Keep items short; avoid generic fluff."
)

def _mock_response(timestamps):
    return {
        "team_weaknesses": ["Slow to match up in transition","Poor weak-side rebounding"],
        "player_weaknesses": ["Late closeouts on shooters","Ball-watching; loses backdoor cuts"],
        "improvement_ideas": ["Assign early help","Weak-side tag & rotate drill","Closeout technique"],
        "evidence": [{"timestamp": f"{t:.1f}s", "note": "Frame suggests breakdown"} for t in timestamps],
    }

def _b64(img_path):
    with open(img_path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()

def _resolve_video_path(video_input):
    # Gradio may pass str path, dict, or object with .name
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        return video_input.get("name") or video_input.get("path") or video_input.get("video")
    return getattr(video_input, "name", None)

def sample_first_half(video_input, max_frames=MAX_FRAMES):
    video_path = _resolve_video_path(video_input)
    print("[DEBUG] Raw video value:", video_input)
    print("[DEBUG] Resolved path:", video_path)

    if not video_path or not os.path.exists(video_path):
        raise FileNotFoundError("Could not resolve uploaded video path from Gradio.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video. Install FFmpeg and/or convert to MP4 (H.264).")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = n / fps if n else 0
    half = dur * 0.5 if dur else 8 * 60
    step = max(half / max_frames, 1.5)

    frames, ts = [], []
    t = 0.0
    while t < half and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        scale = 720 / max(h, w)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        cv2.imwrite(tmp.name, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        frames.append(tmp.name)
        ts.append(round(t, 2))
        t += step
    cap.release()

    if not frames:
        raise RuntimeError("No frames extracted. Convert your file to MP4 (H.264) and try again.")
    print("[DEBUG] Extracted frames:", len(frames))
    return frames, ts

def call_llm(frames_b64, timestamps, team_name, team_hex, player_number):
    if FORCE_MOCK or not OPENAI_API_KEY:
        print("[DEBUG] Using MOCK analysis")
        return _mock_response(timestamps)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)

        content = [{"type": "text",
                    "text": f"Team: {team_name}. Jersey color: {team_hex}. Focus player: #{player_number}.\n" + PROMPT}]
        for url in frames_b64:
            content.append({"type": "image_url", "image_url": {"url": url}})

        chat = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            timeout=60,
        )
        text = chat.choices[0].message.content

        try:
            return json.loads(text)
        except Exception:
            m = re.search(r"\{[\s\S]*\}", text)
            if m:
                return json.loads(m.group(0))
            return {"error": "Model returned non-JSON response."}
    except Exception as e:
        print("[DEBUG] OpenAI error:", e)
        return {"error": f"OpenAI error: {e}"}

def analyze(video, team_name, team_hex, player_number):
    try:
        if video is None:
            return "Please upload a video.", "", "", "", ""
        frames, ts = sample_first_half(video)
    except Exception as e:
        return f"Error: {e}", "", "", "", ""

    # Send tiny batches
    combined = {"team_weaknesses": [], "player_weaknesses": [], "improvement_ideas": [], "evidence": []}
    def dedup(items):
        seen, out = set(), []
        for x in items or []:
            k = x.strip().lower()
            if k not in seen:
                seen.add(k)
                out.append(x)
        return out

    for i in range(0, len(frames), BATCH):
        batch = frames[i:i+BATCH]
        urls = [_b64(p) for p in batch]
        tss = ts[i:i+BATCH]
        res = call_llm(urls, tss, team_name, team_hex, int(player_number))
        if "error" in res:
            return f"Error: {res['error']}", "", "", "", ""
        combined["team_weaknesses"] += res.get("team_weaknesses", [])
        combined["player_weaknesses"] += res.get("player_weaknesses", [])
        combined["improvement_ideas"] += res.get("improvement_ideas", [])
        combined["evidence"] += res.get("evidence", [])

    combined["team_weaknesses"] = dedup(combined["team_weaknesses"])[:8]
    combined["player_weaknesses"] = dedup(combined["player_weaknesses"])[:8]
    combined["improvement_ideas"] = dedup(combined["improvement_ideas"])[:8]

    team_w = "\n".join(f"• {x}" for x in combined["team_weaknesses"]) or "—"
    player_w = "\n".join(f"• {x}" for x in combined["player_weaknesses"]) or "—"
    ideas = "\n".join(f"• {x}" for x in combined["improvement_ideas"]) or "—"
    evid = "\n".join(f"• {e.get('timestamp','?')} — {e.get('note','')}" for e in combined["evidence"][:20]) or "—"

    head = f"Analyzed FIRST HALF only. Team: {team_name} | Player #{player_number} | Color {team_hex}"
    return head, team_w, player_w, ideas, evid

with gr.Blocks(title="AI Insight – Debug") as demo:
    gr.Markdown("## AI Insight (Debug)\nIf something fails, an **Error:** message will appear here and details will print in the Command Prompt window.")
    with gr.Row():
        # Force upload source and mp4 to avoid webcam/file confusion
        video = gr.Video(label="Game video (MP4)", sources=["upload"])
    with gr.Row():
        team = gr.Textbox(label="Team name", value="River City Raptors")
        color = gr.ColorPicker(label="Team color", value="#0ea5e9")
        player = gr.Number(label="Player number", precision=0, value=5)
    btn = gr.Button("Analyze First Half")
    head = gr.Markdown()
    with gr.Row():
        team_w = gr.Markdown(label="Team weaknesses")
        player_w = gr.Markdown(label="Player weaknesses")
    ideas = gr.Markdown(label="Improvement ideas")
    evid = gr.Markdown(label="Evidence")

    btn.click(analyze, inputs=[video, team, color, player],
              outputs=[head, team_w, player_w, ideas, evid])

demo.queue()  # ensure the click is processed
if __name__ == "__main__":
    demo.launch()
