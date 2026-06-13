import os
import uuid
import json
import asyncio
import subprocess
import zipfile
from pathlib import Path
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
import yt_dlp
import whisper
import uvicorn

app = FastAPI(title="ClipForge")
app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")
templates = Jinja2Templates(directory="templates")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
jobs = {}


class ProcessRequest(BaseModel):
    url: str
    num_clips: int = 5
    clip_duration: int = 45


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/process")
async def process_video(req: ProcessRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "queued", "progress": 0, "message": "Starting...", "clips": [], "zip_url": ""}
    background_tasks.add_task(run_pipeline, job_id, req.url, req.num_clips, req.clip_duration)
    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/api/download/{job_id}")
async def download_zip(job_id: str):
    zip_path = f"outputs/{job_id}/shorts.zip"
    if not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP not ready")
    return FileResponse(zip_path, media_type="application/zip", filename="shorts.zip")


def update_job(job_id, status, progress, message, clips=None, zip_url=""):
    jobs[job_id].update({"status": status, "progress": progress, "message": message, "zip_url": zip_url})
    if clips is not None:
        jobs[job_id]["clips"] = clips


async def run_pipeline(job_id, url, num_clips, clip_duration):
    output_dir = Path(f"outputs/{job_id}")
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        update_job(job_id, "processing", 10, "⬇️ Downloading video...")
        video_path = await download_video(url, output_dir)

        update_job(job_id, "processing", 25, "🎵 Extracting audio...")
        audio_path = output_dir / "audio.mp3"
        subprocess.run(["ffmpeg", "-i", str(video_path), "-q:a", "0", "-map", "a", str(audio_path), "-y"], check=True, capture_output=True)

        update_job(job_id, "processing", 40, "📝 Transcribing audio...")
        transcript = await transcribe_audio(str(audio_path))

        update_job(job_id, "processing", 60, "🧠 AI finding best moments...")
        moments = await find_best_moments(transcript, num_clips, clip_duration)

        update_job(job_id, "processing", 75, "✂️ Cutting clips...")
        clips = await cut_clips(video_path, moments, output_dir, clip_duration)

        update_job(job_id, "processing", 90, "📦 Creating ZIP...")
        zip_path = output_dir / "shorts.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for clip in clips:
                zf.write(clip["path"], os.path.basename(clip["path"]))

        clip_list = [{"title": c["title"], "start": c["start"], "end": c["end"], "hook": c.get("hook", ""), "url": f"/outputs/{job_id}/{os.path.basename(c['path'])}"} for c in clips]
        update_job(job_id, "done", 100, f"✅ {len(clips)} shorts ready!", clips=clip_list, zip_url=f"/api/download/{job_id}")

    except Exception as e:
        update_job(job_id, "error", 0, f"❌ Error: {str(e)}")


async def download_video(url, output_dir):
    video_path = output_dir / "video.mp4"
    ydl_opts = {"format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best", "outtmpl": str(video_path), "merge_output_format": "mp4", "quiet": True}
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, lambda: yt_dlp.YoutubeDL(ydl_opts).download([url]))
    return video_path


async def transcribe_audio(audio_path):
    loop = asyncio.get_event_loop()
    def _transcribe():
        model = whisper.load_model("base")
        result = model.transcribe(audio_path, word_timestamps=True)
        return [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in result["segments"]]
    return await loop.run_in_executor(None, _transcribe)


async def find_best_moments(segments, num_clips, clip_duration):
    transcript_text = "\n".join([f"[{s['start']:.1f}s - {s['end']:.1f}s]: {s['text']}" for s in segments])
    prompt = f"""You are an expert viral short-form video editor.

Transcript with timestamps:
{transcript_text[:8000]}

Find the {num_clips} BEST moments for viral YouTube Shorts / TikTok clips.
Each clip ~{clip_duration} seconds. Pick exciting, dramatic, or key moments. No overlaps.

Return ONLY this JSON, no extra text:
{{
  "clips": [
    {{
      "title": "Catchy title",
      "start": 12.5,
      "end": 57.5,
      "hook": "First 3-second hook line"
    }}
  ]
}}"""

    message = client.messages.create(model="claude-sonnet-4-6", max_tokens=1000, messages=[{"role": "user", "content": prompt}])
    text = message.content[0].text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())["clips"]


async def cut_clips(video_path, moments, output_dir, clip_duration):
    clips = []
    for i, moment in enumerate(moments):
        start = moment["start"]
        end = moment.get("end", start + clip_duration)
        output_path = output_dir / f"short_{i+1}.mp4"
        cmd = ["ffmpeg", "-ss", str(start), "-i", str(video_path), "-t", str(end - start), "-vf", "crop=ih*9/16:ih,scale=1080:1920", "-c:v", "libx264", "-c:a", "aac", "-preset", "fast", str(output_path), "-y"]
        subprocess.run(cmd, capture_output=True)
        if output_path.exists():
            clips.append({"title": moment.get("title", f"Short {i+1}"), "start": start, "end": end, "path": str(output_path), "hook": moment.get("hook", "")})
    return clips


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
