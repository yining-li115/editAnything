"""
Usage: python discord_bot.py
Command: !replace "replace the cup with a banana" [+ attachment video OR Google Drive link]
"""
import discord
import asyncio
import os
import re
import subprocess
import sys
import json
import functools

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from orchestrator import run
from components.gemini_edit import load_dotenv
load_dotenv()

TOKEN = os.environ.get("DISCORD_BOT_TOKEN")
WORKSPACE = _HERE
MAX_FILE_MB = 25  # Discord free tier limit

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)


def extract_frames(video_path: str, frames_dir: str) -> int:
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-start_number", "1",
        f"{frames_dir}/frame_%05d.png"
    ], check=True, capture_output=True)
    return len([f for f in os.listdir(frames_dir) if f.endswith(".png")])


def download_gdrive(url: str, out_path: str):
    subprocess.run(["yt-dlp", "-o", out_path, url], check=True)


def find_output_video(result: str) -> str | None:
    m = re.search(r'(/[\w/._-]+\.mp4)', result)
    if m and os.path.exists(m.group(1)):
        return m.group(1)
    return None

def format_eval(scores: dict) -> str:
    dims  = scores.get("dimensions") or {}
    raw   = scores.get("raw_metrics") or {}
    caps  = scores.get("caps_applied") or []
    flags = {k for k, v in (scores.get("critical_flags") or {}).items() if v}
    lines = [
        "📊 Evaluation (full):",
        f"  • Edit success: {dims.get('edit_success', 'N/A')}",
        f"  • Source preservation: {dims.get('source_preservation', 'N/A')}",
        f"  • Temporal consistency: {dims.get('temporal_consistency', 'N/A')}",
        f"  • Rendering quality: {dims.get('rendering_quality', 'N/A')}",
        f"  • Final score: {scores.get('final_score', 'N/A')}/1.0",
        "",
        "  Raw metrics:",
        f"    psnr={raw.get('psnr_unedit','N/A')} ssim={raw.get('ssim_unedit','N/A')} "
        f"lpips={raw.get('lpips_unedit','N/A')} structure_dist={raw.get('structure_distance','N/A')}",
        f"    clip_target={raw.get('clip_target','N/A')} niqe={raw.get('niqe','N/A')} mfs={raw.get('mfs','N/A')}",
    ]
    if flags:
        lines.append(f"  ⚠️ Critical flags: {', '.join(sorted(flags))}")
    if caps:
        lines.append(f"  🔒 Caps applied: {', '.join(caps)}")
    return "\n".join(lines)

async def process_request(message, prompt: str, video_path: str):
    job_id = str(message.id)
    job_dir = os.path.join(WORKSPACE, "jobs", job_id)
    frames_dir = os.path.join(job_dir, "frames")

    loop = asyncio.get_event_loop()
    sent_video = {"done": False}

    async def _send_video(video_out):
        if not video_out or not os.path.exists(video_out):
            return
        if os.path.getsize(video_out) / 1e6 <= MAX_FILE_MB:
            await message.channel.send("✅ Video ready!", file=discord.File(video_out))
        else:
            r = subprocess.run(["curl", "-F", f"file=@{video_out}", "https://file.io"],
                               capture_output=True, text=True)
            link = json.loads(r.stdout).get("link", "lien indisponible")
            await message.channel.send(f"✅ Video ready! {link}")

    def on_video(video_out):   # fired from the worker thread the instant encode finishes
        sent_video["done"] = True
        try:
            asyncio.run_coroutine_threadsafe(_send_video(video_out), loop).result(timeout=180)
        except Exception as e:
            print(f"[bot] video send failed: {e}")

    try:
        await message.channel.send("🎞️ Frame extraction...")
        n_frames = extract_frames(video_path, frames_dir)
        await message.channel.send(f"✅ {n_frames} frames extracted")
        first_frame = os.path.join(frames_dir, "frame_00001.png")
        enriched_prompt = (
            f"{prompt}. "
            f"Frames are at {frames_dir}. "
            f"There are {n_frames} frames total. "
            f"The first frame is at {first_frame}. "
            f"Use {frames_dir} as source_frames_dir for both encoding and evaluate. "
            f"Use {video_path} as source_video_path for evaluate. "
            f"Use case_id '{job_id}' and out_dir '{job_dir}' for evaluate. "
            f"Only process segment_starts [0, 48] (2 segments for testing). "

        )

        n_segments = max(1, n_frames // 48)
        await message.channel.send(
            f"🚀 Pipeline running (~{n_segments * 8}-{n_segments * 15} min)...")
        result, eval_scores = await asyncio.wait_for(
            loop.run_in_executor(
                None, functools.partial(run, enriched_prompt, on_video=on_video)
            ),
            timeout=5400
        )
        # the video was already sent by on_video the moment encode finished; fallback if not
        if not sent_video["done"]:
            vo = find_output_video(result)
            if vo:
                await _send_video(vo)
            else:
                await message.channel.send(f"✅ Pipeline finished:\n```{result[:500]}```")
        # evaluation as a SEPARATE follow-up message, once it's ready
        if eval_scores:
            await message.channel.send(f"📊 {format_eval(eval_scores)}")
        else:
            await message.channel.send("📊 Evaluation unavailable (no scores).")

    except asyncio.TimeoutError:
        await message.channel.send("❌ Pipeline timed out after 90 minutes.")
    
    except Exception as e:
        await message.channel.send(f"❌ Error: {str(e)[:500]}")
    finally:
        # nettoyage vidéo source
        if os.path.exists(video_path):
            os.remove(video_path)


@client.event
async def on_ready():
    print(f"[bot] connected as {client.user}")


@client.event
async def on_message(message):
    if message.author == client.user:
        return
    if not message.content.startswith("!replace"):
        return

    prompt = message.content[len("!replace"):].strip()
    if not prompt:
        await message.channel.send(
            "Usage: `!replace <description>` with a video attachment\n"
            "Example: `!replace replace the cup with a banana`")
        return

    job_id = str(message.id)
    job_dir = os.path.join(WORKSPACE, "jobs", job_id)
    os.makedirs(job_dir, exist_ok=True)
    video_path = os.path.join(job_dir, "input.mp4")

    # cas 1: pièce jointe Discord
    if message.attachments:
        att = message.attachments[0]
        if not att.filename.endswith((".mp4", ".mov", ".avi", ".mkv")):
            await message.channel.send("❌ Supported formats: mp4, mov, avi, mkv")
            return
        await message.channel.send("⬇️ Downloading video...")
        await att.save(video_path)
        await process_request(message, prompt, video_path)

    # cas 2: lien Google Drive / YouTube
    elif "drive.google.com" in message.content or "youtu" in message.content:
        url_match = re.search(r'https?://[^\s]+', message.content)
        if not url_match:
            await message.channel.send("❌ Invalid link")
            return
        await message.channel.send("⬇️ Downloading from link...")
        try:
            download_gdrive(url_match.group(0), video_path)
        except Exception as e:
            await message.channel.send(f"❌ Error downloading: {e}")
            return
        await process_request(message, prompt, video_path)

    else:
        await message.channel.send(
            "❌ Please provide a video attachment or a Google Drive link")


if __name__ == "__main__":
    if not TOKEN:
        raise RuntimeError("Set DISCORD_BOT_TOKEN in .env")
    client.run(TOKEN)