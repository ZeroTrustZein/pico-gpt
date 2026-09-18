"""Build presentation video assets: Option A (Trailer), Option B (Self-Presentation), Option C (Cinematic Cut).

Generates narration audio via edge-tts and composites with slide visuals using ffmpeg.
"""
from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
from pathlib import Path

import edge_tts
import imageio_ffmpeg

ROOT = Path(__file__).resolve().parent
FF = imageio_ffmpeg.get_ffmpeg_exe()
SLIDES_DIR = ROOT / "media" / "slides" if (ROOT / "media" / "slides").exists() else ROOT / "deck_preview"
WORK_DIR = ROOT / "video_work"
MEDIA_DIR = ROOT / "media"

DRAMA_VOICE = "en-US-ChristopherNeural"
BRIT_VOICE = "en-GB-RyanNeural"


def get_duration(path: Path) -> float:
    r = subprocess.run([FF, "-i", str(path)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", r.stderr)
    if not m:
        raise RuntimeError(f"Could not determine duration for {path}")
    h, mn, s = m.groups()
    return int(h) * 3600 + int(mn) * 60 + float(s)


async def synthesize_speech(text: str, voice: str, out_path: Path, rate: str = "+0%") -> None:
    await edge_tts.Communicate(text, voice, rate=rate).save(str(out_path))


def create_still_clip(image: Path, audio: Path, out_path: Path) -> None:
    subprocess.run(
        [
            FF, "-y", "-loop", "1", "-i", str(image), "-i", str(audio),
            "-c:v", "libx264", "-tune", "stillimage", "-c:a", "aac",
            "-b:a", "192k", "-pix_fmt", "yuv420p", "-shortest",
            "-vf", "scale=1280:720", str(out_path)
        ],
        capture_output=True, text=True, check=True
    )


def create_cinematic_clip(image: Path, audio: Path, out_path: Path, zoom_in: bool = True) -> None:
    d = get_duration(audio)
    frames = max(int(d * 25) + 10, 50)
    if zoom_in:
        z = f"zoompan=z='min(zoom+0.0008,1.25)':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps=25"
    else:
        z = f"zoompan=z='if(eq(on,1),1.25,max(zoom-0.0015,1.0))':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1280x720:fps=25"
    vf = f"{z},scale=1280:536:force_original_aspect_ratio=increase,crop=1280:536,pad=1280:720:(ow-iw)/2:(oh-ih)/2,format=yuv420p"
    subprocess.run(
        [
            FF, "-y", "-loop", "1", "-i", str(image), "-i", str(audio),
            "-filter_complex", f"[0:v]{vf}[v]",
            "-map", "[v]", "-map", "1:a", "-c:v", "libx264", "-tune", "stillimage",
            "-c:a", "aac", "-b:a", "192k", "-shortest", str(out_path)
        ],
        capture_output=True, text=True, check=True
    )


def concat_clips(clips: list[Path], out_path: Path) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    listfile = WORK_DIR / f"{out_path.stem}_list.txt"
    listfile.write_text("".join(f"file '{c.as_posix()}'\n" for c in clips), encoding="utf-8")
    subprocess.run(
        [FF, "-y", "-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out_path)],
        capture_output=True, text=True, check=True
    )


def build_sequence(name: str, segments: list[dict], out_path: Path, cinematic: bool = False) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    clips = []
    for i, seg in enumerate(segments):
        audio_file = WORK_DIR / f"{name}_{i:02d}.mp3"
        asyncio.run(synthesize_speech(seg["text"], seg["voice"], audio_file, seg.get("rate", "+0%")))
        clip_file = WORK_DIR / f"{name}_{i:02d}.mp4"
        slide_file = SLIDES_DIR / seg["slide"]
        if cinematic:
            create_cinematic_clip(slide_file, audio_file, clip_file, zoom_in=(i % 2 == 0))
        else:
            create_still_clip(slide_file, audio_file, clip_file)
        clips.append(clip_file)
    concat_clips(clips, out_path)
    print(f"Built {out_path.name}: {get_duration(out_path):.1f}s, {out_path.stat().st_size / 1e6:.2f} MB")


def get_segments(speech_text: str) -> tuple[list[dict], list[dict], list[dict]]:
    trailer = [
        dict(slide="Slide1.PNG", voice=DRAMA_VOICE, rate="-8%",
             text="In a world of trillion-parameter giants... one tiny model dared to dream."),
        dict(slide="Slide3.PNG", voice=DRAMA_VOICE, rate="-8%",
             text="Built from scratch. No shortcuts. No pre-trained giants. A tokenizer. Attention. A training loop. All hand-written. All proven."),
        dict(slide="Slide4.PNG", voice=DRAMA_VOICE, rate="-8%",
             text="Its bigger brother? Fifty-nine million parameters. It memorized everything... and learned nothing. So its own creator shut it down."),
        dict(slide="Slide6.PNG", voice=DRAMA_VOICE, rate="-8%",
             text="Thirty-four tests. Every claim proven. Every bug... caught by the machine that made it."),
        dict(slide="Slide8.PNG", voice=DRAMA_VOICE, rate="-12%",
             text="pico-gpt. Three point three million parameters. One unforgettable lesson. Small... verified... and impossible to ignore."),
    ]
    self_pres = [
        dict(slide="Slide1.PNG", voice=BRIT_VOICE,
             text="Hello. I am pico-gpt. I have been asked to introduce myself. I will do so... in the only language I know."),
        dict(slide="Slide5.PNG", voice=BRIT_VOICE,
             text="I give you... my acceptance speech. Word for word. Generated by me, this very week."),
        dict(slide="Slide5.PNG", voice=BRIT_VOICE, text=speech_text),
        dict(slide="Slide8.PNG", voice=BRIT_VOICE,
             text="Yes. That was my acceptance speech. I am three million parameters. I do not know what honour means. But I passed thirty-four tests... and my fifty-nine-million-parameter brother did not. I am pico-gpt. Thank you."),
    ]
    return trailer, self_pres, trailer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--option", choices=["A", "B", "C", "all"], default="all")
    parser.add_argument("--out-dir", default="media")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speech_file = ROOT / "media" / "model_speech.txt" if (ROOT / "media" / "model_speech.txt").exists() else ROOT / "model_speech.txt"
    speech_text = speech_file.read_text(encoding="utf-8").strip()

    trailer, self_pres, cine = get_segments(speech_text)

    if args.option in ("A", "all"):
        build_sequence("trailer", trailer, out_dir / "option_A_trailer.mp4", cinematic=False)
    if args.option in ("B", "all"):
        build_sequence("selfpres", self_pres, out_dir / "option_B_model_presents.mp4", cinematic=False)
    if args.option in ("C", "all"):
        build_sequence("cinematic", cine, out_dir / "option_C_cinematic.mp4", cinematic=True)


if __name__ == "__main__":
    main()
