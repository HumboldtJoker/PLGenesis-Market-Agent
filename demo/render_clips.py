#!/usr/bin/env python3
"""
Batch render AI-generated clips for the demo video via ComfyUI API.

Reads the shot list, extracts Wan 2.1 prompts, and queues them on
ComfyUI's REST API across available GPUs.

Usage:
    # From the Beast (or any machine with ComfyUI running):
    python render_clips.py                    # Render all AI scenes
    python render_clips.py --scenes 1 4 35 36 # Render specific scenes (hero shots)
    python render_clips.py --dry-run          # Show what would be rendered
    python render_clips.py --variations 1     # Single variation per scene (faster)

Requires: ComfyUI running with Wan 2.1 T2V model loaded.
Default ComfyUI API: http://127.0.0.1:8188
"""

import argparse
import json
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None

logging.basicConfig(level=logging.INFO, format="%(asctime)s [render] %(message)s")
logger = logging.getLogger("render")

# ── Configuration ────────────────────────────────────────────────────────────
COMFYUI_URL = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
OUTPUT_DIR = Path(os.getenv("RENDER_OUTPUT", "./demo/clips"))
SHOTLIST_PATH = Path(__file__).parent / "sovereign-agent-shotlist.md"

# Wan 2.1 defaults from shot list
DEFAULT_STEPS = 35
DEFAULT_CFG = 7.5
DEFAULT_WIDTH = 1280   # 720p
DEFAULT_HEIGHT = 720
DEFAULT_FRAMES = 120   # 5 seconds at 24fps
DEFAULT_FPS = 24
DEFAULT_SAMPLER = "dpm_2m_karras"

# Global negative prompt prefix
NEGATIVE_PREFIX = (
    "worst quality, low quality, normal quality, watermark, signature, "
    "jpeg artifacts, deformed, mutated, disfigured, blurry"
)

HERO_SCENES = {1, 4, 35, 36}  # Get extra variations + upscaling


# ── Shot List Parser ─────────────────────────────────────────────────────────

def parse_shotlist(path: Path = SHOTLIST_PATH) -> List[Dict]:
    """Parse the markdown shot list into structured scene data."""
    content = path.read_text(encoding="utf-8")
    scenes = []

    # Match scene blocks: ### Scene NN — Title
    scene_pattern = re.compile(
        r"### Scene (\d+) — (.+?)\n(.*?)(?=### Scene \d+|## [A-Z]|---|\Z)",
        re.DOTALL,
    )

    for match in scene_pattern.finditer(content):
        scene_num = int(match.group(1))
        title = match.group(2).strip()
        body = match.group(3)

        # Only process AI-generated scenes
        if "AI-generated" not in body:
            continue

        # Extract Wan 2.1 prompt
        prompt_match = re.search(
            r"\*\*Wan 2\.1 Prompt:\*\*\s*`(.+?)`", body, re.DOTALL
        )
        if not prompt_match:
            continue

        prompt = prompt_match.group(1).strip()

        # Extract negative prompt
        neg_match = re.search(
            r"\*\*Negative Prompt:\*\*\s*`(.+?)`", body, re.DOTALL
        )
        negative = neg_match.group(1).strip() if neg_match else ""

        # Combine with global negative prefix
        full_negative = f"{NEGATIVE_PREFIX}, {negative}" if negative else NEGATIVE_PREFIX

        # Extract timestamp
        ts_match = re.search(r"\*\*Timestamp:\*\*\s*(.+)", body)
        timestamp = ts_match.group(1).strip() if ts_match else ""

        scenes.append({
            "scene_num": scene_num,
            "title": title,
            "prompt": prompt,
            "negative_prompt": full_negative,
            "timestamp": timestamp,
            "is_hero": scene_num in HERO_SCENES,
        })

    return scenes


# ── ComfyUI Workflow Builder ─────────────────────────────────────────────────

def build_workflow(prompt: str, negative_prompt: str, seed: int = -1,
                   width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT,
                   frames: int = DEFAULT_FRAMES) -> Dict:
    """
    Build a ComfyUI workflow JSON for Wan 2.1 T2V generation.

    This is a simplified workflow template. The actual node graph depends on
    the ComfyUI installation's available nodes. Adjust node class names
    to match your setup.
    """
    if seed == -1:
        seed = int.from_bytes(os.urandom(4), "big")

    workflow = {
        "client_id": str(uuid.uuid4()),
        "prompt": {
            # KSampler node
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "seed": seed,
                    "steps": DEFAULT_STEPS,
                    "cfg": DEFAULT_CFG,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": 1.0,
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                },
            },
            # Checkpoint loader (Wan 2.1)
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "wan2.1_t2v_720p.safetensors",
                },
            },
            # Empty latent video
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": frames,
                },
            },
            # Positive prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["4", 1],
                },
            },
            # Negative prompt
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": negative_prompt,
                    "clip": ["4", 1],
                },
            },
            # VAE Decode
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["3", 0],
                    "vae": ["4", 2],
                },
            },
            # Save video
            "9": {
                "class_type": "SaveAnimatedWEBP",
                "inputs": {
                    "filename_prefix": "sovereign_agent",
                    "fps": DEFAULT_FPS,
                    "lossless": False,
                    "quality": 90,
                    "method": "default",
                    "images": ["8", 0],
                },
            },
        },
    }

    return workflow


# ── ComfyUI API Client ───────────────────────────────────────────────────────

def queue_prompt(workflow: Dict, server: str = COMFYUI_URL) -> Optional[str]:
    """Queue a workflow on ComfyUI and return the prompt ID."""
    if not requests:
        logger.error("requests library not available")
        return None

    try:
        resp = requests.post(
            f"{server}/prompt",
            json=workflow,
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("prompt_id")
    except Exception as e:
        logger.error("Failed to queue prompt: %s", e)
        return None


def check_progress(prompt_id: str, server: str = COMFYUI_URL) -> Dict:
    """Check the progress of a queued prompt."""
    try:
        resp = requests.get(f"{server}/history/{prompt_id}", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def wait_for_completion(prompt_id: str, server: str = COMFYUI_URL,
                        timeout: int = 600, poll_interval: int = 5) -> bool:
    """Wait for a prompt to complete."""
    start = time.time()
    while time.time() - start < timeout:
        history = check_progress(prompt_id, server)
        if prompt_id in history:
            return True
        time.sleep(poll_interval)
    return False


# ── Main Render Pipeline ─────────────────────────────────────────────────────

def render_scenes(scenes: List[Dict], variations: int = 3,
                  dry_run: bool = False, server: str = COMFYUI_URL):
    """Render all scenes with specified number of variations."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(scenes) * variations
    logger.info("Rendering %d scenes × %d variations = %d clips", len(scenes), variations, total)
    logger.info("Output: %s", OUTPUT_DIR)
    logger.info("Server: %s", server)

    queued = []

    for scene in scenes:
        num = scene["scene_num"]
        scene_variations = variations + 1 if scene["is_hero"] else variations

        logger.info(
            "Scene %02d: %s%s",
            num,
            scene["title"][:50],
            " [HERO]" if scene["is_hero"] else "",
        )

        if dry_run:
            logger.info("  Prompt: %s...", scene["prompt"][:80])
            logger.info("  Variations: %d", scene_variations)
            continue

        for v in range(scene_variations):
            workflow = build_workflow(
                prompt=scene["prompt"],
                negative_prompt=scene["negative_prompt"],
            )

            # Tag the output filename
            workflow["prompt"]["9"]["inputs"]["filename_prefix"] = (
                f"scene_{num:02d}_v{v+1}"
            )

            prompt_id = queue_prompt(workflow, server)
            if prompt_id:
                queued.append({
                    "prompt_id": prompt_id,
                    "scene": num,
                    "variation": v + 1,
                })
                logger.info("  Queued variation %d/%d (id: %s)", v + 1, scene_variations, prompt_id[:8])
            else:
                logger.error("  Failed to queue variation %d/%d", v + 1, scene_variations)

            # Small delay between queues to avoid overwhelming the API
            time.sleep(1)

    if dry_run:
        logger.info("Dry run complete. %d clips would be rendered.", total)
        return

    # Wait for all to complete
    logger.info("All %d clips queued. Waiting for completion...", len(queued))
    completed = 0
    for item in queued:
        if wait_for_completion(item["prompt_id"], server, timeout=600):
            completed += 1
            logger.info(
                "  Scene %02d v%d complete (%d/%d)",
                item["scene"], item["variation"], completed, len(queued),
            )
        else:
            logger.warning(
                "  Scene %02d v%d timed out", item["scene"], item["variation"],
            )

    logger.info("Render complete: %d/%d clips finished", completed, len(queued))


def main():
    parser = argparse.ArgumentParser(description="Render demo video clips via ComfyUI")
    parser.add_argument("--scenes", nargs="+", type=int, help="Specific scene numbers to render")
    parser.add_argument("--variations", type=int, default=3, help="Variations per scene (default: 3)")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be rendered")
    parser.add_argument("--server", default=COMFYUI_URL, help="ComfyUI API URL")
    parser.add_argument("--heroes-only", action="store_true", help="Only render hero scenes (1, 4, 35, 36)")
    args = parser.parse_args()

    scenes = parse_shotlist()
    logger.info("Parsed %d AI-generated scenes from shot list", len(scenes))

    if args.scenes:
        scenes = [s for s in scenes if s["scene_num"] in args.scenes]
        logger.info("Filtered to %d scenes: %s", len(scenes), args.scenes)
    elif args.heroes_only:
        scenes = [s for s in scenes if s["is_hero"]]
        logger.info("Filtered to %d hero scenes", len(scenes))

    if not scenes:
        logger.error("No scenes to render")
        return

    render_scenes(
        scenes,
        variations=args.variations,
        dry_run=args.dry_run,
        server=args.server,
    )


if __name__ == "__main__":
    main()
