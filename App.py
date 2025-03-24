import os
import sys
import subprocess
import time
import json
import glob
import shutil
import platform
import threading
import argparse
from datetime import datetime
from pathlib import Path

import torch
import gradio as gr
import numpy as np
import cv2
from tqdm import tqdm

# Global variables
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")
os.makedirs(CONFIG_DIR, exist_ok=True)

LAST_CONFIG_FILE = os.path.join(CONFIG_DIR, "last_used_config.txt")
DEFAULT_CONFIG_NAME = "Default"

# Global flags for cancelling batch processing
cancel_batch_processing = False

# Global variables for model caching
current_loaded_model_type = None  # "rife" or "upscale"
current_loaded_upscale_model = None

# Statistics variables
processing_stats = {
    "total_videos": 0,
    "processed_videos": 0,
    "skipped_videos": 0,
    "failed_videos": 0,
    "start_time": None,
    "current_video": "",
    "current_video_start_time": None,
    "eta": "N/A",
    "average_speed": "N/A"
}

# Default configuration
def get_default_config():
    return {
        # General settings
        "ffmpeg_gpu": True,
        "ffmpeg_preset": "medium",
        "ffmpeg_crf": 18,
        
        # RIFE settings
        "rife_model_dir": os.path.abspath(os.path.join("Practical-RIFE", "train_log")),
        "rife_multiplier": "2x",
        "rife_fp16": True,
        "rife_uhd": False,
        "rife_scale": 1.0,
        "rife_skip_static": False,
        
        # Upscale settings
        "upscale_model": "RealESRGAN_x4plus",
        "upscale_preset": "general",
        "upscale_outscale": 4,
        "upscale_tile": 0,
        "upscale_tile_pad": 10,
        "upscale_pre_pad": 0,
        "upscale_face_enhance": False,
        "upscale_fp32": False,
        "upscale_denoise_strength": 0.5,
        
        # Batch settings
        "batch_input_folder": "",
        "batch_output_folder": "outputs",
        "skip_existing": True,
        "include_subfolders": True,
        "save_gen_params": True
    }

# Config management functions
def get_config_list():
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    files = os.listdir(CONFIG_DIR)
    configs = [os.path.splitext(f)[0] for f in files if f.endswith(".json") and not f == "last_used_config.txt"]
    return sorted(configs)

def save_config(config_name, config_data):
    if not os.path.exists(CONFIG_DIR):
        os.makedirs(CONFIG_DIR)
    
    config_file_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    with open(config_file_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    
    with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(config_name)
    
    return config_name

def load_config(config_name):
    config_file_path = os.path.join(CONFIG_DIR, f"{config_name}.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return get_default_config()

# Initialize config
default_config = get_default_config()

if os.path.exists(LAST_CONFIG_FILE):
    with open(LAST_CONFIG_FILE, "r", encoding="utf-8") as f:
        last_config_name = f.read().strip()
    config_file_path = os.path.join(CONFIG_DIR, f"{last_config_name}.json")
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = last_config_name
    else:
        default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
        if os.path.exists(default_config_path):
            with open(default_config_path, "r", encoding="utf-8") as f:
                config_loaded = json.load(f)
            last_config = DEFAULT_CONFIG_NAME
        else:
            config_loaded = default_config
            with open(default_config_path, "w", encoding="utf-8") as f:
                json.dump(config_loaded, f, indent=4)
            last_config = DEFAULT_CONFIG_NAME
            with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
                f.write(last_config)
else:
    default_config_path = os.path.join(CONFIG_DIR, f"{DEFAULT_CONFIG_NAME}.json")
    if os.path.exists(default_config_path):
        with open(default_config_path, "r", encoding="utf-8") as f:
            config_loaded = json.load(f)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)
    else:
        config_loaded = default_config
        with open(default_config_path, "w", encoding="utf-8") as f:
            json.dump(config_loaded, f, indent=4)
        last_config = DEFAULT_CONFIG_NAME
        with open(LAST_CONFIG_FILE, "w", encoding="utf-8") as f:
            f.write(DEFAULT_CONFIG_NAME)

# Helper functions
def get_next_output_filename(extension="mp4"):
    """Generate a sequential filename that doesn't overwrite existing files"""
    i = 1
    while True:
        filename = os.path.join(OUTPUTS_DIR, f"{i:04d}.{extension}")
        if not os.path.exists(filename):
            return filename
        i += 1

def save_params_file(params_dict, video_path):
    """Save parameters to a text file with the same name as the video"""
    params_path = os.path.splitext(video_path)[0] + ".txt"
    with open(params_path, "w", encoding="utf-8") as f:
        f.write(f"Video Enhancement Parameters\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for section, params in params_dict.items():
            f.write(f"[{section}]\n")
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")

def update_processing_stats(key, value):
    """Update processing statistics"""
    global processing_stats
    processing_stats[key] = value
    if processing_stats["start_time"] is not None and processing_stats["processed_videos"] > 0:
        elapsed_time = time.time() - processing_stats["start_time"]
        videos_left = processing_stats["total_videos"] - processing_stats["processed_videos"] - processing_stats["skipped_videos"]
        if videos_left > 0 and processing_stats["processed_videos"] > 0:
            avg_time_per_video = elapsed_time / processing_stats["processed_videos"]
            eta_seconds = avg_time_per_video * videos_left
            if eta_seconds < 60:
                processing_stats["eta"] = f"{eta_seconds:.1f} seconds"
            elif eta_seconds < 3600:
                processing_stats["eta"] = f"{eta_seconds/60:.1f} minutes"
            else:
                processing_stats["eta"] = f"{eta_seconds/3600:.1f} hours"
            processing_stats["average_speed"] = f"{avg_time_per_video:.1f} seconds/video"

def print_stats():
    """Print current processing statistics to console"""
    print(f"\n--- Processing Statistics ---")
    print(f"Total videos: {processing_stats['total_videos']}")
    print(f"Processed: {processing_stats['processed_videos']}")
    print(f"Skipped: {processing_stats['skipped_videos']}")
    print(f"Failed: {processing_stats['failed_videos']}")
    print(f"Current video: {processing_stats['current_video']}")
    print(f"ETA: {processing_stats['eta']}")
    print(f"Average speed: {processing_stats['average_speed']}")
    print(f"---------------------------\n")

def reset_processing_stats():
    """Reset processing statistics"""
    global processing_stats
    processing_stats = {
        "total_videos": 0,
        "processed_videos": 0,
        "skipped_videos": 0,
        "failed_videos": 0,
        "start_time": None,
        "current_video": "",
        "current_video_start_time": None,
        "eta": "N/A",
        "average_speed": "N/A"
    }

def video_has_audio(video_path):
    """
    Check if the video file has an audio stream using ffprobe.
    Returns True if an audio stream is found; otherwise, False.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=codec_type",
            "-of", "default=nw=1",
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return bool(result.stdout.strip())
    except Exception as e:
        print(f"Error checking for audio in {video_path}: {e}")
        return False

# RIFE Video FPS Increase Functions
def increase_fps_single(video_path, output_path=None, multiplier="2x", fp16=True, uhd=False, scale=1.0, 
                        skip_static=False, ffmpeg_gpu=True, ffmpeg_preset="medium", ffmpeg_crf=18, 
                        save_gen_params=True):
    """Process a single video with RIFE to increase FPS"""
    global current_loaded_model_type
    if current_loaded_model_type != "rife":
        if current_loaded_model_type == "upscale":
            print("[CMD] Unloading upscale model to save VRAM")
            torch.cuda.empty_cache()
        current_loaded_model_type = "rife"
    if output_path is None:
        output_path = get_next_output_filename()
    print(f"[CMD] Processing video: {video_path}")
    print(f"[CMD] Output will be saved to: {output_path}")
    if not video_has_audio(video_path):
        print("[CMD] No audio track found in video. Skipping audio merge.")
        skip_audio = True
    else:
        skip_audio = False
    if multiplier == "2x":
        multi_val = 2
    elif multiplier == "4x":
        multi_val = 4
    else:
        multi_val = int(multiplier.replace("x", ""))
    model_dir = os.path.abspath(os.path.join("Practical-RIFE", "train_log"))
    rife_cmd = [
        f'"{sys.executable}" "Practical-RIFE/inference_video.py"',
        f'--video "{video_path}"',
        f'--output "{output_path}"',
        f'--model "{model_dir}"',
        f'--multi {multi_val}'
    ]
    if fp16:
        rife_cmd.append("--fp16")
    if uhd:
        rife_cmd.append("--UHD")
    if scale != 1.0:
        rife_cmd.append(f"--scale {scale}")
    if skip_static:
        rife_cmd.append("--skip")
    if skip_audio:
        rife_cmd.append("--no-audio")
    start_time = time.time()
    cmd = " ".join(rife_cmd)
    print(f"[CMD] Executing: {cmd}")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if stdout:
        print(stdout)
    return_code = process.returncode
    end_time = time.time()
    if return_code != 0:
        print(f"[CMD] Error processing video: {stderr}")
        return None, f"Error processing video: {stderr}"
    print(f"[CMD] Video processing completed in {end_time - start_time:.2f} seconds")
    if save_gen_params:
        params = {
            "General": {
                "Input Video": video_path,
                "Output Video": output_path,
                "Processing Time": f"{end_time - start_time:.2f} seconds"
            },
            "RIFE Settings": {
                "Multiplier": multiplier,
                "FP16": fp16,
                "UHD": uhd,
                "Scale": scale,
                "Skip Static Frames": skip_static
            },
            "FFMPEG Settings": {
                "GPU Acceleration": ffmpeg_gpu,
                "Preset": ffmpeg_preset,
                "CRF": ffmpeg_crf
            }
        }
        save_params_file(params, output_path)
    return output_path, "Video processing completed successfully"

def batch_process_fps_increase(input_folder, output_folder, multiplier="2x", fp16=True, uhd=False, 
                              scale=1.0, skip_static=False, ffmpeg_gpu=True, ffmpeg_preset="medium", 
                              ffmpeg_crf=18, skip_existing=True, include_subfolders=True, 
                              save_gen_params=True):
    global cancel_batch_processing
    cancel_batch_processing = False
    reset_processing_stats()
    update_processing_stats("start_time", time.time())
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    if include_subfolders:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, '**', f'*{ext}'), recursive=True))
    else:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
    update_processing_stats("total_videos", len(video_files))
    print(f"[CMD] Found {len(video_files)} videos to process")
    for video_file in video_files:
        if cancel_batch_processing:
            print("[CMD] Batch processing cancelled by user")
            return "Batch processing cancelled by user"
        rel_path = os.path.relpath(video_file, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        update_processing_stats("current_video", rel_path)
        update_processing_stats("current_video_start_time", time.time())
        print_stats()
        if skip_existing and os.path.exists(output_path):
            print(f"[CMD] Skipping {rel_path} as output already exists")
            update_processing_stats("skipped_videos", processing_stats["skipped_videos"] + 1)
            continue
        try:
            print(f"[CMD] Processing {rel_path}")
            result, message = increase_fps_single(
                video_file, output_path, multiplier, fp16, uhd, scale, skip_static,
                ffmpeg_gpu, ffmpeg_preset, ffmpeg_crf, save_gen_params
            )
            if result:
                update_processing_stats("processed_videos", processing_stats["processed_videos"] + 1)
                print(f"[CMD] Successfully processed {rel_path}")
            else:
                update_processing_stats("failed_videos", processing_stats["failed_videos"] + 1)
                print(f"[CMD] Failed to process {rel_path}: {message}")
        except Exception as e:
            update_processing_stats("failed_videos", processing_stats["failed_videos"] + 1)
            print(f"[CMD] Error processing {rel_path}: {str(e)}")
    print("[CMD] Batch processing completed")
    print_stats()
    return "Batch processing completed"

# Video Upscaling Functions
def upscale_video_single(video_path, output_path=None, model_name="RealESRGAN_x4plus", outscale=4,
                         tile=0, tile_pad=10, pre_pad=0, face_enhance=False, fp32=False,
                         denoise_strength=0.5, ffmpeg_gpu=True, ffmpeg_preset="medium", 
                         ffmpeg_crf=18, save_gen_params=True):
    global current_loaded_model_type, current_loaded_upscale_model
    if current_loaded_model_type != "upscale" or current_loaded_upscale_model != model_name:
        if current_loaded_model_type == "rife":
            print("[CMD] Unloading RIFE model to save VRAM")
            torch.cuda.empty_cache()
        current_loaded_model_type = "upscale"
        current_loaded_upscale_model = model_name
    if output_path is None:
        output_path = get_next_output_filename()
    print(f"[CMD] Processing video: {video_path}")
    print(f"[CMD] Output will be saved to: {output_path}")
    is_dat_model = "dat" in model_name.lower()
    if is_dat_model:
        print(f"[CMD] Using DAT model: {model_name}")
        is_dat_model = False
    if not is_dat_model:
        print(f"[CMD] Using Real-ESRGAN model: {model_name}")
        realesrgan_cmd = [
            f'"{sys.executable}" "{os.path.join("Real-ESRGAN", "inference_realesrgan_video.py")}"',
            f'-i "{video_path}"',
            f'-o "{output_path}"',
            f'-n {model_name}',
            f'-s {outscale}'
        ]
        if tile > 0:
            realesrgan_cmd.append(f"--tile {tile}")
        realesrgan_cmd.append(f"--tile_pad {tile_pad}")
        if pre_pad > 0:
            realesrgan_cmd.append(f"--pre_pad {pre_pad}")
        if face_enhance:
            realesrgan_cmd.append("--face_enhance")
        if fp32:
            realesrgan_cmd.append("--fp32")
        if denoise_strength != 1.0:
            realesrgan_cmd.append(f"--denoise_strength {denoise_strength}")
        start_time = time.time()
        cmd = " ".join(realesrgan_cmd)
        print(f"[CMD] Executing: {cmd}")
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        if stdout:
            print(stdout)
        return_code = process.returncode
        end_time = time.time()
        if return_code != 0:
            print(f"[CMD] Error processing video: {stderr}")
            return None, f"Error processing video: {stderr}"
        print(f"[CMD] Video upscaling completed in {end_time - start_time:.2f} seconds")
    if save_gen_params:
        params = {
            "General": {
                "Input Video": video_path,
                "Output Video": output_path,
                "Processing Time": f"{end_time - start_time:.2f} seconds"
            },
            "Upscale Settings": {
                "Model": model_name,
                "Outscale": outscale,
                "Tile": tile,
                "Tile Pad": tile_pad,
                "Pre Pad": pre_pad,
                "Face Enhance": face_enhance,
                "FP32": fp32,
                "Denoise Strength": denoise_strength
            },
            "FFMPEG Settings": {
                "GPU Acceleration": ffmpeg_gpu,
                "Preset": ffmpeg_preset,
                "CRF": ffmpeg_crf
            }
        }
        save_params_file(params, output_path)
    return output_path, "Video upscaling completed successfully"

def batch_process_upscale(input_folder, output_folder, model_name="RealESRGAN_x4plus", outscale=4,
                         tile=0, tile_pad=10, pre_pad=0, face_enhance=False, fp32=False,
                         denoise_strength=0.5, ffmpeg_gpu=True, ffmpeg_preset="medium", 
                         ffmpeg_crf=18, skip_existing=True, include_subfolders=True, 
                         save_gen_params=True):
    global cancel_batch_processing
    cancel_batch_processing = False
    reset_processing_stats()
    update_processing_stats("start_time", time.time())
    os.makedirs(output_folder, exist_ok=True)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
    video_files = []
    if include_subfolders:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, '**', f'*{ext}'), recursive=True))
    else:
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(input_folder, f'*{ext}')))
    update_processing_stats("total_videos", len(video_files))
    print(f"[CMD] Found {len(video_files)} videos to process")
    for video_file in video_files:
        if cancel_batch_processing:
            print("[CMD] Batch processing cancelled by user")
            return "Batch processing cancelled by user"
        rel_path = os.path.relpath(video_file, input_folder)
        output_path = os.path.join(output_folder, rel_path)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        update_processing_stats("current_video", rel_path)
        update_processing_stats("current_video_start_time", time.time())
        print_stats()
        if skip_existing and os.path.exists(output_path):
            print(f"[CMD] Skipping {rel_path} as output already exists")
            update_processing_stats("skipped_videos", processing_stats["skipped_videos"] + 1)
            continue
        try:
            print(f"[CMD] Processing {rel_path}")
            result, message = upscale_video_single(
                video_file, output_path, model_name, outscale, tile, tile_pad, pre_pad,
                face_enhance, fp32, denoise_strength, ffmpeg_gpu, ffmpeg_preset, ffmpeg_crf,
                save_gen_params
            )
            if result:
                update_processing_stats("processed_videos", processing_stats["processed_videos"] + 1)
                print(f"[CMD] Successfully processed {rel_path}")
            else:
                update_processing_stats("failed_videos", processing_stats["failed_videos"] + 1)
                print(f"[CMD] Failed to process {rel_path}: {message}")
        except Exception as e:
            update_processing_stats("failed_videos", processing_stats["failed_videos"] + 1)
            print(f"[CMD] Error processing {rel_path}: {str(e)}")
    print("[CMD] Batch processing completed")
    print_stats()
    return "Batch processing completed"

def cancel_batch():
    global cancel_batch_processing
    cancel_batch_processing = True
    return "Cancelling batch processing... Please wait for current video to finish."

def open_outputs_folder():
    if platform.system() == "Windows":
        os.startfile(OUTPUTS_DIR)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", OUTPUTS_DIR])
    else:
        subprocess.Popen(["xdg-open", OUTPUTS_DIR])
    return "Opened outputs folder"

def save_current_config(config_name, 
                       # General settings
                       ffmpeg_gpu, ffmpeg_preset, ffmpeg_crf,
                       # RIFE settings
                       rife_model_dir, rife_multiplier, rife_fp16, rife_uhd, rife_scale, rife_skip_static,
                       # Upscale settings
                       upscale_model, upscale_preset, upscale_outscale, upscale_tile, upscale_tile_pad,
                       upscale_pre_pad, upscale_face_enhance, upscale_fp32, upscale_denoise_strength,
                       # Batch settings
                       batch_input_folder, batch_output_folder, skip_existing, include_subfolders, save_gen_params):
    config_data = {
        "ffmpeg_gpu": ffmpeg_gpu,
        "ffmpeg_preset": ffmpeg_preset,
        "ffmpeg_crf": ffmpeg_crf,
        "rife_model_dir": rife_model_dir,
        "rife_multiplier": rife_multiplier,
        "rife_fp16": rife_fp16,
        "rife_uhd": rife_uhd,
        "rife_scale": rife_scale,
        "rife_skip_static": rife_skip_static,
        "upscale_model": upscale_model,
        "upscale_preset": upscale_preset,
        "upscale_outscale": upscale_outscale,
        "upscale_tile": upscale_tile,
        "upscale_tile_pad": upscale_tile_pad,
        "upscale_pre_pad": upscale_pre_pad,
        "upscale_face_enhance": upscale_face_enhance,
        "upscale_fp32": upscale_fp32,
        "upscale_denoise_strength": upscale_denoise_strength,
        "batch_input_folder": batch_input_folder,
        "batch_output_folder": batch_output_folder,
        "skip_existing": skip_existing,
        "include_subfolders": include_subfolders,
        "save_gen_params": save_gen_params
    }
    saved_name = save_config(config_name, config_data)
    return f"Configuration saved as {saved_name}", get_config_list(), saved_name

def load_saved_config(config_name):
    config_data = load_config(config_name)
    return (
        # General settings
        config_data.get("ffmpeg_gpu", True),
        config_data.get("ffmpeg_preset", "medium"),
        config_data.get("ffmpeg_crf", 18),
        # RIFE settings
        config_data.get("rife_model_dir", os.path.abspath(os.path.join("Practical-RIFE", "train_log"))),
        config_data.get("rife_multiplier", "2x"),
        config_data.get("rife_fp16", True),
        config_data.get("rife_uhd", False),
        config_data.get("rife_scale", 1.0),
        config_data.get("rife_skip_static", False),
        # Upscale settings
        config_data.get("upscale_model", "RealESRGAN_x4plus"),
        config_data.get("upscale_preset", "general"),
        config_data.get("upscale_outscale", 4),
        config_data.get("upscale_tile", 0),
        config_data.get("upscale_tile_pad", 10),
        config_data.get("upscale_pre_pad", 0),
        config_data.get("upscale_face_enhance", False),
        config_data.get("upscale_fp32", False),
        config_data.get("upscale_denoise_strength", 0.5),
        # Batch settings
        config_data.get("batch_input_folder", ""),
        config_data.get("batch_output_folder", "outputs"),
        config_data.get("skip_existing", True),
        config_data.get("include_subfolders", True),
        config_data.get("save_gen_params", True),
        f"Loaded configuration: {config_name}"
    )

def update_upscale_model_by_preset(preset):
    if preset == "anime":
        return "RealESRGAN_x4plus_anime_6B"
    elif preset == "general":
        return "RealESRGAN_x4plus"
    elif preset == "realism":
        return "4xRealWebPhoto_v4_dat2"
    return "RealESRGAN_x4plus"

def get_available_upscale_models():
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "upscale_models")
    os.makedirs(models_dir, exist_ok=True)
    default_models = [
        "RealESRGAN_x4plus",
        "RealESRGAN_x4plus_anime_6B",
        "4xRealWebPhoto_v4_dat2"
    ]
    model_files = []
    for ext in ['.pth', '.safetensors', '.onnx']:
        model_files.extend(glob.glob(os.path.join(models_dir, f'*{ext}')))
    additional_models = [os.path.splitext(os.path.basename(f))[0] for f in model_files]
    all_models = list(set(default_models + additional_models))
    return sorted(all_models)

def create_ui():
    with gr.Blocks(title="Video Enhancer") as demo:
        gr.Markdown("# Video Enhancer")
        gr.Markdown("Enhance your videos with FPS increase and upscaling")
        with gr.Row():
            with gr.Column(scale=1):
                saved_configs = gr.Dropdown(
                    label="Saved Configs",
                    choices=get_config_list(),
                    value=last_config if last_config in get_config_list() else None
                )
            with gr.Column(scale=2):
                config_name_input = gr.Textbox(
                    label="Config Name",
                    placeholder="Enter config name"
                )
            with gr.Column(scale=1):
                save_config_btn = gr.Button("Save Config")
            with gr.Column(scale=1):
                load_config_btn = gr.Button("Load Config")
        config_status = gr.Textbox(label="Config Status", interactive=False)
        with gr.Tabs() as tabs:
            with gr.TabItem("Video FPS Increase"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input Video")
                        input_video = gr.Video(label="Input Video")
                        with gr.Accordion("RIFE Settings", open=True):
                            rife_multiplier = gr.Radio(
                                label="FPS Multiplier",
                                choices=["2x", "4x", "8x"],
                                value=config_loaded.get("rife_multiplier", "2x")
                            )
                            with gr.Row():
                                rife_fp16 = gr.Checkbox(
                                    label="Use FP16 (faster on modern GPUs)",
                                    value=config_loaded.get("rife_fp16", True)
                                )
                                rife_uhd = gr.Checkbox(
                                    label="UHD Mode (for 4K videos)",
                                    value=config_loaded.get("rife_uhd", False)
                                )
                            with gr.Row():
                                rife_scale = gr.Slider(
                                    label="Scale Factor",
                                    minimum=0.25,
                                    maximum=4.0,
                                    step=0.25,
                                    value=config_loaded.get("rife_scale", 1.0)
                                )
                                rife_skip_static = gr.Checkbox(
                                    label="Skip Static Frames",
                                    value=config_loaded.get("rife_skip_static", False)
                                )
                        with gr.Accordion("FFMPEG Settings", open=False):
                            with gr.Row():
                                ffmpeg_gpu = gr.Checkbox(
                                    label="Use GPU Acceleration",
                                    value=config_loaded.get("ffmpeg_gpu", True)
                                )
                            with gr.Row():
                                ffmpeg_preset = gr.Dropdown(
                                    label="Encoding Preset",
                                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                                    value=config_loaded.get("ffmpeg_preset", "medium")
                                )
                                ffmpeg_crf = gr.Slider(
                                    label="CRF (Quality, lower is better)",
                                    minimum=0,
                                    maximum=51,
                                    step=1,
                                    value=config_loaded.get("ffmpeg_crf", 18)
                                )
                        save_gen_params = gr.Checkbox(
                            label="Save Generation Parameters",
                            value=config_loaded.get("save_gen_params", True)
                        )
                        process_btn = gr.Button("Process Video", variant="primary")
                    with gr.Column():
                        gr.Markdown("### Output Video")
                        output_video = gr.Video(label="Output Video")
                        output_status = gr.Textbox(label="Status", interactive=False)
                        open_outputs_btn = gr.Button("Open Outputs Folder")
                with gr.Accordion("Batch Processing", open=False):
                    with gr.Row():
                        batch_input_folder = gr.Textbox(
                            label="Input Folder",
                            placeholder="Path to folder with videos",
                            value=config_loaded.get("batch_input_folder", "")
                        )
                        batch_output_folder = gr.Textbox(
                            label="Output Folder",
                            placeholder="Path to output folder",
                            value=config_loaded.get("batch_output_folder", "outputs")
                        )
                    with gr.Row():
                        skip_existing = gr.Checkbox(
                            label="Skip Existing Files",
                            value=config_loaded.get("skip_existing", True)
                        )
                        include_subfolders = gr.Checkbox(
                            label="Include Subfolders",
                            value=config_loaded.get("include_subfolders", True)
                        )
                    with gr.Row():
                        batch_process_btn = gr.Button("Start Batch Processing", variant="primary")
                        cancel_batch_btn = gr.Button("Cancel Batch Processing", variant="stop")
                    batch_status = gr.Textbox(label="Batch Status", interactive=False)
            with gr.TabItem("Video Upscaling"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Input Video")
                        upscale_input_video = gr.Video(label="Input Video")
                        with gr.Accordion("Upscale Settings", open=True):
                            upscale_preset = gr.Radio(
                                label="Preset",
                                choices=["general", "anime", "realism"],
                                value=config_loaded.get("upscale_preset", "general")
                            )
                            upscale_model = gr.Dropdown(
                                label="Model",
                                choices=get_available_upscale_models(),
                                value=config_loaded.get("upscale_model", "RealESRGAN_x4plus")
                            )
                            upscale_outscale = gr.Slider(
                                label="Output Scale",
                                minimum=1,
                                maximum=8,
                                step=1,
                                value=config_loaded.get("upscale_outscale", 4)
                            )
                            with gr.Accordion("Advanced Settings", open=False):
                                upscale_tile = gr.Slider(
                                    label="Tile Size (0 for auto)",
                                    minimum=0,
                                    maximum=1024,
                                    step=32,
                                    value=config_loaded.get("upscale_tile", 0)
                                )
                                upscale_tile_pad = gr.Slider(
                                    label="Tile Padding",
                                    minimum=0,
                                    maximum=32,
                                    step=4,
                                    value=config_loaded.get("upscale_tile_pad", 10)
                                )
                                upscale_pre_pad = gr.Slider(
                                    label="Pre Padding",
                                    minimum=0,
                                    maximum=32,
                                    step=4,
                                    value=config_loaded.get("upscale_pre_pad", 0)
                                )
                                with gr.Row():
                                    upscale_face_enhance = gr.Checkbox(
                                        label="Face Enhancement",
                                        value=config_loaded.get("upscale_face_enhance", False)
                                    )
                                    upscale_fp32 = gr.Checkbox(
                                        label="Use FP32 (more accurate but slower)",
                                        value=config_loaded.get("upscale_fp32", False)
                                    )
                                upscale_denoise_strength = gr.Slider(
                                    label="Denoise Strength",
                                    minimum=0,
                                    maximum=1,
                                    step=0.05,
                                    value=config_loaded.get("upscale_denoise_strength", 0.5)
                                )
                        with gr.Accordion("FFMPEG Settings", open=False):
                            with gr.Row():
                                upscale_ffmpeg_gpu = gr.Checkbox(
                                    label="Use GPU Acceleration",
                                    value=config_loaded.get("ffmpeg_gpu", True)
                                )
                            with gr.Row():
                                upscale_ffmpeg_preset = gr.Dropdown(
                                    label="Encoding Preset",
                                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                                    value=config_loaded.get("ffmpeg_preset", "medium")
                                )
                                upscale_ffmpeg_crf = gr.Slider(
                                    label="CRF (Quality, lower is better)",
                                    minimum=0,
                                    maximum=51,
                                    step=1,
                                    value=config_loaded.get("ffmpeg_crf", 18)
                                )
                        upscale_save_gen_params = gr.Checkbox(
                            label="Save Generation Parameters",
                            value=config_loaded.get("save_gen_params", True)
                        )
                        upscale_process_btn = gr.Button("Process Video", variant="primary")
                    with gr.Column():
                        gr.Markdown("### Output Video")
                        upscale_output_video = gr.Video(label="Output Video")
                        upscale_output_status = gr.Textbox(label="Status", interactive=False)
                        upscale_open_outputs_btn = gr.Button("Open Outputs Folder")
                with gr.Accordion("Batch Processing", open=False):
                    with gr.Row():
                        upscale_batch_input_folder = gr.Textbox(
                            label="Input Folder",
                            placeholder="Path to folder with videos",
                            value=config_loaded.get("batch_input_folder", "")
                        )
                        upscale_batch_output_folder = gr.Textbox(
                            label="Output Folder",
                            placeholder="Path to output folder",
                            value=config_loaded.get("batch_output_folder", "outputs")
                        )
                    with gr.Row():
                        upscale_skip_existing = gr.Checkbox(
                            label="Skip Existing Files",
                            value=config_loaded.get("skip_existing", True)
                        )
                        upscale_include_subfolders = gr.Checkbox(
                            label="Include Subfolders",
                            value=config_loaded.get("include_subfolders", True)
                        )
                    with gr.Row():
                        upscale_batch_process_btn = gr.Button("Start Batch Processing", variant="primary")
                        upscale_cancel_batch_btn = gr.Button("Cancel Batch Processing", variant="stop")
                    upscale_batch_status = gr.Textbox(label="Batch Status", interactive=False)
        save_config_btn.click(
            fn=save_current_config,
            inputs=[
                config_name_input,
                ffmpeg_gpu, ffmpeg_preset, ffmpeg_crf,
                gr.State(os.path.abspath(os.path.join("Practical-RIFE", "train_log"))),
                rife_multiplier, rife_fp16, rife_uhd, rife_scale, rife_skip_static,
                upscale_model, upscale_preset, upscale_outscale, upscale_tile, upscale_tile_pad,
                upscale_pre_pad, upscale_face_enhance, upscale_fp32, upscale_denoise_strength,
                batch_input_folder, batch_output_folder, skip_existing, include_subfolders, save_gen_params
            ],
            outputs=[config_status, saved_configs, config_name_input]
        )
        load_config_btn.click(
            fn=load_saved_config,
            inputs=[saved_configs],
            outputs=[
                ffmpeg_gpu, ffmpeg_preset, ffmpeg_crf,
                gr.State(os.path.abspath(os.path.join("Practical-RIFE", "train_log"))),
                rife_multiplier, rife_fp16, rife_uhd, rife_scale, rife_skip_static,
                upscale_model, upscale_preset, upscale_outscale, upscale_tile, upscale_tile_pad,
                upscale_pre_pad, upscale_face_enhance, upscale_fp32, upscale_denoise_strength,
                batch_input_folder, batch_output_folder, skip_existing, include_subfolders, save_gen_params,
                config_status
            ]
        )
        process_btn.click(
            fn=increase_fps_single,
            inputs=[
                input_video,
                gr.State(None),
                rife_multiplier,
                rife_fp16,
                rife_uhd,
                rife_scale,
                rife_skip_static,
                ffmpeg_gpu,
                ffmpeg_preset,
                ffmpeg_crf,
                save_gen_params
            ],
            outputs=[output_video, output_status]
        )
        open_outputs_btn.click(
            fn=open_outputs_folder,
            inputs=[],
            outputs=[output_status]
        )
        batch_process_btn.click(
            fn=batch_process_fps_increase,
            inputs=[
                batch_input_folder,
                batch_output_folder,
                rife_multiplier,
                rife_fp16,
                rife_uhd,
                rife_scale,
                rife_skip_static,
                ffmpeg_gpu,
                ffmpeg_preset,
                ffmpeg_crf,
                skip_existing,
                include_subfolders,
                save_gen_params
            ],
            outputs=[batch_status]
        )
        cancel_batch_btn.click(
            fn=cancel_batch,
            inputs=[],
            outputs=[batch_status]
        )
        upscale_preset.change(
            fn=update_upscale_model_by_preset,
            inputs=[upscale_preset],
            outputs=[upscale_model]
        )
        upscale_process_btn.click(
            fn=upscale_video_single,
            inputs=[
                upscale_input_video,
                gr.State(None),
                upscale_model,
                upscale_outscale,
                upscale_tile,
                upscale_tile_pad,
                upscale_pre_pad,
                upscale_face_enhance,
                upscale_fp32,
                upscale_denoise_strength,
                upscale_ffmpeg_gpu,
                upscale_ffmpeg_preset,
                upscale_ffmpeg_crf,
                upscale_save_gen_params
            ],
            outputs=[upscale_output_video, upscale_output_status]
        )
        upscale_open_outputs_btn.click(
            fn=open_outputs_folder,
            inputs=[],
            outputs=[upscale_output_status]
        )
        upscale_batch_process_btn.click(
            fn=batch_process_upscale,
            inputs=[
                upscale_batch_input_folder,
                upscale_batch_output_folder,
                upscale_model,
                upscale_outscale,
                upscale_tile,
                upscale_tile_pad,
                upscale_pre_pad,
                upscale_face_enhance,
                upscale_fp32,
                upscale_denoise_strength,
                upscale_ffmpeg_gpu,
                upscale_ffmpeg_preset,
                upscale_ffmpeg_crf,
                upscale_skip_existing,
                upscale_include_subfolders,
                upscale_save_gen_params
            ],
            outputs=[upscale_batch_status]
        )
        upscale_cancel_batch_btn.click(
            fn=cancel_batch,
            inputs=[],
            outputs=[upscale_batch_status]
        )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Enhancer App")
    parser.add_argument("--share", action="store_true", help="Share the Gradio app")
    args = parser.parse_args()
    demo = create_ui()
    demo.launch(share=args.share, inbrowser=True)