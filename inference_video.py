#!/usr/bin/env python
# inference_video.py

import os
import cv2
import torch
import argparse
import numpy as np
import time
import subprocess  # For ffmpeg process handling
import warnings
import _thread
import skvideo.io
import sys
import re
from queue import Queue, Empty
from torch.nn import functional as F

# Force immediate console output (no buffering)
sys.stdout.reconfigure(line_buffering=True)

# Fix potential numpy attribute issues.
if not hasattr(np, 'float'):
    np.float = float
if not hasattr(np, 'int'):
    np.int = int

from model.pytorch_msssim import ssim_matlab

warnings.filterwarnings("ignore")

# =============================================================================
# GLOBAL VARIABLES
# =============================================================================
cancel_increase = False
# New variable to control ffmpeg output filtering
show_ffmpeg_output = False  # Set to True to see ffmpeg output

# =============================================================================
# CANCEL PROCESS LISTENER
# =============================================================================
# Listen for keystroke "c" to cancel the interpolation process.
try:
    import msvcrt  # Windows-only module for non-blocking keyboard input
except ImportError:
    msvcrt = None

def listen_for_cancel():
    """
    Listen for a key press to cancel the FPS increase process.
    When the user presses 'c' (or 'C'), set the cancel_increase flag.
    """
    global cancel_increase
    print("Press 'c' at any time to cancel the FPS increase process.")
    try:
        if msvcrt:
            # Non-blocking key detection for Windows.
            while True:
                if msvcrt.kbhit():
                    key = msvcrt.getch().decode('utf-8')
                    if key.lower() == 'c':
                        cancel_increase = True
                        sys.stdout.write("\nCancellation requested. Interpolation will be skipped.\n")
                        sys.stdout.flush()
                        break
                time.sleep(0.1)
        else:
            # Blocking input for non-Windows platforms.
            while True:
                key = input()
                if key.lower() == 'c':
                    cancel_increase = True
                    print("Cancellation requested. Interpolation will be skipped.")
                    break
    except Exception as e:
        print("Error in cancellation listener thread:", e)

# =============================================================================
# Parse command line arguments
# =============================================================================
parser = argparse.ArgumentParser(description='Interpolation for a pair of images')
parser.add_argument('--video', dest='video', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default=None)
parser.add_argument('--img', dest='img', type=str, default=None)
parser.add_argument('--montage', dest='montage', action='store_true', help='montage origin video')
parser.add_argument('--model', dest='modelDir', type=str, default='train_log', help='directory with trained model files')
parser.add_argument('--fp16', dest='fp16', action='store_true', help='fp16 mode for faster and more lightweight inference on cards with Tensor Cores')
parser.add_argument('--UHD', dest='UHD', action='store_true', help='support 4k video')
parser.add_argument('--scale', dest='scale', type=float, default=1.0, help='Try scale=0.5 for 4k video')
parser.add_argument('--skip', dest='skip', action='store_true', help='whether to remove static frames before processing')
parser.add_argument('--fps', dest='fps', type=int, default=None)
parser.add_argument('--png', dest='png', action='store_true', help='whether to output png format frames')
parser.add_argument('--ext', dest='ext', type=str, default='mp4', help='output video extension')
parser.add_argument('--exp', dest='exp', type=int, default=1)
parser.add_argument('--multi', dest='multi', type=int, default=2)
# New argument to skip audio transfer if desired.
parser.add_argument('--no-audio', dest='no_audio', action='store_true', help='Skip audio transfer even if source video has audio')
# New argument to show ffmpeg output
parser.add_argument('--show-ffmpeg', dest='show_ffmpeg', action='store_true', help='Show ffmpeg encoding output')

args = parser.parse_args()
if args.exp != 1:
    args.multi = (2 ** args.exp)
assert (args.video is not None or args.img is not None), "Either --video or --img must be provided."

if args.skip:
    print("skip flag is abandoned, please refer to issue #207.")
if args.UHD and args.scale == 1.0:
    args.scale = 0.5
assert args.scale in [0.25, 0.5, 1.0, 2.0, 4.0]
if args.img is not None:
    args.png = True

# Set ffmpeg output flag
show_ffmpeg_output = args.show_ffmpeg

# =============================================================================
# Setup device and model
# =============================================================================
print("Initializing RIFE model...", end="")
sys.stdout.flush()  # Force immediate output
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    if args.fp16:
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

from train_log.RIFE_HDv3 import Model

model_load_start = time.time()
model = Model()
if not hasattr(model, 'version'):
    model.version = 0
model.load_model(args.modelDir, -1)
model_load_time = time.time() - model_load_start
print(f" done in {model_load_time:.2f}s")
sys.stdout.flush()  # Force immediate output
model.eval()
model.device()

# =============================================================================
# Start cancellation listener in a separate thread
# =============================================================================
import threading
cancel_thread = threading.Thread(target=listen_for_cancel, daemon=True)
cancel_thread.start()

# =============================================================================
# Prepare video/image input/output
# =============================================================================
if args.video is not None:
    print("\nReading video metadata...", end="")
    sys.stdout.flush()  # Force immediate output
    video_open_start = time.time()
    videoCapture = cv2.VideoCapture(args.video)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    tot_frame = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    videoCapture.release()
    video_open_time = time.time() - video_open_start
    print(f" done in {video_open_time:.2f}s")
    
    if args.fps is None:
        fpsNotAssigned = True
        args.fps = fps * args.multi
        print(f"[INFO] Input: {int(tot_frame)} frames at {fps:.2f}fps | Output: {args.fps:.2f}fps ({args.multi}x)")
    else:
        fpsNotAssigned = False
        print(f"[INFO] Input: {int(tot_frame)} frames at {fps:.2f}fps | Output: {args.fps:.2f}fps (user-specified)")
        
    print("Setting up video reader...", end="")
    reader_start_time = time.time()
    videogen = skvideo.io.vreader(args.video)
    
    first_frame_start = time.time()
    try:
        lastframe = next(videogen)
        first_frame_time = time.time() - first_frame_start
        print(f" done in {first_frame_time:.2f}s")
    except Exception as e:
        print(f"\nError: Failed to load first frame: {str(e)}")
        sys.exit(1)
        
    video_path_wo_ext, ext = os.path.splitext(args.video)
    
    if not args.png and fpsNotAssigned and not args.no_audio:
        print("[INFO] Audio will be transferred from source to output")
    else:
        print("[INFO] Audio transfer will be skipped")
else:
    videogen = []
    for f in os.listdir(args.img):
        if 'png' in f:
            videogen.append(f)
    tot_frame = len(videogen)
    videogen.sort(key=lambda x: int(x[:-4]))
    lastframe = cv2.imread(os.path.join(args.img, videogen[0]), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
    videogen = videogen[1:]

h, w, _ = lastframe.shape
vid_out_name = None
ffmpeg_process = None  # This will be used for video saving

# =============================================================================
# Define write and read buffer functions for multithreaded I/O
# =============================================================================
def clear_write_buffer(user_args, write_buffer, ffmpeg_proc=None):
    cnt = 0
    while True:
        item = write_buffer.get()
        if item is None:
            break
        if user_args.png:
            cv2.imwrite('vid_out/{:0>7d}.png'.format(cnt), item[:, :, ::-1])
            cnt += 1
        else:
            # Convert from RGB to BGR and write raw bytes to ffmpeg.
            frame_bgr = item[:, :, ::-1]
            ffmpeg_proc.stdin.write(frame_bgr.tobytes())
            ffmpeg_proc.stdin.flush()  # Ensure data is sent immediately

def build_read_buffer(user_args, read_buffer, videogen):
    try:
        buffer_count = 0
        loading_start_time = time.time()
        
        # Display initial loading message
        print("Pre-loading frames into buffer...", end="")
        sys.stdout.flush()
        
        for frame in videogen:
            if user_args.img is not None:
                frame = cv2.imread(os.path.join(user_args.img, frame), cv2.IMREAD_UNCHANGED)[:, :, ::-1].copy()
            if user_args.montage:
                frame = frame[:, left: left + w]
            
            read_buffer.put(frame)
            buffer_count += 1
            
            # Only update status occasionally to avoid console spam
            if buffer_count % 500 == 0:
                elapsed = time.time() - loading_start_time
                fps = buffer_count / elapsed if elapsed > 0 else 0
                print(f"\rPre-loading: {buffer_count}/{int(tot_frame)} frames ({buffer_count/tot_frame*100:.1f}%) at {fps:.1f} fps", end="")
                sys.stdout.flush()
    except Exception as e:
        print(f"\rError loading frames: {str(e)}")
    
    # Final loading status
    total_time = time.time() - loading_start_time
    buffer_fps = buffer_count/total_time if total_time > 0 else 0
    print(f"\rPre-loading: Complete - {buffer_count} frames loaded in {total_time:.2f}s ({buffer_fps:.1f} fps)")
    sys.stdout.flush()  # Force immediate output
    read_buffer.put(None)

# =============================================================================
# Define interpolation function
# =============================================================================
def make_inference(I0, I1, n, current_frame):
    global cancel_increase
    if model.version >= 3.9:
        res = []
        for i in range(n):
            timestep = (i + 1) / (n + 1)
            if cancel_increase:
                break
            res.append(model.inference(I0, I1, timestep, args.scale))
        return res
    else:
        middle = model.inference(I0, I1, args.scale)
        if n == 1:
            return [middle]
        first_half = make_inference(I0, middle, n=n // 2, current_frame=current_frame)
        second_half = make_inference(middle, I1, n=n // 2, current_frame=current_frame)
        if n % 2:
            return [*first_half, middle, *second_half]
        else:
            return [*first_half, *second_half]

# =============================================================================
# Prepare output video settings
# =============================================================================
if args.png:
    if not os.path.exists('vid_out'):
        os.mkdir('vid_out')
    print("[INFO] Output format: PNG sequence in 'vid_out' directory")
else:
    print("Setting up FFmpeg encoder...", end="")
    sys.stdout.flush()
    if args.output is not None:
        vid_out_name = args.output
    else:
        vid_out_name = '{}_{}X_{}fps.{}'.format(video_path_wo_ext, args.multi, int(np.round(args.fps)), args.ext)
        
    print(f" {os.path.basename(vid_out_name)} [{w}x{h}@{int(np.round(args.fps))}fps]")
    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f'{w}x{h}',
        '-r', str(int(np.round(args.fps))),
        '-i', '-',  # input from pipe
        '-an',
        '-vcodec', 'libx264',
        '-profile:v', 'high',
        '-level', '3.1',
        '-preset', 'veryslow',
        '-crf', '12',
        '-pix_fmt', 'yuv420p',
        '-x264-params', 'ref=4:cabac=1',
        vid_out_name
    ]
    # Use DEVNULL for stdout/stderr if not showing FFmpeg output
    ffmpeg_stdout = None if show_ffmpeg_output else subprocess.DEVNULL
    ffmpeg_stderr = None if show_ffmpeg_output else subprocess.DEVNULL
    ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, 
                                     stdout=ffmpeg_stdout, stderr=ffmpeg_stderr)

# =============================================================================
# Audio transfer function (unchanged)
# =============================================================================
def transferAudio(sourceVideo, targetVideo):
    import shutil
    import os
    temp_dir = "./temp"
    tempAudioFileName = os.path.join(temp_dir, "audio.mkv")
    
    audio_check_cmd = f'ffprobe -loglevel error -select_streams a -show_entries stream=index -of csv=p=0 "{sourceVideo}"'
    audio_streams = os.popen(audio_check_cmd).read().strip()
    if not audio_streams:
        print("No audio stream found in source video. Skipping audio transfer.")
        return

    if os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # Hide ffmpeg output unless show_ffmpeg_output is True
    redirect = "" if show_ffmpeg_output else " > nul 2>&1" if os.name == 'nt' else " > /dev/null 2>&1"
    
    print("Extracting audio from source video...", end="")
    sys.stdout.flush()
    os.system(f'ffmpeg -y -i "{sourceVideo}" -map 0:a? -c:a copy -vn "{tempAudioFileName}"{redirect}')
    if not os.path.exists(tempAudioFileName) or os.path.getsize(tempAudioFileName) == 0:
        print(" failed!")
        return
    print(" done")

    targetNoAudio = os.path.splitext(targetVideo)[0] + "_noaudio" + os.path.splitext(targetVideo)[1]
    os.rename(targetVideo, targetNoAudio)
    
    print("Merging audio with new video...", end="")
    sys.stdout.flush()
    os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName}" -c copy "{targetVideo}"{redirect}')

    if os.path.getsize(targetVideo) == 0:
        print(" failed with copy, trying with transcode...")
        sys.stdout.flush()
        tempAudioFileName_aac = os.path.join(temp_dir, "audio.m4a")
        os.system(f'ffmpeg -y -i "{sourceVideo}" -map 0:a? -c:a aac -b:a 160k -vn "{tempAudioFileName_aac}"{redirect}')
        os.system(f'ffmpeg -y -i "{targetNoAudio}" -i "{tempAudioFileName_aac}" -c copy "{targetVideo}"{redirect}')
        if os.path.getsize(targetVideo) == 0:
            os.rename(targetNoAudio, targetVideo)
            print(" failed! Output will have no audio.")
        else:
            print(" done with AAC audio")
            os.remove(targetNoAudio)
    else:
        print(" done")
        os.remove(targetNoAudio)

    shutil.rmtree(temp_dir)

# =============================================================================
# Calculate padding to ensure dimensions are multiples of a base value.
# =============================================================================
if args.montage:
    left = w // 4
    w = w // 2
tmp = max(128, int(128 / args.scale))
ph = ((h - 1) // tmp + 1) * tmp
pw = ((w - 1) // tmp + 1) * tmp
padding = (0, pw - w, 0, ph - h)

# =============================================================================
# Setup buffers and start processing threads
# =============================================================================
print("\nSetting up processing buffers...", end="")
sys.stdout.flush()
if args.montage:
    lastframe = lastframe[:, left: left + w]

write_buffer = Queue(maxsize=500)
read_buffer = Queue(maxsize=500)

print(" done")
print("Starting reader and writer threads...")
_thread.start_new_thread(build_read_buffer, (args, read_buffer, videogen))
_thread.start_new_thread(clear_write_buffer, (args, write_buffer, None if args.png else ffmpeg_process))

print("Preparing first frame...")
I1 = torch.from_numpy(np.transpose(lastframe, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
def pad_image(img):
    if args.fp16:
        return F.pad(img, padding).half()
    else:
        return F.pad(img, padding)
I1 = pad_image(I1)
temp = None  # Temporarily holds frame when processing static scenes

# Wait a moment for buffer to start filling
time.sleep(0.5)

# =============================================================================
# Main processing loop: read frames, run interpolation if needed,
# update progress, check for cancellation, and write out results.
# =============================================================================
# Initialize frame counter and timing variables
current_frame = 0
start_time = time.time()
last_update_time = start_time
frames_since_update = 0
recent_fps_values = []

# Initialize status line
print("\nProcessing frames: ")
sys.stdout.flush()  # Force immediate output

# Function to filter ffmpeg output if not showing it
def check_ffmpeg_output():
    if not show_ffmpeg_output and ffmpeg_process and ffmpeg_process.stdout:
        try:
            # Non-blocking read from ffmpeg stdout
            line = ffmpeg_process.stdout.readline().decode('utf-8', errors='ignore').strip()
            pass
        except Exception:
            pass

while True:
    if cancel_increase:
        print("\n[INFO] Cancel button clicked. Cancelling processing immediately.")
        break
    if temp is not None:
        frame = temp
        temp = None
    else:
        try:
            frame = read_buffer.get(timeout=0.1)
        except Empty:
            continue
    if frame is None:
        break
        
    current_frame += 1
    frames_since_update += 1
    
    current_time = time.time()
    update_interval = min(1.0, max(0.2, tot_frame / 5000))
    
    if current_frame == 1 or current_frame % 100 == 0 or current_frame == tot_frame:
        elapsed = current_time - start_time
        time_per_frame = elapsed / current_frame if current_frame > 0 else 0
        
        recent_fps = frames_since_update / (current_time - last_update_time) if current_time > last_update_time else 0
        if recent_fps > 0:
            recent_fps_values.append(recent_fps)
            if len(recent_fps_values) > 5:
                recent_fps_values.pop(0)
        
        avg_recent_fps = sum(recent_fps_values) / len(recent_fps_values) if recent_fps_values else 0
        remaining_frames = tot_frame - current_frame
        eta_seconds = remaining_frames / avg_recent_fps if avg_recent_fps > 0 else 0
        
        eta_str = ""
        if eta_seconds > 0:
            if eta_seconds < 60:
                eta_str = f"{eta_seconds:.0f}s"
            elif eta_seconds < 3600:
                eta_str = f"{eta_seconds//60:.0f}m {eta_seconds%60:.0f}s"
            else:
                eta_str = f"{eta_seconds//3600:.0f}h {(eta_seconds%3600)//60:.0f}m"
        
        percent_complete = (current_frame / tot_frame) * 100 if tot_frame > 0 else 0
        
        bar_length = 30
        filled_length = int(bar_length * current_frame // tot_frame)
        bar = '=' * filled_length + '-' * (bar_length - filled_length)
        
        print(f"\r[{bar}] {current_frame}/{int(tot_frame)} ({percent_complete:.1f}%) | {avg_recent_fps:.1f} fps | ETA: {eta_str}    ", end="")
        sys.stdout.flush()
        
        check_ffmpeg_output()
        
        if current_time > last_update_time:
            last_update_time = current_time
            frames_since_update = 0
    
    I0 = I1
    I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
    I1 = pad_image(I1)
    I0_small = F.interpolate(I0, (32, 32), mode='bilinear', align_corners=False)
    I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
    ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])

    break_flag = False
    if ssim > 0.996:
        frame = read_buffer.get() 
        if frame is None:
            break_flag = True
            frame = lastframe
        else:
            temp = frame
        I1 = torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(device, non_blocking=True).unsqueeze(0).float() / 255.
        I1 = pad_image(I1)
        I1 = model.inference(I0, I1, scale=args.scale)
        I1_small = F.interpolate(I1, (32, 32), mode='bilinear', align_corners=False)
        ssim = ssim_matlab(I0_small[:, :3], I1_small[:, :3])
        frame = (I1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]

    if cancel_increase:
        print("\nFPS increase cancelled. Outputting original frames only.")
        output = []
    else:
        if ssim < 0.2:
            output = []
            for i in range(args.multi - 1):
                output.append(I0)
        else:
            output = make_inference(I0, I1, args.multi - 1, current_frame)

    if args.montage:
        write_buffer.put(np.concatenate((lastframe, lastframe), 1))
        for mid in output:
            mid_np = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))
            write_buffer.put(np.concatenate((lastframe, mid_np[:h, :w]), 1))
    else:
        write_buffer.put(lastframe)
        for mid in output:
            mid_np = ((mid[0] * 255.).byte().cpu().numpy().transpose(1, 2, 0))
            write_buffer.put(mid_np[:h, :w])
    
    lastframe = frame
    if break_flag:
        break

if args.montage:
    write_buffer.put(np.concatenate((lastframe, lastframe), 1))
else:
    write_buffer.put(lastframe)
write_buffer.put(None)

# Wait for all frames to be written
print('\nFinalizing video output...', end="")
sys.stdout.flush()
while not write_buffer.empty():
    time.sleep(0.1)

# Calculate total processing time
total_time = time.time() - start_time
final_fps = current_frame / total_time if total_time > 0 else 0

print(f" done")
print(f"[SUCCESS] Processing complete! {current_frame} frames in {total_time:.2f}s ({final_fps:.2f} fps)")
sys.stdout.flush()

if not args.png:
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()

if (not args.no_audio) and (not args.png) and fpsNotAssigned and (args.video is not None):
    print("\nProcessing audio...")
    try:
        transferAudio(args.video, vid_out_name)
    except Exception:
        print("Audio transfer failed. Interpolated video will have no audio")
        targetNoAudio = os.path.splitext(vid_out_name)[0] + "_noaudio" + os.path.splitext(vid_out_name)[1]
        os.rename(targetNoAudio, vid_out_name)
        
print(f"\n[DONE] Completed video saved to: {vid_out_name}")