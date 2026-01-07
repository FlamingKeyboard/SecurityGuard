"""Benchmark comparison: Image vs Video analysis."""

import asyncio
import subprocess
import time
import os
import sys
from pathlib import Path
from datetime import datetime
import logging

# Suppress noisy logging
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.basicConfig(level=logging.WARNING)

# Try to import psutil for CPU/memory monitoring
try:
    import psutil
    process = psutil.Process()
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not installed, CPU/memory metrics unavailable")


async def benchmark_test():
    import config
    from gemini_analyzer import analyze_frame, analyze_video, get_gemini_api_key

    print('=' * 70)
    print('PERFORMANCE BENCHMARK: Image vs Video Analysis')
    print('=' * 70)
    print(f'Timestamp: {datetime.now().isoformat()}')
    print(f'Gemini Model: {config.GEMINI_MODEL}')
    print()

    # Check API key
    if not get_gemini_api_key():
        print('ERROR: No Gemini API key')
        return

    results = {
        'frames': {},
        'video': {},
    }

    test_dir = config.FRAME_CAPTURE_DIR

    # =========================================================================
    # TEST 1: Frame-based analysis (3 frames)
    # =========================================================================
    print('-' * 70)
    print('TEST 1: Frame-based Analysis (3 separate frames)')
    print('-' * 70)

    # Create 3 test frames
    frames = []
    print('Creating 3 test frames...')

    frame_create_start = time.perf_counter()
    for i in range(3):
        frame_path = test_dir / f'bench_frame_{i}.jpg'
        cmd = [
            'ffmpeg', '-y', '-f', 'lavfi',
            '-i', f'testsrc=duration=0.1:size=640x480:rate=1',
            '-frames:v', '1', '-q:v', '2',
            str(frame_path)
        ]
        subprocess.run(cmd, capture_output=True)
        frames.append(frame_path)
    frame_create_time = time.perf_counter() - frame_create_start
    print(f'  Created {len(frames)} frames in {frame_create_time:.2f}s')

    # Baseline memory
    if HAS_PSUTIL:
        mem_baseline = process.memory_info().rss / 1024 / 1024

    # Analyze each frame and measure time
    print('Analyzing frames with Gemini...')

    total_start = time.perf_counter()
    frame_times = []
    cpu_samples = []

    for i, frame in enumerate(frames):
        if HAS_PSUTIL:
            psutil.cpu_percent(interval=None)  # Prime
        frame_start = time.perf_counter()
        analysis = await analyze_frame(frame, camera_name=f'Bench Frame {i}')
        frame_time = time.perf_counter() - frame_start
        if HAS_PSUTIL:
            cpu_sample = psutil.cpu_percent(interval=None)
            cpu_samples.append(cpu_sample)
            print(f'  Frame {i+1}: {frame_time:.2f}s (CPU: {cpu_sample:.1f}%)')
        else:
            print(f'  Frame {i+1}: {frame_time:.2f}s')
        frame_times.append(frame_time)

    total_frame_time = time.perf_counter() - total_start

    if HAS_PSUTIL:
        mem_after = process.memory_info().rss / 1024 / 1024
        results['frames']['cpu'] = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0
        results['frames']['memory'] = mem_after - mem_baseline
    else:
        results['frames']['cpu'] = 0
        results['frames']['memory'] = 0

    results['frames']['total'] = frame_create_time + total_frame_time
    results['frames']['api_time'] = total_frame_time
    results['frames']['create_time'] = frame_create_time

    # Cleanup frames
    for f in frames:
        f.unlink()

    print()
    print(f'  Frame creation: {frame_create_time:.2f}s')
    print(f'  Gemini API (3 calls): {total_frame_time:.2f}s')
    print(f'  TOTAL TIME: {results["frames"]["total"]:.2f}s')
    if HAS_PSUTIL:
        print(f'  Avg CPU: {results["frames"]["cpu"]:.1f}%')
        print(f'  Memory Delta: {results["frames"]["memory"]:+.1f} MB')
    print()

    # Small pause between tests
    await asyncio.sleep(2)

    # =========================================================================
    # TEST 2: Video-based analysis (3-second clip)
    # =========================================================================
    print('-' * 70)
    print('TEST 2: Video-based Analysis (3-second clip)')
    print('-' * 70)

    # Reset baseline
    if HAS_PSUTIL:
        mem_baseline = process.memory_info().rss / 1024 / 1024

    # Create test video
    video_path = test_dir / 'bench_video.mp4'
    print('Creating 3-second test video...')

    if HAS_PSUTIL:
        psutil.cpu_percent(interval=None)
    video_create_start = time.perf_counter()
    cmd = [
        'ffmpeg', '-y', '-f', 'lavfi',
        '-i', 'testsrc=duration=3:size=640x480:rate=30',
        '-c:v', 'libx264', '-preset', 'ultrafast',
        str(video_path)
    ]
    subprocess.run(cmd, capture_output=True)
    video_create_time = time.perf_counter() - video_create_start
    video_file_size = video_path.stat().st_size

    if HAS_PSUTIL:
        cpu_create = psutil.cpu_percent(interval=None)
        print(f'  Created: {video_file_size} bytes in {video_create_time:.2f}s (CPU: {cpu_create:.1f}%)')
    else:
        cpu_create = 0
        print(f'  Created: {video_file_size} bytes in {video_create_time:.2f}s')

    # Analyze video
    print('Analyzing video with Gemini (includes upload + processing wait)...')

    if HAS_PSUTIL:
        psutil.cpu_percent(interval=None)
    video_start = time.perf_counter()
    analysis = await analyze_video(video_path, camera_name='Bench Video')
    video_time = time.perf_counter() - video_start

    if HAS_PSUTIL:
        cpu_after = psutil.cpu_percent(interval=None)
        mem_after = process.memory_info().rss / 1024 / 1024
        results['video']['cpu'] = (cpu_create + cpu_after) / 2
        results['video']['memory'] = mem_after - mem_baseline
        print(f'  Upload + Process + Analyze: {video_time:.2f}s (CPU: {cpu_after:.1f}%)')
    else:
        results['video']['cpu'] = 0
        results['video']['memory'] = 0
        print(f'  Upload + Process + Analyze: {video_time:.2f}s')

    results['video']['total'] = video_create_time + video_time
    results['video']['api_time'] = video_time
    results['video']['create_time'] = video_create_time
    results['video']['file_size'] = video_file_size

    # Cleanup
    video_path.unlink()

    print()
    print(f'  Video creation: {video_create_time:.2f}s')
    print(f'  Gemini API (1 call): {video_time:.2f}s')
    print(f'  TOTAL TIME: {results["video"]["total"]:.2f}s')
    if HAS_PSUTIL:
        print(f'  Avg CPU: {results["video"]["cpu"]:.1f}%')
        print(f'  Memory Delta: {results["video"]["memory"]:+.1f} MB')
    print()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print('=' * 70)
    print('BENCHMARK RESULTS SUMMARY')
    print('=' * 70)
    print()

    print(f'{"Metric":<35} {"Frames (3x)":<15} {"Video (3s)":<15}')
    print('-' * 70)
    print(f'{"Media Creation":<35} {results["frames"]["create_time"]:>6.2f}s        {results["video"]["create_time"]:>6.2f}s')
    print(f'{"Gemini API Time":<35} {results["frames"]["api_time"]:>6.2f}s        {results["video"]["api_time"]:>6.2f}s')
    print(f'{"Total End-to-End Latency":<35} {results["frames"]["total"]:>6.2f}s        {results["video"]["total"]:>6.2f}s')
    print(f'{"Number of API Calls":<35} {"3":>6}         {"1":>6}')
    if HAS_PSUTIL:
        print(f'{"Avg CPU During Analysis":<35} {results["frames"]["cpu"]:>6.1f}%        {results["video"]["cpu"]:>6.1f}%')
        print(f'{"Memory Delta":<35} {results["frames"]["memory"]:>+6.1f} MB      {results["video"]["memory"]:>+6.1f} MB')
    print()

    diff = results['video']['total'] - results['frames']['total']
    pct = abs(diff) / results['frames']['total'] * 100
    if diff > 0:
        print(f'=> Video is {diff:.2f}s ({pct:.0f}%) SLOWER than frames')
    else:
        print(f'=> Video is {abs(diff):.2f}s ({pct:.0f}%) FASTER than frames')

    print()
    print('KEY INSIGHTS:')
    print('  - Video adds ~3-4s for Gemini file processing (upload + ACTIVE wait)')
    print('  - Video uses 1 API call vs 3 (fewer round-trips, potential cost savings)')
    print('  - On real RTSP: remux (-c:v copy) = near-zero CPU vs 3x ffmpeg decode')
    print('  - Video provides temporal context (motion direction, behavior patterns)')
    print()
    print('RECOMMENDATION:')
    if diff > 2:
        print('  For latency-sensitive use: stick with frames (faster)')
        print('  For better AI context: consider video (more accurate)')
    else:
        print('  Video latency is acceptable - consider enabling for better AI analysis')


if __name__ == '__main__':
    asyncio.run(benchmark_test())
