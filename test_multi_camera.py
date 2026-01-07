"""End-to-end test: Single camera vs Multi-camera capture + analysis."""

import asyncio
import logging
import time
from datetime import datetime

# Suppress noisy logging
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('pubnub').setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(message)s')

import config
from vivint_client import VivintClient
from frame_capture import capture_video, capture_multiple_cameras
from gemini_analyzer import analyze_video, analyze_multiple_videos


async def test_single_camera(client: VivintClient, camera_name: str) -> dict:
    """Test single camera capture + analysis."""
    print(f"\n{'='*70}")
    print(f"TEST: Single Camera Capture + Analysis")
    print(f"Camera: {camera_name}")
    print('='*70)

    # Find camera
    camera = None
    for cam in client.cameras:
        if cam.name == camera_name:
            camera = cam
            break

    if not camera:
        return {"error": f"Camera {camera_name} not found"}

    rtsp_url = client.get_rtsp_url(camera.id)
    if not rtsp_url:
        return {"error": "No RTSP URL available"}

    results = {}

    # Capture
    print(f"\n[1/2] Capturing video from {camera_name}...")
    capture_start = time.perf_counter()
    capture_result = await capture_video(rtsp_url, camera_name)
    capture_time = time.perf_counter() - capture_start
    results['capture_time'] = capture_time

    if not capture_result.success:
        return {"error": f"Capture failed: {capture_result.error}"}

    print(f"      Captured in {capture_time:.2f}s")
    print(f"      File: {capture_result.video_path.name} ({capture_result.video_path.stat().st_size} bytes)")

    # Analyze
    print(f"\n[2/2] Analyzing with Gemini...")
    analyze_start = time.perf_counter()
    analysis = await analyze_video(
        capture_result.video_path,
        camera_name=camera_name,
        event_type="motion",
    )
    analyze_time = time.perf_counter() - analyze_start
    results['analyze_time'] = analyze_time

    if analysis:
        print(f"      Analyzed in {analyze_time:.2f}s")
        print(f"      Risk: {analysis.risk_tier}")
        print(f"      Summary: {analysis.summary[:100]}...")
    else:
        print(f"      Analysis failed")

    results['total_time'] = capture_time + analyze_time
    results['success'] = analysis is not None

    # Cleanup
    capture_result.video_path.unlink(missing_ok=True)

    return results


async def test_multi_camera(client: VivintClient, primary_camera: str) -> dict:
    """Test multi-camera capture + analysis."""
    print(f"\n{'='*70}")
    print(f"TEST: Multi-Camera Capture + Analysis")
    print(f"Primary: {primary_camera}")
    print(f"Adjacent: {config.CAMERA_ADJACENCY.get(primary_camera, [])}")
    print('='*70)

    # Build camera URLs dict
    camera_urls = {}

    # Add primary camera
    for cam in client.cameras:
        if cam.name == primary_camera:
            url = client.get_rtsp_url(cam.id)
            if url:
                camera_urls[primary_camera] = url
            break

    # Add adjacent cameras
    adjacent_names = config.CAMERA_ADJACENCY.get(primary_camera, [])
    for adj_name in adjacent_names:
        for cam in client.cameras:
            if cam.name == adj_name:
                url = client.get_rtsp_url(cam.id)
                if url:
                    camera_urls[adj_name] = url
                break

    if not camera_urls:
        return {"error": "No camera URLs available"}

    results = {'cameras': list(camera_urls.keys())}

    # Capture from all cameras simultaneously
    print(f"\n[1/2] Capturing from {len(camera_urls)} cameras simultaneously...")
    capture_start = time.perf_counter()
    multi_capture = await capture_multiple_cameras(
        camera_urls,
        primary_camera=primary_camera,
    )
    capture_time = time.perf_counter() - capture_start
    results['capture_time'] = capture_time

    if not multi_capture.success:
        return {"error": f"Multi-capture failed: {multi_capture.error}"}

    print(f"      Captured {len(multi_capture.videos)} videos in {capture_time:.2f}s")
    for cam_name, video_path in multi_capture.videos.items():
        print(f"        - {cam_name}: {video_path.name} ({video_path.stat().st_size} bytes)")

    # Analyze all videos together
    print(f"\n[2/2] Analyzing {len(multi_capture.videos)} videos with Gemini...")
    analyze_start = time.perf_counter()
    analysis = await analyze_multiple_videos(
        video_paths=multi_capture.videos,
        primary_camera=primary_camera,
        event_type="motion",
    )
    analyze_time = time.perf_counter() - analyze_start
    results['analyze_time'] = analyze_time

    if analysis:
        print(f"      Analyzed in {analyze_time:.2f}s")
        print(f"      Risk: {analysis.risk_tier}")
        print(f"      Summary: {analysis.summary[:100]}...")
    else:
        print(f"      Analysis failed")

    results['total_time'] = capture_time + analyze_time
    results['success'] = analysis is not None

    # Cleanup
    for video_path in multi_capture.videos.values():
        video_path.unlink(missing_ok=True)

    return results


async def main():
    print('='*70)
    print('END-TO-END BENCHMARK: Single vs Multi-Camera')
    print('='*70)
    print(f'Timestamp: {datetime.now().isoformat()}')
    print(f'Video duration: {config.VIDEO_CAPTURE_DURATION_SECONDS}s')
    print(f'Gemini model: {config.GEMINI_MODEL}')
    print()

    # Connect to Vivint
    print("Connecting to Vivint...")
    client = VivintClient()
    if not await client.connect():
        print("Failed to connect to Vivint")
        return

    print(f"Connected. Found {len(client.cameras)} cameras:")
    for cam in client.cameras:
        print(f"  - {cam.name}")

    # Use Driveway as primary (has 2 adjacent cameras)
    primary = "Driveway"

    # Test 1: Single camera
    single_results = await test_single_camera(client, primary)

    # Brief pause between tests
    await asyncio.sleep(2)

    # Test 2: Multi-camera
    multi_results = await test_multi_camera(client, primary)

    # Disconnect
    await client.disconnect()

    # Summary
    print(f"\n{'='*70}")
    print("BENCHMARK RESULTS SUMMARY")
    print('='*70)
    print()

    if "error" in single_results:
        print(f"Single camera: ERROR - {single_results['error']}")
    else:
        print(f"SINGLE CAMERA ({primary}):")
        print(f"  Capture time:  {single_results['capture_time']:>6.2f}s")
        print(f"  Analyze time:  {single_results['analyze_time']:>6.2f}s")
        print(f"  TOTAL:         {single_results['total_time']:>6.2f}s")

    print()

    if "error" in multi_results:
        print(f"Multi camera: ERROR - {multi_results['error']}")
    else:
        cameras = multi_results.get('cameras', [])
        print(f"MULTI-CAMERA ({len(cameras)} cameras: {', '.join(cameras)}):")
        print(f"  Capture time:  {multi_results['capture_time']:>6.2f}s (parallel)")
        print(f"  Analyze time:  {multi_results['analyze_time']:>6.2f}s")
        print(f"  TOTAL:         {multi_results['total_time']:>6.2f}s")

    # Comparison
    if "error" not in single_results and "error" not in multi_results:
        print()
        print("COMPARISON:")

        capture_diff = multi_results['capture_time'] - single_results['capture_time']
        analyze_diff = multi_results['analyze_time'] - single_results['analyze_time']
        total_diff = multi_results['total_time'] - single_results['total_time']

        print(f"  Capture overhead:  {capture_diff:+.2f}s ({len(cameras)} cameras vs 1)")
        print(f"  Analyze overhead:  {analyze_diff:+.2f}s ({len(cameras)} videos vs 1)")
        print(f"  Total overhead:    {total_diff:+.2f}s")

        if total_diff > 0:
            pct = (total_diff / single_results['total_time']) * 100
            print(f"\n  => Multi-camera adds {total_diff:.2f}s ({pct:.0f}%) for {len(cameras)-1} extra camera(s)")
        else:
            print(f"\n  => Multi-camera is faster (unexpected)")

        # Cost efficiency
        print()
        print("VALUE ANALYSIS:")
        print(f"  Single: 1 camera view, 1 API call")
        print(f"  Multi:  {len(cameras)} camera views, 1 API call (same cost)")
        print(f"  => {len(cameras)}x more visual context for ~{total_diff:.1f}s extra latency")


if __name__ == "__main__":
    asyncio.run(main())
