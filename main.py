import os
import asyncio
import threading
import time
import wave
import tempfile
import numpy as np
from queue import Queue, Empty

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn
from faster_whisper import WhisperModel

# ── Load model ────────────────────────────────────────────────────────────
print("⏳ Loading Whisper...")
model = WhisperModel("turbo", device="cuda", compute_type="float16", num_workers=4)
print("✅ Whisper ready")

app = FastAPI()
NUM_WORKERS = 4  # 4 parallel transcription threads hitting GPU simultaneously


# ── Transcribe ────────────────────────────────────────────────────────────
def transcribe(audio_np: np.ndarray, seq: int) -> dict:
    duration = len(audio_np) / 16000
    rms = np.sqrt(np.mean(audio_np ** 2))

    if len(audio_np) < 4000 or rms < 0.004:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name

    with wave.open(tmp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes((audio_np * 32767).astype(np.int16).tobytes())

    t0 = time.perf_counter()
    segments, _ = model.transcribe(
        tmp_path,
        beam_size=1,
        language="en",
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = " ".join(s.text for s in segments).strip()
    os.unlink(tmp_path)

    whisper_ms = round((time.perf_counter() - t0) * 1000, 1)
    rtf = round((whisper_ms / 1000) / duration, 2)
    print(f"📝 [seq={seq} | {whisper_ms}ms | RTF={rtf}]: {text}")

    return {"text": text, "whisper_ms": whisper_ms, "rtf": rtf, "seq": seq}


# ── Routes ────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    loop = asyncio.get_event_loop()
    audio_queue: Queue = Queue()
    result_queue: Queue = Queue()
    running = True
    seq_counter = 0
    OVERLAP_SAMPLES = 8000

    overlap_lock = threading.Lock()
    prev_tail = {"data": np.array([], dtype=np.float32)}

    # ── Workers: NUM_WORKERS threads all hitting GPU in parallel ──────────
    def worker():
        while running:
            try:
                item = audio_queue.get(timeout=0.5)
            except Empty:
                continue
            if item is None:
                break

            raw_bytes, seq, queued_at = item
            queue_ms = round((time.perf_counter() - queued_at) * 1000, 1)
            current = np.frombuffer(raw_bytes, dtype=np.float32).copy()

            with overlap_lock:
                tail = prev_tail["data"]
                audio_with_overlap = np.concatenate([tail, current]) if len(tail) > 0 else current
                prev_tail["data"] = current[-OVERLAP_SAMPLES:] if len(current) >= OVERLAP_SAMPLES else current.copy()

            result = transcribe(audio_with_overlap, seq)
            if result:
                result["queue_ms"] = queue_ms
                result_queue.put(result)

    # ── Sender: delivers results in original sequence order ───────────────
    def sender():
        pending = {}
        next_seq = 0
        while running or not result_queue.empty():
            try:
                result = result_queue.get(timeout=0.5)
            except Empty:
                continue
            pending[result.get("seq", next_seq)] = result
            while next_seq in pending:
                r = pending.pop(next_seq)
                r["type"] = "transcript"
                asyncio.run_coroutine_threadsafe(websocket.send_json(r), loop)
                next_seq += 1

    # Launch workers
    worker_threads = [threading.Thread(target=worker, daemon=True, name=f"w{i}") for i in range(NUM_WORKERS)]
    sender_thread  = threading.Thread(target=sender, daemon=True)
    for t in worker_threads: t.start()
    sender_thread.start()

    async def drain_notifier():
        while audio_queue.qsize() > 0 or result_queue.qsize() > 0:
            await asyncio.sleep(0.3)
        try:
            await websocket.send_json({"type": "done"})
        except Exception:
            pass

    async def keepalive():
        while running:
            await asyncio.sleep(8)
            try:
                await websocket.send_json({"type": "ping", "queue": audio_queue.qsize()})
            except Exception:
                break

    ping_task = asyncio.create_task(keepalive())

    try:
        while True:
            data = await websocket.receive_bytes()
            if len(data) < 500:
                if data.decode("utf-8", errors="ignore").strip() == "STOP":
                    print("⏹️  Draining queue...")
                    asyncio.create_task(drain_notifier())
                continue
            audio_queue.put((data, seq_counter, time.perf_counter()))
            print(f"📥 seq={seq_counter} | {len(data)}b | queue={audio_queue.qsize()}")
            seq_counter += 1

    except WebSocketDisconnect:
        print("🔌 Disconnected")
    finally:
        running = False
        ping_task.cancel()
        for _ in range(NUM_WORKERS):
            audio_queue.put(None)  # poison pill per worker
        for t in worker_threads: t.join(timeout=5)
        sender_thread.join(timeout=2)


if __name__ == "__main__":
    print("🎙️  Transcriber running at http://localhost:8000")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)