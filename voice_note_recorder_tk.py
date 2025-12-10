import os

# --------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------

DISABLE_SYSTEM_PROXIES = True  # отключать ли системные прокси (V2Ray/TUN и т.п.)
NO_PROXY_LIST = "127.0.0.1,localhost"

SERVER_BASE = "http://localhost:8000"
TRANSCRIBE_URL = f"{SERVER_BASE}/v1/audio/transcriptions"
LANGUAGE = "ru"

FS = 44100
CHANNELS = 1

OUTPUT_FOLDER_DEFAULT = r"C:\Users\finsi\Documents\GameDev_Mind\_recordings"
CURRENT_OUTPUT_FOLDER = OUTPUT_FOLDER_DEFAULT

BG_COLOR = "#1E1E1E"
FG_COLOR = "#FFFFFF"
FG_MUTED = "#999999"
BTN_RECORD_COLOR = "#C0392B"
BTN_SECONDARY_COLOR = "#3C7A89"

# warmup
WARMUP_TIMEOUT_SECONDS = 15  # было 60 → меньше блокировок

# chunking
DEFAULT_CHUNK_SEC = 20          # длина чанка
DEFAULT_OVERLAP_SEC = 2         # наползание чанков
MIC_CHUNK_THRESHOLD_SEC = 120.0  # микрофон: включать чанкинг, если запись > 2 минут
UPLOAD_CHUNK_THRESHOLD_SEC = 180.0  # файлы: включать чанкинг, если запись > 3 минут

# --------------------------------------------------------------
# ABSOLUTE PROXY KILL — теперь опционально
# --------------------------------------------------------------

if DISABLE_SYSTEM_PROXIES:
    for var in (
        "http_proxy",
        "https_proxy",
        "ftp_proxy",
        "all_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "FTP_PROXY",
        "ALL_PROXY",
    ):
        if var in os.environ:
            del os.environ[var]

    os.environ["NO_PROXY"] = NO_PROXY_LIST
    os.environ["no_proxy"] = NO_PROXY_LIST

import requests
from requests.sessions import Session


class NoProxySession(Session):
    def __init__(self):
        super().__init__()
        # игнорировать переменные окружения
        self.trust_env = False
        self.proxies = {"http": None, "https": None}


requests_session = NoProxySession()


def requests_post(url, files=None, data=None, timeout=600):
    return requests_session.post(url, files=files, data=data, timeout=timeout)


# --------------------------------------------------------------
# IMPORTS
# --------------------------------------------------------------

import datetime
import sys
import tempfile
import threading
import time
import socket
import subprocess
import platform
import ctypes
import difflib
import re
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write, read
import tkinter as tk
from tkinter import filedialog, scrolledtext

# --------------------------------------------------------------
# GLOBAL STATE
# --------------------------------------------------------------

audio_buffer = []
recording = False
recording_lock = threading.Lock()
current_stream = None
recording_start_time = None


# --------------------------------------------------------------
# WINDOWS HELPERS
# --------------------------------------------------------------

def minimize_own_console():
    """Сворачивает консольное окно на Windows (без обязательного закрытия)."""
    if platform.system() != "Windows":
        return
    try:
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 6)  # SW_MINIMIZE
    except Exception as e:
        print("[WIN] minimize_own_console error:", e)


# --------------------------------------------------------------
# SERVER / WARMUP
# --------------------------------------------------------------

def wait_for_uvicorn(timeout=WARMUP_TIMEOUT_SECONDS):
    """Ждёт пока /docs станет доступен или истечёт timeout."""
    print(f"[WARMUP] Waiting for API availability on /docs (timeout {timeout}s) ...")
    start = time.time()
    url = f"{SERVER_BASE}/docs"
    while time.time() - start < timeout:
        try:
            r = requests_session.get(url, timeout=1)
            if r.status_code == 200:
                print("[WARMUP] API is ready ✓")
                return True
        except Exception:
            pass
        time.sleep(1)
    print("[WARMUP] API did NOT become ready within timeout ❌")
    return False


def ensure_output_folder():
    os.makedirs(CURRENT_OUTPUT_FOLDER, exist_ok=True)


def save_markdown(text: str):
    ensure_output_folder()
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(CURRENT_OUTPUT_FOLDER, f"voice_note_{now}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# --------------------------------------------------------------
# AUDIO IO
# --------------------------------------------------------------

def record_to_wav():
    """Сливает буфер в один массив и пишет во временный WAV."""
    with recording_lock:
        if not audio_buffer:
            raise RuntimeError("Нет аудиоданных")
        audio = np.concatenate(audio_buffer, axis=0)
        audio_buffer.clear()
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    write(temp, FS, audio.astype(np.float32))
    print("[AUDIO] size:", os.path.getsize(temp))
    return temp


def load_wav_mono(path):
    """
    Загружает WAV, приводит к mono float32 [-1, 1], ресемплит до FS.
    """
    sr, data = read(path)

    # приведение к mono
    if data.ndim > 1:
        data = data.mean(axis=1)

    # приведение типа → float32 [-1;1]
    if np.issubdtype(data.dtype, np.integer):
        max_val = np.iinfo(data.dtype).max
        data = data.astype(np.float32) / float(max_val)
    elif np.issubdtype(data.dtype, np.floating):
        data = data.astype(np.float32)
    else:
        raise TypeError(f"Unsupported WAV dtype: {data.dtype}")

    # ресемплинг при необходимости (простая линейная интерполяция)
    if sr != FS:
        orig_len = len(data)
        new_len = int(orig_len * FS / sr)
        x_old = np.arange(orig_len, dtype=np.float32)
        x_new = np.linspace(0, orig_len - 1, new_len, dtype=np.float32)
        data = np.interp(x_new, x_old, data).astype(np.float32)
        sr = FS

    return data, sr


def decode_audio_to_mono_array(file_path):
    """
    Универсальный декодер любого аудиофайла через ffmpeg в mono float32.
    Если это уже WAV, используем прямое чтение.
    Для всех остальных форматов:
      ffmpeg -i input -ac 1 -ar FS -f wav temp.wav
      → load_wav_mono(temp.wav)
    Требуется ffmpeg в PATH.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".wav":
        return load_wav_mono(file_path)

    print(f"[DECODE] Using ffmpeg for: {file_path}")
    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    cmd = [
        "ffmpeg",
        "-y",           # overwrite
        "-i", file_path,
        "-ac", "1",     # mono
        "-ar", str(FS), # sample rate
        "-f", "wav",
        tmp_wav,
    ]
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        audio, sr = load_wav_mono(tmp_wav)
        return audio, sr
    finally:
        try:
            os.remove(tmp_wav)
        except Exception:
            pass


def split_into_chunks(audio, sr, chunk_sec=DEFAULT_CHUNK_SEC, overlap_sec=DEFAULT_OVERLAP_SEC):
    """
    Делит аудио на куски фиксированной длины с наползанием.
    Меньший chunk_sec → меньше проблем с обрезанием фраз.
    """
    step_sec = max(chunk_sec - overlap_sec, 1.0)
    chunk_size = int(chunk_sec * sr)
    step_size = int(step_sec * sr)

    res = []
    pos = 0
    n = len(audio)
    while pos < n:
        end = pos + chunk_size
        res.append(audio[pos:end])
        pos += step_size
    return res


# --------------------------------------------------------------
# TEXT STITCHING
# --------------------------------------------------------------

def normalize_text(t):
    return " ".join(t.strip().lower().split())


def fuzzy_stitch(prev, cur, max_words=20, similarity_threshold=0.82):
    """
    Примитивная склейка по совпадающему хвосту/началу (по словам).
    """
    prev = prev or ""
    cur = cur or ""
    if not prev.strip():
        return cur
    if not cur.strip():
        return prev

    prev_words = prev.strip().split()
    cur_words = cur.strip().split()

    n = min(max_words, len(prev_words), len(cur_words))
    for k in range(n, 0, -1):
        tail = " ".join(prev_words[-k:])
        head = " ".join(cur_words[:k])
        if normalize_text(tail) == normalize_text(head):
            return prev + " " + " ".join(cur_words[k:])

        # если строки не совпадают буквально, но очень похожи → считаем дублированным хвостом
        ratio = difflib.SequenceMatcher(None, normalize_text(tail), normalize_text(head)).ratio()
        if ratio >= similarity_threshold:
            return prev + " " + " ".join(cur_words[k:])

    return prev + " " + cur


def deduplicate_sentences(text: str, similarity_threshold: float = 0.9) -> str:
    """
    Убирает подряд идущие дубли предложений/фраз (после чанкинга).
    Делается без внешних зависимостей: SequenceMatcher + простая сегментация.
    """

    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    cleaned = []
    last_norm = ""

    for part in parts:
        norm = normalize_text(part)
        if not norm:
            continue
        if last_norm and difflib.SequenceMatcher(None, last_norm, norm).ratio() >= similarity_threshold:
            # слишком похоже на предыдущую фразу → считаем дублем
            continue
        cleaned.append(part)
        last_norm = norm

    return "\n".join(cleaned)


def suppress_repeated_transcript_tail(
    text: str,
    similarity_threshold: float = 0.8,
    min_sentences_per_half: int = 6,
) -> str:
    """
    Если вторая половина текста почти совпадает с первой (частый случай
    повторной транскрипции целиком), отбрасывает хвост и оставляет более
    раннюю версию. Работает поверх deduplicate_sentences.
    """

    sentences = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", text) if p.strip()]
    if len(sentences) < min_sentences_per_half * 2:
        return text

    mid = len(sentences) // 2
    first_half = " ".join(sentences[:mid])
    second_half = " ".join(sentences[mid:])

    ratio = difflib.SequenceMatcher(
        None, normalize_text(first_half), normalize_text(second_half)
    ).ratio()
    if ratio >= similarity_threshold:
        return "\n".join(sentences[:mid]).strip()
    return text


def postprocess_transcript(text: str) -> str:
    """Общий постпроцессинг: dedup + защита от повторной копии текста."""

    cleaned = deduplicate_sentences(text)
    cleaned = suppress_repeated_transcript_tail(cleaned)
    return cleaned.strip()


# --------------------------------------------------------------
# WHISPER HTTP CLIENT
# --------------------------------------------------------------

def transcribe_single(file_path: str) -> str:
    print(f"[HTTP] Sending file: {file_path}")
    with open(file_path, "rb") as f:
        resp = requests_post(
            TRANSCRIBE_URL,
            files={"file": (os.path.basename(file_path), f, "audio/wav")},
            data={"language": LANGUAGE, "response_format": "json"},
            timeout=1200,
        )
    print("[HTTP] Response:", resp.status_code)
    print("[HTTP] Preview:", resp.text[:200])
    resp.raise_for_status()
    data = resp.json()
    return data.get("text") or data.get("transcript") or ""


def transcribe_chunk_array(chunk: np.ndarray, sr: int) -> str:
    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    try:
        write(temp, sr, chunk.astype(np.float32))
        return transcribe_single(temp)
    finally:
        try:
            os.remove(temp)
        except Exception:
            pass


def transcribe_maybe_chunk(
    file_path: str,
    progress_cb=None,
    chunk_threshold_sec: float = 180.0,
) -> str:
    """
    Универсальная логика:
      1) Декодируем аудио через ffmpeg (для любых форматов).
      2) Если длительность <= chunk_threshold_sec → один запрос (используем исходный файл).
      3) Если > threshold → режем на чанки и шлём по кускам, склеиваем fuzzy_stitch.
    """
    try:
        audio, sr = decode_audio_to_mono_array(file_path)
    except Exception as e:
        print("[CHUNK] decode_audio_to_mono_array failed, fallback to single:", e)
        return transcribe_single(file_path)

    duration = len(audio) / float(sr)
    print(f"[CHUNK] Duration {duration:.1f}s (threshold {chunk_threshold_sec:.1f}s)")

    if duration <= chunk_threshold_sec:
        print("[CHUNK] Using single request")
        return postprocess_transcript(transcribe_single(file_path))

    print("[CHUNK] Using chunked mode")
    chunks = split_into_chunks(audio, sr, chunk_sec=DEFAULT_CHUNK_SEC, overlap_sec=DEFAULT_OVERLAP_SEC)
    total = len(chunks)
    full_text = ""
    for i, ch in enumerate(chunks, start=1):
        print(f"[CHUNK] {i}/{total}")
        part = transcribe_chunk_array(ch, sr)
        full_text = fuzzy_stitch(full_text, part)
        if progress_cb:
            progress_cb(i, total)
    return postprocess_transcript(full_text.strip())


def transcribe(file_path: str, progress_cb=None, chunk_threshold_sec: float = 180.0) -> str:
    """
    Обёртка, чтобы удобно задавать разные пороги:
      - для микрофона: MIC_CHUNK_THRESHOLD_SEC
      - для файлов: UPLOAD_CHUNK_THRESHOLD_SEC
    """
    return transcribe_maybe_chunk(
        file_path=file_path,
        progress_cb=progress_cb,
        chunk_threshold_sec=chunk_threshold_sec,
    )


# --------------------------------------------------------------
# DOCKER / ENV CHECKS
# --------------------------------------------------------------

def check_docker_cli():
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def check_docker_daemon():
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def check_container():
    try:
        r = subprocess.run(
            ["docker", "ps", "--filter", "name=whisper-local", "--format", "{{.ID}}"],
            capture_output=True,
            text=True,
        )
        return bool(r.stdout.strip())
    except Exception:
        return False


def check_api():
    try:
        socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
        return True
    except Exception:
        return False


def warmup_whisper():
    if not wait_for_uvicorn(timeout=WARMUP_TIMEOUT_SECONDS):
        print("[WARMUP] Skipped (server not detected or too slow)")
        return False

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    write(temp, FS, np.zeros((FS,), dtype=np.float32))
    try:
        _ = transcribe_single(temp)
        print("[WARMUP] Completed ✓")
        return True
    except Exception as e:
        print("[WARMUP] Failed:", e)
        return False
    finally:
        try:
            os.remove(temp)
        except Exception:
            pass


# --------------------------------------------------------------
# LOG REDIRECTOR (THREAD-SAFE ДЛЯ TKINTER)
# --------------------------------------------------------------

class TextRedirector:
    """Пишет stdout/stderr в Text, но все операции UI проводит через .after()."""

    def __init__(self, widget):
        self.widget = widget

    def _append(self, msg):
        self.widget.config(state=tk.NORMAL)
        self.widget.insert(tk.END, msg)
        self.widget.see(tk.END)
        self.widget.config(state=tk.DISABLED)

    def write(self, msg):
        # даже если print идёт из другого потока — UI обновится в главном
        self.widget.after(0, self._append, msg)

    def flush(self):
        pass


# --------------------------------------------------------------
# TKINTER APP
# --------------------------------------------------------------

class VoiceRecorderApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.configure(bg=BG_COLOR)
        self.root.title("Whisper Voice Notes")

        self.cli_ok = tk.BooleanVar()
        self.daemon_ok = tk.BooleanVar()
        self.container_ok = tk.BooleanVar()
        self.api_ok = tk.BooleanVar()
        self.warmup_done = False
        self.warmup_failed = False

        self.build_ui()
        self.check_environment_async()
        self.warmup_async()
        self.root.after(200, self.update_timer)

    # ---------------- UI BUILD ----------------

    def build_ui(self):
        self.root.geometry("820x440")

        main = tk.Frame(self.root, bg=BG_COLOR)
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # LEFT PANEL
        left = tk.Frame(main, bg=BG_COLOR)
        left.pack(side="left", fill="y")

        tk.Label(
            left,
            text="Whisper Voice Notes",
            bg=BG_COLOR,
            fg=FG_COLOR,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w")

        self.status_label = tk.Label(
            left,
            text="Loading...",
            bg=BG_COLOR,
            fg=FG_MUTED,
            font=("Segoe UI", 10),
        )
        self.status_label.pack(anchor="w", pady=(6, 2))

        # warmup indicator
        warmup_frame = tk.Frame(left, bg=BG_COLOR)
        warmup_frame.pack(anchor="w", pady=(0, 6))

        self.warmup_indicator = tk.Label(
            warmup_frame,
            text="●",
            font=("Segoe UI", 14, "bold"),
            bg=BG_COLOR,
            fg="#E67E22",
        )
        self.warmup_indicator.pack(side="left", padx=(0, 6))

        self.warmup_label = tk.Label(
            warmup_frame,
            text="Warmup: waiting for server...",
            bg=BG_COLOR,
            fg=FG_MUTED,
            font=("Segoe UI", 9),
        )
        self.warmup_label.pack(side="left")

        # env checks
        self.check_labels = {}
        for key, label in [
            ("cli", "Docker CLI"),
            ("daemon", "Docker daemon"),
            ("container", "whisper-local container"),
            ("api", "Whisper API"),
        ]:
            lbl = tk.Label(
                left,
                text=f"[ ] {label}",
                bg=BG_COLOR,
                fg=FG_MUTED,
                font=("Segoe UI", 9),
            )
            lbl.pack(anchor="w")
            self.check_labels[key] = lbl

        # timer
        self.timer_label = tk.Label(
            left,
            text="00:00",
            bg=BG_COLOR,
            fg=FG_COLOR,
            font=("Consolas", 20, "bold"),
        )
        self.timer_label.pack(anchor="w", pady=10)

        # controls
        ctrl = tk.Frame(left, bg=BG_COLOR)
        ctrl.pack(anchor="w", pady=(4, 4))

        self.rec_button = tk.Button(
            ctrl,
            text="Start recording",
            width=16,
            bg=BTN_RECORD_COLOR,
            fg="white",
            command=self.toggle_recording,
        )
        self.rec_button.pack(side="left", padx=(0, 8))

        self.upload_button = tk.Button(
            ctrl,
            text="Transcribe file...",
            width=16,
            bg=BTN_SECONDARY_COLOR,
            fg="white",
            command=self.upload_file,
        )
        self.upload_button.pack(side="left")

        # save path controls
        save_ctrl = tk.Frame(left, bg=BG_COLOR)
        save_ctrl.pack(anchor="w", pady=(4, 4))

        self.save_alt_button = tk.Button(
            save_ctrl,
            text="Save to folder...",
            width=16,
            bg=BTN_SECONDARY_COLOR,
            fg="white",
            command=self.choose_save_folder,
        )
        self.save_alt_button.pack(side="left", padx=(0, 8))

        self.save_reset_button = tk.Button(
            save_ctrl,
            text="Use Obsidian path",
            width=16,
            bg=BTN_SECONDARY_COLOR,
            fg="white",
            command=self.reset_save_folder,
        )
        self.save_reset_button.pack(side="left")

        from_path = CURRENT_OUTPUT_FOLDER
        self.save_path_label = tk.Label(
            left,
            text=f"Save to:\n{from_path}",
            bg=BG_COLOR,
            fg=FG_MUTED,
            font=("Consolas", 8),
            justify="left",
            wraplength=320,
        )
        self.save_path_label.pack(anchor="w", pady=(4, 8))

        self.result_label = tk.Label(
            left,
            text="",
            bg=BG_COLOR,
            fg=FG_MUTED,
            wraplength=320,
            justify="left",
        )
        self.result_label.pack(anchor="w", pady=4)

        # RIGHT PANEL (LOG)
        right = tk.Frame(main, bg=BG_COLOR)
        right.pack(side="left", fill="both", expand=True)

        tk.Label(
            right,
            text="Log",
            bg=BG_COLOR,
            fg=FG_COLOR,
            font=("Segoe UI", 12, "bold"),
        ).pack(anchor="w")

        self.log_area = scrolledtext.ScrolledText(
            right,
            bg="#252526",
            fg="#D4D4D4",
            font=("Consolas", 9),
            state="disabled",
        )
        self.log_area.pack(fill="both", expand=True)

        sys.stdout = TextRedirector(self.log_area)
        sys.stderr = TextRedirector(self.log_area)

    # ---------------- SAVE PATH ----------------

    def choose_save_folder(self):
        global CURRENT_OUTPUT_FOLDER
        path = filedialog.askdirectory(title="Choose folder to save notes")
        if not path:
            return
        CURRENT_OUTPUT_FOLDER = path
        self.save_path_label.config(text=f"Save to:\n{CURRENT_OUTPUT_FOLDER}")
        print("[SAVE] Output folder set to:", CURRENT_OUTPUT_FOLDER)
        self.status_label.config(text="Save path updated", fg="#4CAF50")

    def reset_save_folder(self):
        global CURRENT_OUTPUT_FOLDER
        CURRENT_OUTPUT_FOLDER = OUTPUT_FOLDER_DEFAULT
        self.save_path_label.config(text=f"Save to:\n{CURRENT_OUTPUT_FOLDER}")
        print("[SAVE] Output folder reset to default:", CURRENT_OUTPUT_FOLDER)
        self.status_label.config(text="Save path reset to Obsidian", fg="#4CAF50")

    # ---------------- WARMUP ANIMATION ----------------

    def animate_warmup(self):
        if self.warmup_done:
            self.warmup_indicator.config(fg="#4CAF50")
            self.warmup_label.config(text="Warmup: ready ✓", fg="#4CAF50")
            return
        if self.warmup_failed:
            self.warmup_indicator.config(fg="#777777")
            self.warmup_label.config(
                text="Warmup: skipped (using first request)", fg="#777777"
            )
            return
        current = self.warmup_indicator.cget("fg")
        self.warmup_indicator.config(
            fg="#E67E22" if current != "#E67E22" else "#A65E22"
        )
        self.warmup_label.config(text="Warmup: running...")
        self.root.after(500, self.animate_warmup)

    # ---------------- ENV CHECKS ----------------

    def update_checks(self):
        mapping = {
            "cli": self.cli_ok.get(),
            "daemon": self.daemon_ok.get(),
            "container": self.container_ok.get(),
            "api": self.api_ok.get(),
        }
        all_ok = True
        for key, ok in mapping.items():
            lbl = self.check_labels[key]
            base = lbl.cget("text").replace("[✓]", "[ ]")
            if ok:
                lbl.config(text=base.replace("[ ]", "[✓]"), fg="#4CAF50")
            else:
                lbl.config(text=base.replace("[✓]", "[ ]"), fg=FG_MUTED)
                all_ok = False
        if all_ok:
            self.status_label.config(text="Ready ✓", fg="#4CAF50")
        else:
            self.status_label.config(text="Loading...", fg="#E67E22")

    def check_environment_async(self):
        def worker():
            self.cli_ok.set(check_docker_cli())
            self.daemon_ok.set(check_docker_daemon())
            self.container_ok.set(check_container())
            self.api_ok.set(check_api())
            self.root.after(0, self.update_checks)

        threading.Thread(target=worker, daemon=True).start()

    def warmup_async(self):
        def worker():
            success = warmup_whisper()
            if success:
                self.warmup_done = True
            else:
                self.warmup_failed = True

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(500, self.animate_warmup)

    # ---------------- RECORDING ----------------

    def toggle_recording(self):
        global recording, current_stream, recording_start_time
        if not recording:
            audio_buffer.clear()
            print("[REC] start")
            current_stream = sd.InputStream(
                samplerate=FS,
                channels=CHANNELS,
                callback=self.audio_cb,
            )
            current_stream.start()
            recording = True
            recording_start_time = time.time()
            self.rec_button.config(text="Stop recording")
            self.status_label.config(text="Recording...", fg="#E67E22")
        else:
            print("[REC] stop")
            recording = False
            if current_stream:
                current_stream.stop()
                current_stream.close()
            self.rec_button.config(text="Start recording")
            self.status_label.config(text="Processing...", fg="#E67E22")
            threading.Thread(
                target=self.process_recording,
                daemon=True,
            ).start()

    def audio_cb(self, data, frames, time_info, status):
        if recording:
            with recording_lock:
                audio_buffer.append(data.copy())

    def process_recording(self):
        try:
            wav = record_to_wav()

            def progress(done, total):
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"Processing chunks {done}/{total}", fg="#E67E22"
                    ),
                )

            # микрофон: включаем чанкинг, если длительность > MIC_CHUNK_THRESHOLD_SEC
            txt = transcribe(
                wav,
                progress_cb=progress,
                chunk_threshold_sec=MIC_CHUNK_THRESHOLD_SEC,
            )
            out = save_markdown(txt)
            self.root.after(0, lambda: self.after_success(out))
        except Exception as e:
            print("[ERROR recording]", e)
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Error: {e}", fg="#E74C3C"
                ),
            )

    # ---------------- FILE UPLOAD ----------------

    def upload_file(self):
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[
                ("Audio files", "*.wav;*.mp3;*.m4a;*.flac;*.ogg;*.opus;*.mp4;*.webm;*.mkv"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.status_label.config(text="Processing...", fg="#E67E22")
        threading.Thread(
            target=self.process_uploaded,
            args=(path,),
            daemon=True,
        ).start()

    def process_uploaded(self, path):
        try:
            def progress(done, total):
                self.root.after(
                    0,
                    lambda: self.status_label.config(
                        text=f"Processing chunks {done}/{total}", fg="#E67E22"
                    ),
                )

            txt = transcribe(
                path,
                progress_cb=progress,
                chunk_threshold_sec=UPLOAD_CHUNK_THRESHOLD_SEC,
            )
            out = save_markdown(txt)
            self.root.after(0, lambda: self.after_success(out))
        except Exception as e:
            print("[ERROR upload]", e)
            self.root.after(
                0,
                lambda: self.status_label.config(
                    text=f"Error: {e}", fg="#E74C3C"
                ),
            )

    # ---------------- POST SUCCESS ----------------

    def after_success(self, out):
        self.status_label.config(text="Done ✓", fg="#4CAF50")
        self.result_label.config(text=f"Saved:\n{out}")

    # ---------------- TIMER ----------------

    def update_timer(self):
        global recording_start_time
        if recording and recording_start_time:
            elapsed = int(time.time() - recording_start_time)
            m, s = divmod(elapsed, 60)
            self.timer_label.config(text=f"{m:02d}:{s:02d}")
        self.root.after(200, self.update_timer)


# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------

if __name__ == "__main__":
    minimize_own_console()

    root = tk.Tk()
    root.title("Whisper Voice Notes")

    def quit_app():
        global recording, current_stream
        recording = False
        if current_stream:
            try:
                current_stream.stop()
                current_stream.close()
            except Exception:
                pass
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", quit_app)
    app = VoiceRecorderApp(root)
    root.mainloop()
