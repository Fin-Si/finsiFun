"""
Mouse+Keyboard Macro Recorder (GUI)

✅ Records:
- mouse moves (throttled)
- mouse clicks (left/right/middle)
- mouse scroll
- keyboard press/release

✅ Plays back by hotkey (optionally looped)
✅ Hotkeys editable inside GUI
✅ 3 playback speeds: 1x / 2x / 3x
✅ Record hotkey toggles start/stop

Requirements:
    pip install pynput

Run:
    python macro_gui_recorder.py

Notes (Windows):
- Some apps (games, admin windows) may block global hooks unless you run this script as Administrator.
"""

import json
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

from pynput import mouse, keyboard
from pynput.mouse import Button as MouseButton, Controller as MouseController
from pynput.keyboard import Key, KeyCode, Controller as KeyboardController


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "macro_gui_config.json")


# ---------- helpers: key naming / hotkey formatting ----------

SPECIAL_KEY_TO_HOTKEY = {
    Key.ctrl: "<ctrl>",
    Key.ctrl_l: "<ctrl>",
    Key.ctrl_r: "<ctrl>",
    Key.alt: "<alt>",
    Key.alt_l: "<alt>",
    Key.alt_r: "<alt>",
    Key.shift: "<shift>",
    Key.shift_l: "<shift>",
    Key.shift_r: "<shift>",
    Key.cmd: "<cmd>",
    Key.cmd_l: "<cmd>",
    Key.cmd_r: "<cmd>",
    Key.enter: "<enter>",
    Key.space: "<space>",
    Key.tab: "<tab>",
    Key.esc: "<esc>",
    Key.backspace: "<backspace>",
    Key.delete: "<delete>",
    Key.home: "<home>",
    Key.end: "<end>",
    Key.page_up: "<page_up>",
    Key.page_down: "<page_down>",
    Key.up: "<up>",
    Key.down: "<down>",
    Key.left: "<left>",
    Key.right: "<right>",
}

for i in range(1, 25):
    SPECIAL_KEY_TO_HOTKEY[getattr(Key, f"f{i}")] = f"<f{i}>"


def keycode_to_char(k: KeyCode) -> str | None:
    if k.char and len(k.char) == 1 and k.char.isprintable():
        return k.char.lower()
    vk = getattr(k, "vk", None)
    if vk is None:
        return None
    try:
        candidate = chr(vk)
    except (TypeError, ValueError):
        return None
    if len(candidate) != 1 or not candidate.isprintable():
        return None
    return candidate.lower()


def key_to_hotkey_token(k) -> str | None:
    """Convert pynput key to GlobalHotKeys token."""
    if isinstance(k, KeyCode):
        return keycode_to_char(k)
    if k in SPECIAL_KEY_TO_HOTKEY:
        return SPECIAL_KEY_TO_HOTKEY[k]
    return None


def normalize_hotkey_string(s: str) -> str:
    """
    Normalize user-entered hotkey to pynput GlobalHotKeys format:
    '<ctrl>+<alt>+r'
    """
    s = s.strip().lower().replace(" ", "")
    parts = [p for p in s.split("+") if p]
    norm = []
    for p in parts:
        if p in ("ctrl", "control"):
            norm.append("<ctrl>")
        elif p == "alt":
            norm.append("<alt>")
        elif p == "shift":
            norm.append("<shift>")
        elif p in ("win", "cmd", "meta", "super"):
            norm.append("<cmd>")
        elif p.startswith("<") and p.endswith(">"):
            norm.append(p)
        else:
            if p.startswith("f") and p[1:].isdigit():
                norm.append(f"<{p}>")
            elif len(p) == 1:
                norm.append(p)
            else:
                norm.append(p)

    # remove duplicates but preserve order
    seen = set()
    out = []
    for p in norm:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return "+".join(out)


# ---------- recorder/player core ----------

class MacroApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Macro Recorder (mouse+keyboard)")

        # State
        self.recording = False
        self.playing = False

        self.events = []  # list of dicts, each has 'dt' and event payload
        self.events_lock = threading.Lock()

        self._last_t = None
        self.record_lock = threading.Lock()  # protects _last_t and event stamping

        # Throttle for mouse move
        self._last_move_t = 0.0
        self._last_move_xy = None
        self.MOVE_THROTTLE_SEC = 0.005  # 5ms
        self.MOVE_MIN_PIXELS = 0       # record all movement for accuracy

        self.mouse_ctrl = MouseController()
        self.kb_ctrl = KeyboardController()

        # Pynput listeners
        self.mouse_listener = None
        self.kb_listener = None

        # GlobalHotKeys manager
        self.hotkeys_manager = None
        self.hotkeys_lock = threading.Lock()
        self.hotkeys_suspended = False

        # Config vars
        cfg = self.load_config()
        self.var_record_hotkey = tk.StringVar(value=cfg.get("record_hotkey", "<ctrl>+<alt>+r"))
        self.var_play_hotkey = tk.StringVar(value=cfg.get("play_hotkey", "<ctrl>+<alt>+p"))
        self.var_loop = tk.BooleanVar(value=cfg.get("loop", False))
        self.var_speed = tk.IntVar(value=cfg.get("speed", 1))  # 1/2/3

        # Build UI
        self.build_ui()

        # Start global hotkeys
        self.restart_global_hotkeys()

        # Ensure clean exit
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------- UI ----------
    def build_ui(self):
        pad = {"padx": 10, "pady": 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, **pad)

        hotkeys_box = ttk.LabelFrame(frm, text="Hotkeys (global)")
        hotkeys_box.pack(fill="x", **pad)

        row = ttk.Frame(hotkeys_box)
        row.pack(fill="x", **pad)

        ttk.Label(row, text="Record toggle (start/stop):").grid(row=0, column=0, sticky="w")
        self.ent_record = ttk.Entry(row, textvariable=self.var_record_hotkey, width=30)
        self.ent_record.grid(row=0, column=1, sticky="we", padx=6)
        ttk.Button(row, text="Set…", command=lambda: self.capture_hotkey_into(self.var_record_hotkey)).grid(row=0, column=2)

        ttk.Label(row, text="Play/Stop:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.ent_play = ttk.Entry(row, textvariable=self.var_play_hotkey, width=30)
        self.ent_play.grid(row=1, column=1, sticky="we", padx=6, pady=(6, 0))
        ttk.Button(row, text="Set…", command=lambda: self.capture_hotkey_into(self.var_play_hotkey)).grid(row=1, column=2, pady=(6, 0))

        row.columnconfigure(1, weight=1)

        ttk.Button(hotkeys_box, text="Apply hotkeys", command=self.apply_hotkeys).pack(anchor="e", padx=10, pady=(0, 10))

        opts = ttk.LabelFrame(frm, text="Playback options")
        opts.pack(fill="x", **pad)

        ttk.Checkbutton(opts, text="Loop playback", variable=self.var_loop).pack(anchor="w", padx=10, pady=4)

        speed_row = ttk.Frame(opts)
        speed_row.pack(fill="x", padx=10, pady=4)
        ttk.Label(speed_row, text="Speed:").pack(side="left")

        for val, txt in [(1, "1x"), (2, "2x"), (3, "3x")]:
            ttk.Radiobutton(speed_row, text=txt, value=val, variable=self.var_speed).pack(side="left", padx=10)

        controls = ttk.LabelFrame(frm, text="Controls")
        controls.pack(fill="x", **pad)

        btn_row = ttk.Frame(controls)
        btn_row.pack(fill="x", padx=10, pady=10)

        self.btn_record = ttk.Button(btn_row, text="Start recording", command=self.toggle_recording)
        self.btn_record.pack(side="left")

        self.btn_play = ttk.Button(btn_row, text="Play", command=self.toggle_playback)
        self.btn_play.pack(side="left", padx=10)

        self.btn_clear = ttk.Button(btn_row, text="Clear recording", command=self.clear_recording)
        self.btn_clear.pack(side="left")

        self.status = tk.StringVar(value="Ready.")
        self.mode = tk.StringVar(value="Idle")
        self.mode_label = ttk.Label(frm, textvariable=self.mode)
        self.mode_label.pack(fill="x", padx=12, pady=(2, 0))
        self.set_mode("Idle", "black")
        ttk.Label(frm, textvariable=self.status).pack(fill="x", padx=12, pady=(2, 10))

        hint = (
            "Tip: Click “Set…” and press a key combo (e.g., Ctrl+Alt+R).\n"
            "Record hotkey toggles start/stop. Play hotkey toggles play/stop."
        )
        ttk.Label(frm, text=hint).pack(fill="x", padx=12, pady=(0, 10))

    def set_status(self, s: str):
        self.status.set(s)

    def set_mode(self, text: str, color: str):
        self.mode.set(text)
        self.mode_label.configure(foreground=color)

    # ---------- config ----------
    def load_config(self) -> dict:
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def save_config(self):
        data = {
            "record_hotkey": normalize_hotkey_string(self.var_record_hotkey.get()),
            "play_hotkey": normalize_hotkey_string(self.var_play_hotkey.get()),
            "loop": bool(self.var_loop.get()),
            "speed": int(self.var_speed.get()),
        }
        try:
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ---------- hotkey capture inside GUI ----------
    def capture_hotkey_into(self, target_var: tk.StringVar):
        if self.recording:
            messagebox.showwarning("Hotkey", "Stop recording before changing hotkeys.")
            return
        if self.playing:
            messagebox.showwarning("Hotkey", "Stop playback before changing hotkeys.")
            return

        top = tk.Toplevel(self.root)
        top.title("Press hotkey…")
        top.geometry("360x120")
        top.transient(self.root)
        top.grab_set()

        info = ttk.Label(top, text="Press the key combination now.\nRelease all keys to confirm.")
        info.pack(padx=12, pady=12)

        pressed_tokens = set()
        started = time.time()
        done_flag = {"done": False}
        held = set()

        def on_press(k):
            token = key_to_hotkey_token(k)
            if token:
                pressed_tokens.add(token)
                held.add(token)

        def on_release(k):
            token = key_to_hotkey_token(k)
            if token and token in held:
                held.remove(token)

            if not held and pressed_tokens and not done_flag["done"]:
                done_flag["done"] = True
                hotkey = "+".join(sorted(pressed_tokens, key=lambda x: (0 if x.startswith("<") else 1, x)))
                target_var.set(hotkey)
                try:
                    listener.stop()
                except Exception:
                    pass
                top.destroy()

        listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        listener.start()

        def on_close():
            if not done_flag["done"]:
                try:
                    listener.stop()
                except Exception:
                    pass
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", on_close)

        def watchdog():
            while top.winfo_exists() and not done_flag["done"]:
                if time.time() - started > 20:
                    self.root.after(0, on_close)
                    break
                time.sleep(0.2)

        threading.Thread(target=watchdog, daemon=True).start()

    # ---------- global hotkeys ----------
    def apply_hotkeys(self):
        self.var_record_hotkey.set(normalize_hotkey_string(self.var_record_hotkey.get()))
        self.var_play_hotkey.set(normalize_hotkey_string(self.var_play_hotkey.get()))
        self.restart_global_hotkeys()
        self.save_config()
        self.set_status("Hotkeys applied.")

    def suspend_global_hotkeys(self):
        with self.hotkeys_lock:
            self.hotkeys_suspended = True
            try:
                if self.hotkeys_manager:
                    self.hotkeys_manager.stop()
            except Exception:
                pass

    def resume_global_hotkeys(self):
        with self.hotkeys_lock:
            self.hotkeys_suspended = False
        # restart outside lock to avoid long lock holding
        self.restart_global_hotkeys()

    def restart_global_hotkeys(self):
        with self.hotkeys_lock:
            if self.hotkeys_suspended:
                return
            try:
                if self.hotkeys_manager:
                    self.hotkeys_manager.stop()
            except Exception:
                pass

            rec = normalize_hotkey_string(self.var_record_hotkey.get())
            play = normalize_hotkey_string(self.var_play_hotkey.get())

            if rec == play:
                self.set_status("Warning: Record hotkey and Play hotkey are the same (change one).")

            def safe_call(fn):
                def _inner():
                    self.root.after(0, fn)
                return _inner

            mapping = {
                rec: safe_call(self.toggle_recording),
                play: safe_call(self.toggle_playback),
            }
            try:
                self.hotkeys_manager = keyboard.GlobalHotKeys(mapping)
                self.hotkeys_manager.start()
            except Exception as e:
                self.hotkeys_manager = None
                self.set_status(f"Failed to register hotkeys: {e}")

    # ---------- recording ----------
    def toggle_recording(self):
        if self.playing:
            self.set_status("Stop playback before recording.")
            return
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        # Disable global hotkeys so record/play hotkeys won't be captured into macro
        self.suspend_global_hotkeys()

        with self.events_lock:
            self.events.clear()

        with self.record_lock:
            self._last_t = time.perf_counter()
            self._last_move_t = 0.0
            self._last_move_xy = None

        self.recording = True
        self.btn_record.config(text="Stop recording")
        self.set_mode("Recording", "red")
        self.set_status("Recording… (use record hotkey to stop)")

        def stamp_dt_and_update_last():
            now = time.perf_counter()
            dt = now - self._last_t
            self._last_t = now
            return dt

        def safe_append(ev: dict):
            with self.events_lock:
                self.events.append(ev)

        # mouse callbacks
        def on_move(x, y):
            if not self.recording:
                return
            now = time.perf_counter()
            with self.record_lock:
                if now - self._last_move_t < self.MOVE_THROTTLE_SEC:
                    return
                if self._last_move_xy is not None:
                    lx, ly = self._last_move_xy
                    if abs(x - lx) <= self.MOVE_MIN_PIXELS and abs(y - ly) <= self.MOVE_MIN_PIXELS:
                        return
                dt = stamp_dt_and_update_last()
                self._last_move_t = now
                self._last_move_xy = (x, y)
            safe_append({"dt": dt, "type": "mouse_move", "x": x, "y": y})

        def on_click(x, y, button, pressed):
            if not self.recording:
                return
            btn = str(button)
            with self.record_lock:
                dt = stamp_dt_and_update_last()
            safe_append({"dt": dt, "type": "mouse_click", "x": x, "y": y, "button": btn, "pressed": bool(pressed)})

        def on_scroll(x, y, dx, dy):
            if not self.recording:
                return
            with self.record_lock:
                dt = stamp_dt_and_update_last()
            safe_append({"dt": dt, "type": "mouse_scroll", "x": x, "y": y, "dx": dx, "dy": dy})

        # keyboard callbacks
        def on_press(k):
            if not self.recording:
                return
            with self.record_lock:
                dt = stamp_dt_and_update_last()
            safe_append({"dt": dt, "type": "key_press", "key": self.serialize_key(k)})

        def on_release(k):
            if not self.recording:
                return
            with self.record_lock:
                dt = stamp_dt_and_update_last()
            safe_append({"dt": dt, "type": "key_release", "key": self.serialize_key(k)})

        self.mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        self.kb_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self.mouse_listener.start()
        self.kb_listener.start()

    def stop_recording(self):
        self.recording = False
        self.btn_record.config(text="Start recording")
        self.set_mode("Idle", "black")

        try:
            if self.mouse_listener:
                self.mouse_listener.stop()
        except Exception:
            pass
        try:
            if self.kb_listener:
                self.kb_listener.stop()
        except Exception:
            pass

        self.mouse_listener = None
        self.kb_listener = None

        # Re-enable global hotkeys after recording ends
        self.resume_global_hotkeys()

        with self.events_lock:
            n = len(self.events)
        self.set_status(f"Recording stopped. Events captured: {n}")

    def clear_recording(self):
        if self.recording or self.playing:
            self.set_status("Stop recording/playback before clearing.")
            return
        with self.events_lock:
            self.events.clear()
        self.set_status("Recording cleared.")

    # ---------- playback ----------
    def toggle_playback(self):
        if self.recording:
            self.set_status("Stop recording before playback.")
            return

        with self.events_lock:
            has_events = bool(self.events)
        if not has_events:
            self.set_status("Nothing recorded yet.")
            return

        if not self.playing:
            self.start_playback()
        else:
            self.request_stop_playback()

    def start_playback(self):
        # Disable global hotkeys so play hotkey won't get triggered by playback output
        self.suspend_global_hotkeys()

        self.playing = True
        self.btn_play.config(text="Stop")
        self.set_mode("Playback", "green")
        self.set_status("Playing… (use play hotkey to stop)")

        # snapshot events once per run loop (protect against clear/edit)
        with self.events_lock:
            events_snapshot = list(self.events)

        def runner():
            pressed_keys = set()
            pressed_mouse = set()
            try:
                speed = int(self.var_speed.get())
                factor = {1: 1.0, 2: 2.0, 3: 3.0}.get(speed, 1.0)
                loop = bool(self.var_loop.get())

                while self.playing:
                    self.play_events(events_snapshot, factor, pressed_keys, pressed_mouse)
                    if not loop:
                        break
            finally:
                # Safety-release to avoid stuck keys/buttons
                for k in list(pressed_keys):
                    try:
                        self.kb_ctrl.release(k)
                    except Exception:
                        pass
                for b in list(pressed_mouse):
                    try:
                        self.mouse_ctrl.release(b)
                    except Exception:
                        pass
                self.root.after(0, self.stop_playback_ui)

        threading.Thread(target=runner, daemon=True).start()

    def request_stop_playback(self):
        # Only flips flag; UI will be updated by runner finally (single path)
        self.playing = False

    def stop_playback_ui(self):
        self.playing = False
        self.btn_play.config(text="Play")
        self.set_mode("Idle", "black")

        # Re-enable hotkeys after playback ends
        self.resume_global_hotkeys()

        with self.events_lock:
            has_events = bool(self.events)
        self.set_status("Ready. Playback stopped." if has_events else "Ready.")

    def play_events(self, events, speed_factor: float, pressed_keys: set, pressed_mouse: set):
        for ev in events:
            if not self.playing:
                break

            dt = float(ev.get("dt", 0.0))
            if dt > 0:
                time.sleep(dt / speed_factor)

            t = ev.get("type")
            if t == "mouse_move":
                x = ev.get("x"); y = ev.get("y")
                if x is None or y is None:
                    continue
                self.mouse_ctrl.position = (x, y)

            elif t == "mouse_click":
                x = ev.get("x"); y = ev.get("y")
                if x is None or y is None:
                    continue
                self.mouse_ctrl.position = (x, y)
                btn = self.deserialize_mouse_button(ev.get("button", "Button.left"))
                if bool(ev.get("pressed", False)):
                    self.mouse_ctrl.press(btn)
                    pressed_mouse.add(btn)
                else:
                    self.mouse_ctrl.release(btn)
                    pressed_mouse.discard(btn)

            elif t == "mouse_scroll":
                x = ev.get("x"); y = ev.get("y")
                if x is None or y is None:
                    continue
                self.mouse_ctrl.position = (x, y)
                dx = int(ev.get("dx", 0))
                dy = int(ev.get("dy", 0))
                self.mouse_ctrl.scroll(dx, dy)

            elif t == "key_press":
                k = self.deserialize_key(ev.get("key"))
                if k is not None:
                    self.kb_ctrl.press(k)
                    pressed_keys.add(k)

            elif t == "key_release":
                k = self.deserialize_key(ev.get("key"))
                if k is not None:
                    self.kb_ctrl.release(k)
                    pressed_keys.discard(k)

    # ---------- serialization ----------
    def serialize_key(self, k):
        if isinstance(k, KeyCode):
            char = keycode_to_char(k)
            if char:
                return {"type": "char", "value": char}
            vk = getattr(k, "vk", None)
            if vk is not None:
                return {"type": "vk", "value": vk}
            return {"type": "unknown", "value": str(k)}
        elif isinstance(k, Key):
            return {"type": "key", "value": k.name}
        else:
            return {"type": "unknown", "value": str(k)}

    def deserialize_key(self, d):
        if not isinstance(d, dict):
            return None
        if d.get("type") == "char":
            v = d.get("value")
            if v is None:
                return None
            return KeyCode.from_char(v)
        if d.get("type") == "key":
            name = d.get("value")
            if not name:
                return None
            return getattr(Key, name, None)
        if d.get("type") == "vk":
            value = d.get("value")
            if value is None:
                return None
            return KeyCode.from_vk(value)
        return None

    def deserialize_mouse_button(self, s: str):
        if not isinstance(s, str):
            return MouseButton.left
        if "left" in s:
            return MouseButton.left
        if "right" in s:
            return MouseButton.right
        if "middle" in s:
            return MouseButton.middle
        return MouseButton.left

    # ---------- lifecycle ----------
    def on_close(self):
        try:
            self.save_config()
        except Exception:
            pass

        try:
            with self.hotkeys_lock:
                if self.hotkeys_manager:
                    self.hotkeys_manager.stop()
        except Exception:
            pass

        try:
            if self.recording:
                self.stop_recording()
        except Exception:
            pass

        # request stop playback; runner will cleanup if running
        self.playing = False

        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        root.call("tk", "scaling", 1.2)
    except Exception:
        pass
    MacroApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
