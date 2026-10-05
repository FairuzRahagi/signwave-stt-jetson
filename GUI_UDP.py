import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import queue
import jiwer
import os
import threading
import string
import sys
import numpy as np
import socket
import soundfile as sf
import sherpa_onnx
import onnxruntime as ort
import paho.mqtt.client as mqtt
import time

# ============================================================
# KONFIGURASI - Sesuaikan dengan setup kamu
# ============================================================
MODEL_DIR = "./models/sherpa-onnx-streaming-zipformer2-id"

# --- UDP Mic Configuration (replaces sounddevice) ---
UDP_IP          = "0.0.0.0"   # Listen on all adapters
UDP_PORT        = 8080         # Must match ESP32 transmitter port
UDP_BUFFER_SIZE = 1024         # Match ESP32 packet size
AUDIO_SAMPLERATE = 16000       # ESP32 sends 16kHz PCM directly
#   If your ESP32 sends 48kHz instead, change to 48000 and
#   the worker will auto-downsample to 16kHz for Sherpa.

# --- MQTT (send final transcript to ESP32 display) ---
MQTT_BROKER = "10.42.0.1"
MQTT_TOPIC  = "stt/layar"
# ============================================================


class SherpaEvalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sherpa-ONNX Evaluation Tool (UDP Mic, STT, WER & RTF)")
        self.root.geometry("1000x780")

        self.is_recording      = False
        self.live_worker_thread = None
        self.ui_queue          = queue.Queue()
        self.live_segments     = []
        self.session_delays    = []
        self.recognizer        = None
        self.mqtt_client       = None

        self._init_mqtt()
        self._init_sherpa_engine()
        self._setup_ui()

    # ==========================================
    # INIT
    # ==========================================
    def _init_mqtt(self):
        self.mqtt_client = mqtt.Client()
        try:
            self.mqtt_client.connect(MQTT_BROKER, 1883, 60)
            self.mqtt_client.loop_start()
            print("[INFO] Terhubung ke MQTT Broker Lokal (Siap kirim ke ESP32)")
        except Exception as e:
            print(f"[WARNING] Gagal konek ke MQTT: {e}. Pastikan mosquitto berjalan.")

    def _init_sherpa_engine(self):
        providers   = ort.get_available_providers()
        alat_proses = "cuda" if "CUDAExecutionProvider" in providers else "cpu"
        try:
            self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
                tokens  = f"{MODEL_DIR}/tokens.txt",
                encoder = f"{MODEL_DIR}/encoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx",
                decoder = f"{MODEL_DIR}/decoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx",
                joiner  = f"{MODEL_DIR}/joiner-iter-100000-avg-15-chunk-32-left-256.int8.onnx",
                num_threads = 2,
                sample_rate = 16000,
                feature_dim = 80,
                provider    = alat_proses,
            )
            print(f"[INFO] Model Sherpa Zipformer2 dimuat via: {alat_proses.upper()}")
        except Exception as e:
            messagebox.showerror("Init Error", f"Gagal memuat model Sherpa:\n{e}")

    def normalize_text(self, text):
        if not text:
            return ""
        text = text.lower()
        text = text.translate(str.maketrans("", "", string.punctuation))
        return " ".join(text.split())

    # ==========================================
    # UI SETUP
    # ==========================================
    def _setup_ui(self):
        self.notebook = ttk.Notebook(self.root)

        # ── TAB 1: LIVE CAPTIONING ──────────────────────────
        self.tab_live = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_live, text="Mode 1: Live Captioning (UDP)")

        ctrl = ttk.LabelFrame(self.tab_live, text="Kontrol", padding=10)
        ctrl.pack(fill="x", pady=5)

        # UDP port display
        ttk.Label(ctrl, text=f"Listening UDP Port: {UDP_PORT}", font=("Arial", 10)).pack(side="left", padx=(0, 20))

        self.live_start_btn = ttk.Button(ctrl, text="Start Live STT", command=self.start_live_caption)
        self.live_start_btn.pack(side="left", padx=5)

        self.live_stop_btn = ttk.Button(ctrl, text="Stop & Evaluasi", command=self.stop_live_caption, state="disabled")
        self.live_stop_btn.pack(side="left", padx=5)

        ref_frame = ttk.LabelFrame(self.tab_live, text="Teks Referensi (Ground Truth) - Opsional", padding=10)
        ref_frame.pack(fill="x", pady=5)
        self.live_ref_text = ScrolledText(ref_frame, height=5, font=("Arial", 14), wrap=tk.WORD)
        self.live_ref_text.pack(fill="x", expand=True)

        result_frame = ttk.LabelFrame(self.tab_live, text="Hasil Transkripsi Real-time", padding=10)
        result_frame.pack(fill="both", expand=True, pady=5)

        # Top info row: partial (left) | latency (right)
        info_row = ttk.Frame(result_frame)
        info_row.pack(fill="x", pady=(0, 5))

        self.live_partial_label = ttk.Label(
            info_row, text="Mengetik: ...", font=("Arial", 12, "italic"), foreground="gray"
        )
        self.live_partial_label.pack(side="left")

        self.live_latency_label = ttk.Label(
            info_row, text="Avg Chunk Delay: 0.0 ms", font=("Arial", 10, "bold"), foreground="#0052cc"
        )
        self.live_latency_label.pack(side="right")

        self.live_result_text = ScrolledText(
            result_frame, height=10, font=("Arial", 14), wrap=tk.WORD, state="disabled"
        )
        self.live_result_text.pack(fill="both", expand=True)

        self.live_status_label = ttk.Label(result_frame, text="Status: Idle")
        self.live_status_label.pack(anchor="w", pady=(5, 0))

        # ── TAB 2: FILE EVALUATION ──────────────────────────
        self.tab_file = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.tab_file, text="Mode 2: Evaluasi File Audio")

        input_frame = ttk.LabelFrame(self.tab_file, text="Input Data", padding=10)
        input_frame.pack(fill="x", pady=5)

        self.file_audio_btn = ttk.Button(input_frame, text="1. Upload File Audio (.wav)", command=self.load_audio_file)
        self.file_audio_btn.pack(fill="x", pady=2)
        self.file_audio_label = ttk.Label(input_frame, text="File: (Belum ada)")
        self.file_audio_label.pack(anchor="w", padx=5)
        self.audio_file_path = ""

        ttk.Label(input_frame, text="2. Masukkan Teks Referensi (Ground Truth):").pack(anchor="w", pady=(10, 0))
        self.file_ref_text = ScrolledText(input_frame, height=5, wrap=tk.WORD)
        self.file_ref_text.pack(fill="x", expand=True, pady=2)

        self.file_eval_btn = ttk.Button(
            self.tab_file, text="3. Mulai Evaluasi File", command=self.start_file_evaluation
        )
        self.file_eval_btn.pack(pady=10)

        self.file_result_text = ScrolledText(self.tab_file, height=15, wrap=tk.WORD, state="disabled")
        self.file_result_text.pack(fill="both", expand=True, pady=5)

        self.notebook.pack(expand=True, fill="both")

    # ==========================================
    # MODE 1 — LIVE CAPTIONING VIA UDP
    # ==========================================
    def start_live_caption(self):
        if not self.recognizer:
            messagebox.showerror("Error", "Model Sherpa belum siap!")
            return

        self.live_segments  = []
        self.session_delays = []

        self.live_result_text.config(state="normal")
        self.live_result_text.delete("1.0", tk.END)
        self.live_result_text.config(state="disabled")
        self.live_partial_label.config(text="Mengetik: ...")
        self.live_latency_label.config(text="Avg Chunk Delay: 0.0 ms")
        self.live_status_label.config(text="Status: Menunggu paket UDP...")

        self.live_start_btn.config(state="disabled")
        self.live_stop_btn.config(state="normal")

        self.is_recording = True
        self.live_worker_thread = threading.Thread(target=self._live_udp_worker, daemon=True)
        self.live_worker_thread.start()

        self.root.after(100, self.check_live_queue)

    def _live_udp_worker(self):
        """
        Receives raw 16-bit mono PCM packets over UDP,
        feeds them into Sherpa-ONNX, tracks chunk decode latency.
        If AUDIO_SAMPLERATE != 16000, downsamples before feeding.
        """
        stream = self.recognizer.create_stream()

        # Open UDP socket (mirrors your tested receiver script)
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # 1 MB buffer
            sock.bind((UDP_IP, UDP_PORT))
            sock.settimeout(0.5)   # non-blocking poll so we can check is_recording flag
        except Exception as e:
            self.ui_queue.put({"type": "error", "text": f"UDP bind failed on port {UDP_PORT}:\n{e}"})
            return

        print(f"[UDP] Listening on {UDP_IP}:{UDP_PORT} ...")

        last_text        = ""
        last_change_time = time.time()
        SILENCE_TIMEOUT  = 1.5

        chunk_total_time = 0.0
        chunk_count      = 0

        downsample_ratio = AUDIO_SAMPLERATE // 16000   # e.g. 3 if 48kHz, 1 if 16kHz

        try:
            while self.is_recording:
                # ── Receive UDP packet ──────────────────────────
                try:
                    data, addr = sock.recvfrom(UDP_BUFFER_SIZE)
                except socket.timeout:
                    # No packet yet — still check silence timeout below
                    data = None

                if data:
                    # Convert raw bytes → float32 samples
                    # ESP32 sends 16-bit signed little-endian PCM
                    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    # Downsample if ESP32 sends at a rate higher than 16kHz
                    if downsample_ratio > 1:
                        samples = samples[::downsample_ratio]

                    stream.accept_waveform(16000, samples)

                # ── Decode all ready chunks & measure latency ──
                processed = False
                while self.recognizer.is_ready(stream):
                    t0 = time.perf_counter()
                    self.recognizer.decode_stream(stream)
                    t1 = time.perf_counter()

                    chunk_total_time += (t1 - t0)
                    chunk_count      += 1
                    processed         = True

                if processed and chunk_count > 0:
                    avg_ms = (chunk_total_time / chunk_count) * 1000
                    self.ui_queue.put({"type": "latency", "value": avg_ms})
                    self.session_delays.append(avg_ms)
                    if chunk_count > 10:          # rolling reset keeps data fresh
                        chunk_total_time = 0.0
                        chunk_count      = 0

                # ── Partial / final text logic ─────────────────
                text = self.recognizer.get_result(stream)

                if text and text != last_text:
                    self.ui_queue.put({"type": "partial", "text": text})
                    last_text        = text
                    last_change_time = time.time()

                if last_text and (time.time() - last_change_time) > SILENCE_TIMEOUT:
                    self.ui_queue.put({"type": "final", "text": last_text})
                    self.recognizer.reset(stream)
                    last_text        = ""
                    last_change_time = time.time()

                if self.recognizer.is_endpoint(stream):
                    if text:
                        self.ui_queue.put({"type": "final", "text": text})
                    self.recognizer.reset(stream)
                    last_text        = ""
                    last_change_time = time.time()

        except Exception as e:
            self.ui_queue.put({"type": "error", "text": str(e)})
        finally:
            sock.close()
            print("[UDP] Socket closed.")

    def check_live_queue(self):
        try:
            while True:
                data = self.ui_queue.get_nowait()

                if data["type"] == "error":
                    messagebox.showerror("Error", data["text"])
                    self.is_recording = False
                    self.live_start_btn.config(state="normal")
                    self.live_stop_btn.config(state="disabled")
                    self.live_status_label.config(text="Status: Error.")
                    return

                elif data["type"] == "latency":
                    self.live_latency_label.config(text=f"Avg Chunk Delay: {data['value']:.1f} ms")
                    # Update status to show we are actively receiving
                    self.live_status_label.config(text="Status: Merekam via UDP...")

                elif data["type"] == "partial":
                    self.live_partial_label.config(text=f"Mengetik: {data['text']}")

                elif data["type"] == "final":
                    self.live_partial_label.config(text="Mengetik: ...")
                    final_text = data["text"].strip()
                    self.live_segments.append(final_text)

                    self.live_result_text.config(state="normal")
                    self.live_result_text.insert(tk.END, f"{final_text} ")
                    self.live_result_text.see(tk.END)
                    self.live_result_text.config(state="disabled")

                    # Send final transcript to ESP32 display via MQTT
                    if self.mqtt_client and final_text:
                        try:
                            self.mqtt_client.publish(MQTT_TOPIC, final_text)
                        except Exception as e:
                            print(f"[ERROR] Gagal kirim MQTT: {e}")

        except queue.Empty:
            pass

        if self.is_recording:
            self.root.after(100, self.check_live_queue)

    def stop_live_caption(self):
        self.is_recording = False

        if self.live_worker_thread and self.live_worker_thread.is_alive():
            self.live_worker_thread.join(timeout=2.0)

        self.live_start_btn.config(state="normal")
        self.live_stop_btn.config(state="disabled")
        self.live_status_label.config(text="Status: Dihentikan.")
        self.live_partial_label.config(text="")

        raw_reference = self.live_ref_text.get("1.0", tk.END).strip()
        raw_hypothesis = " ".join(self.live_segments).strip()

        overall_avg_delay = (
            sum(self.session_delays) / len(self.session_delays) if self.session_delays else 0
        )

        if not raw_reference or not raw_hypothesis:
            msg = (
                f"Sesi selesai. Teks Referensi kosong, melewatkan WER.\n"
                f"Rata-rata Delay Decoding: {overall_avg_delay:.2f} ms"
            )
            messagebox.showinfo("Info Sesi", msg)
            return

        reference  = self.normalize_text(raw_reference)
        hypothesis = self.normalize_text(raw_hypothesis)

        try:
            wer = jiwer.wer(reference, hypothesis)
            mer = jiwer.mer(reference, hypothesis)
            wil = jiwer.wil(reference, hypothesis)

            report = (
                f"--- LAPORAN LIVE (UDP + SHERPA-ONNX) ---\n"
                f"WER               : {wer*100:.2f}%\n"
                f"MER               : {mer*100:.2f}%\n"
                f"WIL               : {wil*100:.2f}%\n"
                f"Avg Chunk Delay   : {overall_avg_delay:.2f} ms\n"
                f"-----------------------------------------\n"
                f"Ref: {reference}\n"
                f"Hyp: {hypothesis}\n"
            )
            self._show_report_window(report)
            self.live_status_label.config(
                text=f"Status: Selesai. WER: {wer*100:.2f}% | Delay: {overall_avg_delay:.1f}ms"
            )
        except Exception as e:
            messagebox.showerror("Error WER", f"Gagal menghitung WER: {e}")

    # ==========================================
    # MODE 2 — FILE EVALUATION
    # ==========================================
    def load_audio_file(self):
        path = filedialog.askopenfilename(filetypes=[("WAV Audio", "*.wav")])
        if path:
            self.audio_file_path = path
            self.file_audio_label.config(text=f"File: {os.path.basename(path)}")

    def start_file_evaluation(self):
        ref = self.file_ref_text.get("1.0", tk.END).strip()
        if not self.audio_file_path or not ref:
            messagebox.showerror("Error", "File audio (.wav) dan Teks Referensi wajib diisi.")
            return

        self.file_result_text.config(state="normal")
        self.file_result_text.delete("1.0", tk.END)
        self.file_result_text.insert("1.0", "Memproses file dengan Sherpa-ONNX...\n")
        self.file_result_text.config(state="disabled")
        self.file_eval_btn.config(state="disabled")

        threading.Thread(target=self._file_eval_worker, args=(ref,), daemon=True).start()

    def _file_eval_worker(self, raw_reference):
        try:
            audio_data, sample_rate = sf.read(self.audio_file_path, dtype="float32")

            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)

            if sample_rate != 16000:
                try:
                    import scipy.signal as scipy_signal
                    audio_data = scipy_signal.resample_poly(
                        audio_data, 16000, sample_rate
                    ).astype(np.float32)
                    self.root.after(0, lambda: self._append_file_log(
                        f"[INFO] Audio di-resample dari {sample_rate}Hz ke 16000Hz\n"
                    ))
                except ImportError:
                    self.root.after(0, lambda: self._append_file_log(
                        f"[WARNING] scipy tidak tersedia. File {sample_rate}Hz mungkin kurang akurat.\n"
                    ))

            stream = self.recognizer.create_stream()
            stream.accept_waveform(16000, audio_data)

            tail_padding = np.zeros(int(0.3 * 16000), dtype=np.float32)
            stream.accept_waveform(16000, tail_padding)

            # RTF measurement (from Code 1)
            t_start = time.perf_counter()
            while self.recognizer.is_ready(stream):
                self.recognizer.decode_stream(stream)
            t_end = time.perf_counter()

            processing_time = t_end - t_start
            audio_duration  = len(audio_data) / 16000.0
            rtf             = processing_time / audio_duration if audio_duration > 0 else 0

            raw_hypothesis = self.recognizer.get_result(stream)
            reference      = self.normalize_text(raw_reference)
            hypothesis     = self.normalize_text(raw_hypothesis)

            wer = jiwer.wer(reference, hypothesis) * 100
            mer = jiwer.mer(reference, hypothesis) * 100
            wil = jiwer.wil(reference, hypothesis) * 100

            result_str = (
                f"\n[SELESAI]\n"
                f"Hasil Transkripsi : {raw_hypothesis}\n\n"
                f"WER               : {wer:.2f}%\n"
                f"MER               : {mer:.2f}%\n"
                f"WIL               : {wil:.2f}%\n"
                f"Real-Time Factor  : {rtf:.3f}  "
                f"(Proses: {processing_time:.2f}s | Durasi: {audio_duration:.2f}s)\n"
                f"\nRef: {reference}\n"
                f"Hyp: {hypothesis}\n"
            )
            self.root.after(0, lambda: self._append_file_log(result_str))

        except Exception as e:
            self.root.after(0, lambda: self._append_file_log(f"\nERROR: {e}\n"))
        finally:
            self.root.after(0, lambda: self.file_eval_btn.config(state="normal"))

    def _append_file_log(self, text):
        self.file_result_text.config(state="normal")
        self.file_result_text.insert(tk.END, text)
        self.file_result_text.see(tk.END)
        self.file_result_text.config(state="disabled")

    def _show_report_window(self, text):
        win = tk.Toplevel(self.root)
        win.title("Laporan Evaluasi")
        win.geometry("620x420")
        st = ScrolledText(win, font=("Courier", 11))
        st.pack(fill="both", expand=True, padx=10, pady=10)
        st.insert(tk.END, text)
        st.config(state="disabled")


if __name__ == "__main__":
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc")
    except ImportError:
        root = tk.Tk()

    app = SherpaEvalApp(root)
    root.mainloop()