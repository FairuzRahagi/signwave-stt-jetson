# SignWave STT: Offline Indonesian Streaming Speech-to-Text on NVIDIA Jetson

Real-time, fully local Indonesian speech-to-text running on an NVIDIA Jetson Orin Nano, with the transcript pushed over MQTT to ESP32-driven TFT displays. It is the teacher-to-student speech subsystem of **SignWave**, a two-way assistive communication system for Deaf and Hard of Hearing (DHH) students, built as a final-year project (EL4091, Team 14) at Institut Teknologi Bandung.

> This repository contains **my subsystem only** (the STT engine and its evaluation tool). The sign-language (BISINDO) recognition subsystem and the ESP32 display firmware were built by teammates and are not included.

## Why local and streaming

Cloud speech services need a stable connection, cost money per use, add latency and send classroom audio off-site. This system runs entirely on the edge device and emits text incrementally while the teacher is still speaking, instead of waiting for a full sentence.

## System overview

```
 Wireless mic            Jetson Orin Nano                          Classroom displays
 (ESP32, 16 kHz PCM)     (also the Wi-Fi hotspot + MQTT broker)    (ESP32 + TFT LCD)
 ┌──────────────┐  UDP   ┌─────────────────────────────────┐ MQTT ┌────────────────┐
 │ audio capture├───────►│ chunking -> Sherpa-ONNX          ├─────►│ subscribe and   │
 └──────────────┘ :8080  │ Zipformer2 streaming recognizer  │ topic│ render text     │
                         │ + tkinter evaluation GUI         │ stt/ │ (several screens)│
                         └─────────────────────────────────┘ layar└────────────────┘
```

1. **Audio acquisition:** an ESP32 microphone node streams 16 kHz PCM over UDP to the Jetson (port 8080). If the node sends 48 kHz instead, the worker downsamples to 16 kHz.
2. **Inference:** audio is cut into streaming chunks (640 ms in the thesis design) and fed to a Sherpa-ONNX online recognizer with an Indonesian streaming Zipformer2 transducer (RNN-T). Tokens are emitted incrementally. The code uses CUDA when onnxruntime exposes it and falls back to CPU otherwise.
3. **Distribution:** finalized transcript segments are published to the MQTT topic `stt/layar`, which the ESP32 displays subscribe to. The Jetson acts as the broker.

## Results

Test material: a 150-word instructional passage that imitates classroom speech, read in quiet and noisy (60 dB) conditions. Requirements were WER <= 20 % and end-to-end latency <= 10 s.

| Metric | Quiet | Noisy (60 dB) | Requirement |
|---|---|---|---|
| Word Error Rate (average) | 6.49 % | 6.93 % | <= 20 % |
| Real-Time Factor (average, both conditions) | 0.15 | 0.15 | < 1.0 |
| Continuous operation | 135 minutes without failure or performance drop | 135 minutes without failure or performance drop | n/a |

Most transcription errors fall on the sibilant fricative /s/, whose high-frequency energy is hard for the model and microphone chain. MQTT distribution to the displays was verified functionally without noticeable delay.

### How the engine was chosen

Four implementation iterations were tested on the same passage:

| # | Hardware and model | Average WER | Max latency |
|---|---|---|---|
| 1 | Orange Pi 5 Pro, whisper.cpp (CPU, quantized GGML) | about 60 % | about 17 s |
| 2 | Orange Pi 5 Pro, Whisper RKNN (NPU) | about 45 % | about 12 s |
| 3 | Jetson Orin Nano, Faster-Whisper (CUDA, CTranslate2, FP16/INT8) | about 12 % | about 2.5 s |
| 4 | Jetson Orin Nano, **Sherpa-ONNX Zipformer2 streaming** (chunk 640 ms) | **about 6.5 %** | **about 0.2 s** |

Iterations 3 and 4 met the requirements. Iteration 4 was chosen because it is far lighter on the GPU, at the cost of seeing only a limited left context (the model used is `encoder-iter-100000-avg-15-chunk-32-left-256`). The Jetson was run in MAXN power mode (25 W) with clocks locked and monitored with `tegrastats` to rule out thermal throttling.

The full methodology is described in the thesis (written in Indonesian, not included in this repository).

## Files

| File | Purpose |
|---|---|
| `GUI_UDP.py` | Tkinter application: receives UDP audio, runs the streaming recognizer, shows live text, computes WER/RTF against a reference text, and publishes results to MQTT |
| `requirements.txt` | Python dependencies |

## Setup

1. Install dependencies (on Jetson use the `onnxruntime-gpu` build that matches your JetPack):

   ```bash
   pip install -r requirements.txt
   ```

2. Download the Indonesian streaming Zipformer2 model for Sherpa-ONNX (MIT licensed) and place it so that this layout exists:

   ```
   models/sherpa-onnx-streaming-zipformer2-id/
     tokens.txt
     encoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx
     decoder-iter-100000-avg-15-chunk-32-left-256.int8.onnx
     joiner-iter-100000-avg-15-chunk-32-left-256.int8.onnx
   ```

   The model is not included in this repository because of its size (hundreds of MB). Its published WER, from its model card, is 9.81 % on average across YODAS, Common Voice, FLEURS and LibriVox-ID.

3. Edit the configuration block at the top of `GUI_UDP.py`:

   | Constant | Default | Meaning |
   |---|---|---|
   | `MODEL_DIR` | `./models/sherpa-onnx-streaming-zipformer2-id` | Model folder |
   | `UDP_PORT` | `8080` | Must match the ESP32 transmitter |
   | `AUDIO_SAMPLERATE` | `16000` | Sample rate sent by the ESP32 |
   | `MQTT_BROKER` | `10.42.0.1` | Broker address (the Jetson hotspot address in my setup) |
   | `MQTT_TOPIC` | `stt/layar` | Topic the displays subscribe to |

4. Make sure an MQTT broker (for example Mosquitto) is running, then start the app:

   ```bash
   python GUI_UDP.py
   ```

## Tech stack

Python, Sherpa-ONNX, ONNX Runtime (CUDA), Zipformer2 RNN-T, UDP sockets, MQTT (paho), Tkinter, jiwer for WER, NVIDIA Jetson Orin Nano, ESP32.

## Acknowledgements

Speech model: Sherpa-ONNX streaming Zipformer2 for Indonesian (MIT license), trained on YODAS2, Common Voice, FLEURS and LibriVox-ID. Inference engine: [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx).
