# ESP32-WiFi-Classifier
On-device Wi-Fi signal pattern classification using ESP32 and TensorFlow Lite Micro, with web dashboard
-This project demonstrates real-time Wi-Fi environment classification using an ESP32 and a TinyML model (TensorFlow Lite Micro).

The ESP32 scans nearby Wi-Fi networks, normalizes RSSI values, performs on-device inference, and classifies the environment into predefined classes. Results are available via Serial Monitor and a web dashboard.

## Features
- Wi-Fi scanning using ESP32
- RSSI normalization
- TinyML inference (TensorFlow Lite Micro)
- Readable signal strength labels
- Optional web dashboard
- Fully offline ML inference (edge AI)

## Classes
| Label | Meaning |
|------|--------|
| Class_A | Strong-dominant Wi-Fi environment |
| Class_B | Medium / mixed signal environment |
| Class_C | Weak / noisy Wi-Fi environment |

## Hardware
- ESP32 Dev Module

## Software
- Arduino IDE
- TensorFlow Lite for Microcontrollers

## How to Run
1. Open `esp32_wifi_classifier.ino`
2. Install ESP32 board support
3. Upload to ESP32
4. Open Serial Monitor (115200 baud)
5. Optionally open browser at ESP32 IP for dashboard

## Applications
- Indoor localization
- Wi-Fi quality monitoring
- Smart building analytics
- Edge AI demonstration
