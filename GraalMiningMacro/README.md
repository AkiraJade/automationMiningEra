# Graal Mining Macro

A modular, vision-driven, mining-focused automation application for **Graal Online Era**.

## Architecture & Principles
- **Perception-Driven**: Continuous observation ➔ state evaluation ➔ decision ➔ action ➔ verification ➔ recovery.
- **Safety First**: F12 emergency stop listener immediately halts all actions, releases keys, and locks execution.
- **Live Game Display**: Embedded real-time capture display centered in the PySide6 UI.
- **Observation Mode**: Testable passive monitoring mode without dispatching game keypresses.

## Project Structure
```
GraalMiningMacro/
├── main.py                     # Entry point
├── requirements.txt            # Python dependencies
├── app/
│   ├── core/                   # Config, logger, events bus
│   ├── window/                 # Win32 HWND detection & tracking
│   ├── capture/                # MSS live capture & worker thread
│   ├── coordinates/            # Screen/Client/Normalized coordinate mapper
│   ├── input/                  # PyDirectInput keyboard/mouse & F12 safety
│   ├── vision/                 # OpenCV HSV color, template & YOLO detectors
│   ├── mining/                 # Mining state machine, perception & targets
│   └── gui/                    # PySide6 MainWindow, Dashboard & GamePreview
└── tests/                      # Pytest automated test suite
```

## Setup & Running
```bash
pip install -r requirements.txt
python main.py
```
To run automated tests:
```bash
python -m pytest tests/ -v
```
