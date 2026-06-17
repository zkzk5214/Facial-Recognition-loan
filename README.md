# Human_Face (Deprecated)

> **This branch (`main`) contains legacy code and is no longer actively maintained.**
>
> Active development has moved to the following branches:
> - `ailoan`
> - `lanke`

## Overview

Face detection, alignment, recognition, and similarity comparison service. This was the original implementation before the codebase was refactored and split into separate service branches.

## Project Structure

```
Human_Face/
├── server.py              # Flask API server
├── deploy_dev.sh          # Dev deployment
├── deploy_stg.sh          # Staging deployment
├── deploy_pro.sh          # Production deployment
├── inferencer/
│   ├── blur.py            # Blur detection
│   ├── detector.py        # Face detection + alignment
│   ├── general.py         # YOLO post-processing utilities
│   ├── geo_check.py       # Face completeness validation
│   ├── hand_crafted.py    # Legacy bbox filtering (unused)
│   ├── quality.py         # Image quality detection
│   ├── recognizer.py      # Face embedding extraction
│   ├── scrfd.py           # SCRFD detector (unused)
│   └── yolo_detection.py  # YOLO inference
├── utils/
│   └── log.py             # Logger utility
├── resources/             # Model weights
└── unit_test/             # Tests
```

## Note

This code is preserved for reference only. For production deployments and new development, please switch to the `ailoan` or `lanke` branch.
