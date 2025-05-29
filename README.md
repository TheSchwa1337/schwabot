# Schwabot System

A highly modular, mathematically rigorous, and fault-tolerant core for trading/automation with recursive, hash-driven, and CPU-synchronized logic.

## Features

- Core math library with tick sequence, price delta, profit ratio calculations
- RITTLE-GEMM ring value schema for tracking key metrics
- Net stop loss pattern value book with state machine
- MEMKEY CPU-sync math logic with hash-based triggers
- Desync correction code (DCC) for drift detection
- Bound collision hashing system (BCHS) for entropy scoring
- Safe temperature-based throttling and resource management

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/schwabot.git
cd schwabot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Hash Cycles

Use the CLI script to run hash cycles with automatic throttling:

```bash
python scripts/run_hash_cycles.py --input-file data.txt --batch-size 32
```

Options:
- `--min-workers`: Minimum worker threads (default: 1)
- `--max-workers`: Maximum worker threads (default: CPU count)
- `--batch-size`: Hash operation batch size (default: 16)
- `--poll-interval`: Throttle check interval in seconds (default: 0.5)
- `--input-file`: Input file to hash (default: stdin)
- `--chunk-size`: Input chunk size in bytes (default: 1024)

### Monitoring System Metrics

Run the sensor polling script to monitor system metrics:

```bash
./scripts/poll_sensors.sh
```

This will log CPU/GPU temperatures, loads, and memory usage to `/var/log/schwabot_metrics.log`.

## Project Structure

```
schwabot/
├── core/                    # Core math and hash logic
│   ├── mathlib.py          # Core math library
│   ├── rittle_gemm.py      # Ring value schema
│   └── schwabot_stop.py    # Stop loss patterns
├── extensions/             # Extended features
│   └── mathlib_v2.py       # Extended math library
├── scaling/               # Resource management
│   ├── monitor_portals.py  # System monitoring
│   ├── throttle_manager.py # Temperature throttling
│   └── hash_dispatcher.py  # Hash cycle dispatch
├── tests/                 # Unit tests
├── examples/              # Example notebooks
├── scripts/               # CLI tools
│   ├── run_hash_cycles.py # Hash cycle runner
│   └── poll_sensors.sh    # Metrics monitor
└── docs/                  # Documentation
```

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Adding New Features

1. Add core functionality in `core/`
2. Add extended features in `extensions/`
3. Add resource management in `scaling/`
4. Add tests in `tests/`
5. Update documentation in `docs/`

## License

MIT License - see LICENSE file for details 