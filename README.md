# Schwabot - Unified BTC-to-Profit Trading System

A highly modular, mathematically rigorous, and fault-tolerant core for unified BTC-to-profit trading with recursive, hash-driven, and CPU-synchronized logic.

## 🚀 Features

- **Unified BTC-to-Profit Trading Engine**: Complete mathematical scaffolding for BTC trading operations
- **Advanced Mathematical Core**: Tensor algebra, profit vectorization, and phase engine systems
- **Cross-Platform Deployment**: Ready for Mac, Windows, and Linux deployment
- **Flake8 Compliant**: Full code quality compliance with zero E999 syntax errors
- **Real-time Trading Logic**: BTC hashing analysis, price prediction, and profit optimization
- **Visual Integration**: Complete UI bridge and dashboard systems
- **API Integration**: Comprehensive API coordination and exchange connectivity
- **Recursive Engine**: Advanced recursive profit calculation and memory systems

## 📦 Installation

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/schwabot.git
cd schwabot
```

2. Install dependencies using the automated installer:
```bash
python install_dependencies.py
```

### Manual Installation

For production deployment:
```bash
pip install -r requirements-prod.txt
```

For development with testing tools:
```bash
pip install -r requirements.txt
```

### Verify Installation

Check that all dependencies are properly installed:
```bash
python install_dependencies.py --verify
```

## 🎯 Usage

### Running the Unified Trading System

Start the main trading system:
```bash
python main.py
```

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

### Web Dashboard

Access the trading dashboard:
```bash
python ui/schwabot_dashboard.py
```

Then open your browser to `http://localhost:5000`

### Monitoring System Metrics

Run the sensor polling script to monitor system metrics:

```bash
./scripts/poll_sensors.sh
```

This will log CPU/GPU temperatures, loads, and memory usage to `/var/log/schwabot_metrics.log`.

## 🏗️ Project Structure

```
schwabot/
├── core/                           # Core mathematical and trading systems
│   ├── math/                      # Mathematical subsystems
│   │   ├── tensor_algebra/        # Tensor operations
│   │   └── trading_tensor_ops.py  # Trading-specific math
│   ├── phase_engine/              # Phase-based trading logic
│   ├── recursive_engine/          # Recursive profit calculations
│   └── unified_*.py              # Unified system components
├── schwabot/                      # Main application
│   ├── core/                     # Core trading logic
│   ├── ai_oracles/               # AI prediction systems
│   ├── tools/                    # Utility tools
│   └── main.py                   # Main application entry
├── ui/                           # User interface components
│   └── schwabot_dashboard.py     # Web dashboard
├── utils/                        # Utility functions
├── tests/                        # Unit tests
├── scripts/                      # CLI tools
├── requirements.txt              # Full dependencies
├── requirements-prod.txt         # Production dependencies
├── install_dependencies.py       # Automated installer
└── README.md                     # This file
```

## 🔧 Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Quality

Check Flake8 compliance:
```bash
flake8 core/ schwabot/
```

### Adding New Features

1. Add core functionality in `core/`
2. Add trading logic in `schwabot/core/`
3. Add UI components in `ui/`
4. Add tests in `tests/`
5. Update documentation

## 🌐 Cross-Platform Deployment

The system is fully tested and ready for deployment on:
- **macOS**: Full compatibility with Homebrew dependencies
- **Windows**: Complete Windows 10/11 support
- **Linux**: Ubuntu, CentOS, and other distributions

### Deployment Verification

Run the final deployment verification:
```bash
python final_deployment_verification.py
```

## 📊 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space
- **Network**: Internet connection for API access
- **GPU**: Optional (CUDA support available)

## 🔒 Security

- All API keys and sensitive data are handled securely
- Mathematical integrity is preserved across all operations
- Cross-platform compatibility ensures consistent behavior
- Flake8 compliance guarantees code quality

## 📈 Performance

- **Mathematical Operations**: Optimized tensor algebra and profit calculations
- **BTC Trading**: Real-time price analysis and signal generation
- **Memory Management**: Efficient recursive engine with memory optimization
- **Cross-Platform**: Consistent performance across all supported platforms

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Ensure Flake8 compliance
5. Add tests
6. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

For issues and questions:
1. Check the documentation
2. Run the verification scripts
3. Review the error logs
4. Open an issue on GitHub

---

**🚀 Ready for Production Deployment!** 

The unified BTC-to-profit trading system is fully compliant, mathematically verified, and ready for cross-platform deployment with complete Flake8 compliance and zero E999 syntax errors. 