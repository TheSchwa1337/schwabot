import os
import sys
import json
import time
import logging
from datetime import datetime

# Path setup
SCHWABOT_DIR = os.path.dirname(os.path.abspath(__file__))
R1_INSTRUCTIONS = os.path.join(SCHWABOT_DIR, '../r1/instructions/next_command.json')
TRADE_ROUTER = os.path.join(SCHWABOT_DIR, 'trade_router.py')
LOG_FILE = os.path.join(SCHWABOT_DIR, 'logs', 'instruction_listener.log')

logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

POLL_INTERVAL = 2  # seconds

def main():
    print('[InstructionListener] Starting command listener...')
    while True:
        if os.path.exists(R1_INSTRUCTIONS):
            try:
                with open(R1_INSTRUCTIONS, 'r') as f:
                    cmd = json.load(f)
                # Remove the command file to prevent re-execution
                os.remove(R1_INSTRUCTIONS)
                # Route to trade_router.py
                direction = cmd.get('direction')
                percent = cmd.get('percent')
                trigger_hash = cmd.get('trigger_hash')
                cycle_key = cmd.get('cycle_key')
                args = [sys.executable, TRADE_ROUTER, '--direction', direction]
                if percent is not None:
                    args += ['--percent', str(percent)]
                if trigger_hash:
                    args += ['--trigger-hash', trigger_hash]
                if cycle_key:
                    args += ['--cycle-key', cycle_key]
                logging.info(f'Executing trade_router.py with args: {args}')
                os.system(' '.join(args))
            except Exception as e:
                logging.error(f'Error processing instruction: {e}')
        time.sleep(POLL_INTERVAL)

if __name__ == '__main__':
    main() 