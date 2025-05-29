#!/bin/bash
# Poll system sensors and log metrics

# Configuration
LOG_FILE="/var/log/schwabot_metrics.log"
POLL_INTERVAL=5  # seconds
MAX_LOG_SIZE=100M  # Rotate log at this size

# Ensure log directory exists
mkdir -p "$(dirname "$LOG_FILE")"

# Function to get CPU metrics
get_cpu_metrics() {
    # CPU temperature
    if [ -f /sys/class/thermal/thermal_zone0/temp ]; then
        cpu_temp=$(cat /sys/class/thermal/thermal_zone0/temp)
        cpu_temp=$((cpu_temp / 1000))
    else
        cpu_temp="N/A"
    fi
    
    # CPU load
    cpu_load=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}')
    
    echo "CPU: ${cpu_temp}°C, Load: ${cpu_load}%"
}

# Function to get GPU metrics
get_gpu_metrics() {
    if command -v nvidia-smi &> /dev/null; then
        # NVIDIA GPU
        gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits)
        gpu_load=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
        echo "GPU: ${gpu_temp}°C, Load: ${gpu_load}%"
    else
        echo "GPU: N/A"
    fi
}

# Function to get memory metrics
get_memory_metrics() {
    free -h | grep Mem | awk '{print "Memory: " $3 "/" $2 " used"}'
}

# Function to rotate log if needed
rotate_log() {
    if [ -f "$LOG_FILE" ]; then
        size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE")
        if [ "$size" -gt "$MAX_LOG_SIZE" ]; then
            mv "$LOG_FILE" "${LOG_FILE}.1"
            touch "$LOG_FILE"
        fi
    fi
}

# Main polling loop
echo "Starting sensor polling..."
echo "Logging to: $LOG_FILE"

while true; do
    # Get timestamp
    timestamp=$(date "+%Y-%m-%d %H:%M:%S")
    
    # Get metrics
    cpu_metrics=$(get_cpu_metrics)
    gpu_metrics=$(get_gpu_metrics)
    mem_metrics=$(get_memory_metrics)
    
    # Log metrics
    echo "[$timestamp] $cpu_metrics | $gpu_metrics | $mem_metrics" >> "$LOG_FILE"
    
    # Check log rotation
    rotate_log
    
    # Wait for next poll
    sleep "$POLL_INTERVAL"
done 