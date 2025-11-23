#!/usr/bin/env python3
"""
Script to monitor and log GPU and CPU usage over time.
Logs usage to a file and calculates average usage when the script is terminated.
"""

import time
import signal
import sys
from datetime import datetime
from pathlib import Path
import psutil

# Try to import GPU monitoring libraries
try:
    import pynvml
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("Warning: pynvml not available. GPU monitoring will be disabled.")
    print("Install with: pip install nvidia-ml-py")

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class ComputeBenchmark:
    def __init__(self, log_file=None, interval=1.0):
        """
        Initialize the compute benchmark monitor.
        
        Args:
            log_file: Path to log file. If None, generates timestamped filename.
            interval: Sampling interval in seconds (default: 1.0)
        """
        self.interval = interval
        self.running = True
        self.cpu_readings = []
        self.gpu_readings = []
        self.timestamps = []
        
        # Setup log file
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"compute_benchmark_{timestamp}.log"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU monitoring if available
        self.gpu_available = False
        if GPU_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                print(f"✅ GPU monitoring enabled (NVIDIA GPU detected)")
            except Exception as e:
                print(f"⚠️  GPU monitoring unavailable: {e}")
        elif GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if len(gpus) > 0:
                    self.gpu_available = True
                    print(f"✅ GPU monitoring enabled (GPUtil, {len(gpus)} GPU(s) detected)")
            except Exception as e:
                print(f"⚠️  GPU monitoring unavailable: {e}")
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"📊 Starting compute benchmark monitoring...")
        print(f"📝 Logging to: {self.log_file}")
        print(f"⏱️  Sampling interval: {self.interval}s")
        print(f"🛑 Press Ctrl+C to stop and view averages\n")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.running = False
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage."""
        return psutil.cpu_percent(interval=None)
    
    def get_gpu_usage(self):
        """Get current GPU usage percentage."""
        if not self.gpu_available:
            return None
        
        try:
            if GPU_AVAILABLE:
                # Use pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                return util.gpu
            elif GPUTIL_AVAILABLE:
                # Use GPUtil
                gpus = GPUtil.getGPUs()
                if len(gpus) > 0:
                    return gpus[0].load * 100
        except Exception as e:
            print(f"⚠️  Error reading GPU usage: {e}")
            return None
        
        return None
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage percentage."""
        if not self.gpu_available:
            return None
        
        try:
            if GPU_AVAILABLE:
                # Use pynvml
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                return (mem_info.used / mem_info.total) * 100
            elif GPUTIL_AVAILABLE:
                # Use GPUtil
                gpus = GPUtil.getGPUs()
                if len(gpus) > 0:
                    return gpus[0].memoryUtil * 100
        except Exception as e:
            print(f"⚠️  Error reading GPU memory: {e}")
            return None
        
        return None
    
    def log_reading(self, timestamp, cpu_usage, gpu_usage, gpu_memory):
        """Log a single reading to file and store in memory."""
        # Format log line
        gpu_str = f"{gpu_usage:.2f}%" if gpu_usage is not None else "N/A"
        gpu_mem_str = f"{gpu_memory:.2f}%" if gpu_memory is not None else "N/A"
        log_line = f"{timestamp.isoformat()}, CPU: {cpu_usage:.2f}%, GPU: {gpu_str}, GPU_Memory: {gpu_mem_str}\n"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        # Store in memory for average calculation
        self.timestamps.append(timestamp)
        self.cpu_readings.append(cpu_usage)
        if gpu_usage is not None:
            self.gpu_readings.append(gpu_usage)
    
    def run(self):
        """Main monitoring loop."""
        # Write header to log file
        with open(self.log_file, 'w') as f:
            f.write(f"# Compute Benchmark Log\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write(f"# Format: timestamp, CPU: X%, GPU: Y%, GPU_Memory: Z%\n")
            f.write(f"# Sampling interval: {self.interval}s\n\n")
        
        try:
            while self.running:
                timestamp = datetime.now()
                cpu_usage = self.get_cpu_usage()
                gpu_usage = self.get_gpu_usage()
                gpu_memory = self.get_gpu_memory_usage()
                
                # Log the reading
                self.log_reading(timestamp, cpu_usage, gpu_usage, gpu_memory)
                
                # Print to console
                gpu_str = f"{gpu_usage:.2f}%" if gpu_usage is not None else "N/A"
                gpu_mem_str = f"{gpu_memory:.2f}%" if gpu_memory is not None else "N/A"
                print(f"[{timestamp.strftime('%H:%M:%S')}] CPU: {cpu_usage:.2f}% | GPU: {gpu_str} | GPU Memory: {gpu_mem_str}")
                
                # Sleep until next reading
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            self.running = False
        
        finally:
            self.print_averages()
    
    def print_averages(self):
        """Calculate and print average usage statistics."""
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY")
        print("="*60)
        
        if len(self.cpu_readings) == 0:
            print("No readings collected.")
            return
        
        # Calculate CPU averages
        cpu_avg = sum(self.cpu_readings) / len(self.cpu_readings)
        cpu_min = min(self.cpu_readings)
        cpu_max = max(self.cpu_readings)
        
        print(f"\n🖥️  CPU Usage:")
        print(f"   Average: {cpu_avg:.2f}%")
        print(f"   Minimum: {cpu_min:.2f}%")
        print(f"   Maximum: {cpu_max:.2f}%")
        print(f"   Samples: {len(self.cpu_readings)}")
        
        # Calculate GPU averages if available
        if len(self.gpu_readings) > 0:
            gpu_avg = sum(self.gpu_readings) / len(self.gpu_readings)
            gpu_min = min(self.gpu_readings)
            gpu_max = max(self.gpu_readings)
            
            print(f"\n🎮 GPU Usage:")
            print(f"   Average: {gpu_avg:.2f}%")
            print(f"   Minimum: {gpu_min:.2f}%")
            print(f"   Maximum: {gpu_max:.2f}%")
            print(f"   Samples: {len(self.gpu_readings)}")
        else:
            print(f"\n🎮 GPU Usage: No GPU readings available")
        
        # Calculate duration
        if len(self.timestamps) > 1:
            duration = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
            print(f"\n⏱️  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        print(f"\n📝 Full log saved to: {self.log_file}")
        print("="*60)
        
        # Append summary to log file
        with open(self.log_file, 'a') as f:
            f.write(f"\n# Summary\n")
            f.write(f"# Ended: {datetime.now().isoformat()}\n")
            f.write(f"# CPU Average: {cpu_avg:.2f}% (Min: {cpu_min:.2f}%, Max: {cpu_max:.2f}%)\n")
            if len(self.gpu_readings) > 0:
                f.write(f"# GPU Average: {gpu_avg:.2f}% (Min: {gpu_min:.2f}%, Max: {gpu_max:.2f}%)\n")
            if len(self.timestamps) > 1:
                f.write(f"# Duration: {duration:.2f} seconds\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor and log GPU and CPU usage over time"
    )
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Path to log file (default: auto-generated with timestamp)'
    )
    parser.add_argument(
        '--interval', '-i',
        type=float,
        default=1.0,
        help='Sampling interval in seconds (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    benchmark = ComputeBenchmark(
        log_file=args.log_file,
        interval=args.interval
    )
    
    benchmark.run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

