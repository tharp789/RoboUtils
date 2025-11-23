#!/usr/bin/env python3
"""
Script to monitor and log GPU and CPU usage over time on NVIDIA Jetson devices.
Uses tegrastats to collect system statistics.
Logs usage to a file and calculates average usage when the script is terminated.
"""

import time
import signal
import sys
import subprocess
import re
from datetime import datetime
from pathlib import Path


class ComputeBenchmark:
    def __init__(self, log_file=None, interval=1.0):
        """
        Initialize the compute benchmark monitor for Jetson devices.
        
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
            log_file = f"compute_benchmark_jetson_{timestamp}.log"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize tegrastats subprocess
        self.tegrastats_process = None
        self.tegrastats_available = False
        
        # Check if tegrastats is available
        try:
            result = subprocess.run(['which', 'tegrastats'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=2)
            if result.returncode == 0:
                self.tegrastats_available = True
                print(f"✅ tegrastats found, starting monitoring...")
            else:
                print(f"⚠️  tegrastats not found. Please ensure it's installed on your Jetson device.")
        except Exception as e:
            print(f"⚠️  Error checking for tegrastats: {e}")
        
        # Start tegrastats if available
        if self.tegrastats_available:
            try:
                # Start tegrastats with interval matching our sampling rate
                interval_ms = int(self.interval * 1000)
                # Use unbuffered output for real-time reading
                self.tegrastats_process = subprocess.Popen(
                    ['sudo', 'tegrastats', '--interval', str(interval_ms)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=0  # Unbuffered
                )
                # Give tegrastats a moment to start
                time.sleep(0.5)
                print(f"✅ tegrastats started successfully")
            except Exception as e:
                print(f"⚠️  Failed to start tegrastats: {e}")
                print(f"   Note: tegrastats requires sudo privileges")
                self.tegrastats_available = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        print(f"📊 Starting compute benchmark monitoring (Jetson)...")
        print(f"📝 Logging to: {self.log_file}")
        print(f"⏱️  Sampling interval: {self.interval}s")
        print(f"🛑 Press Ctrl+C to stop and view averages\n")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.running = False
    
    def _parse_tegrastats_line(self, line):
        """
        Parse a line from tegrastats output to extract CPU and GPU usage.
        
        Example line format:
        RAM 1234/12345MB (lfb 1234x4MB) CPU [0%,1%,2%,3%] EMC_FREQ 0% GR3D_FREQ 0% APE 25 MTS fg 0% bg 0% AO@0C GPU@0C PMIC@100C
        
        Returns:
            tuple: (cpu_usage_percent, gpu_usage_percent) or (None, None) if parsing fails
        """
        cpu_usage = None
        gpu_usage = None
        
        try:
            # Extract CPU usage - format: CPU [0%,1%,2%,3%]
            cpu_match = re.search(r'CPU\s+\[([\d%@,\s]+)\]', line)
            if cpu_match:
                cpu_values_str = cpu_match.group(1)
                # Extract individual CPU percentages
                cpu_percentages = re.findall(r'(\d+)%', cpu_values_str)
                if cpu_percentages:
                    # Calculate average CPU usage across all cores
                    cpu_values = [int(p) for p in cpu_percentages]
                    cpu_usage = sum(cpu_values) / len(cpu_values)
            
            # Extract GPU usage - format: GR3D_FREQ X%
            gpu_match = re.search(r'GR3D_FREQ\s+(\d+)%', line)
            if gpu_match:
                gpu_usage = int(gpu_match.group(1))
        
        except Exception as e:
            # Silently handle parsing errors
            pass
        
        return cpu_usage, gpu_usage
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage from tegrastats."""
        if not self.tegrastats_available or self.tegrastats_process is None:
            return None
        
        # Read a line from tegrastats output
        try:
            line = self.tegrastats_process.stdout.readline()
            if line:
                cpu_usage, _ = self._parse_tegrastats_line(line)
                return cpu_usage
        except Exception as e:
            pass
        
        return None
    
    def get_gpu_usage(self):
        """Get current GPU usage percentage from tegrastats."""
        if not self.tegrastats_available or self.tegrastats_process is None:
            return None
        
        # Read a line from tegrastats output
        try:
            line = self.tegrastats_process.stdout.readline()
            if line:
                _, gpu_usage = self._parse_tegrastats_line(line)
                return gpu_usage
        except Exception as e:
            pass
        
        return None
    
    def get_usage(self):
        """
        Get both CPU and GPU usage from a single tegrastats line.
        This is more efficient than reading twice.
        
        Returns:
            tuple: (cpu_usage, gpu_usage)
        """
        if not self.tegrastats_available or self.tegrastats_process is None:
            return None, None
        
        try:
            # Read a line from tegrastats output
            # Since tegrastats outputs at regular intervals, this should not block for long
            line = self.tegrastats_process.stdout.readline()
            if line and line.strip():
                return self._parse_tegrastats_line(line)
        except Exception as e:
            # Handle any read errors
            pass
        
        return None, None
    
    def log_reading(self, timestamp, cpu_usage, gpu_usage):
        """Log a single reading to file and store in memory."""
        # Format log line
        cpu_str = f"{cpu_usage:.2f}%" if cpu_usage is not None else "N/A"
        gpu_str = f"{gpu_usage:.2f}%" if gpu_usage is not None else "N/A"
        log_line = f"{timestamp.isoformat()}, CPU: {cpu_str}, GPU: {gpu_str}\n"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        # Store in memory for average calculation
        self.timestamps.append(timestamp)
        if cpu_usage is not None:
            self.cpu_readings.append(cpu_usage)
        if gpu_usage is not None:
            self.gpu_readings.append(gpu_usage)
    
    def run(self):
        """Main monitoring loop."""
        if not self.tegrastats_available:
            print("❌ tegrastats is not available. Cannot proceed with monitoring.")
            return
        
        # Write header to log file
        with open(self.log_file, 'w') as f:
            f.write(f"# Compute Benchmark Log (Jetson - tegrastats)\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            f.write(f"# Format: timestamp, CPU: X%, GPU: Y%\n")
            f.write(f"# Sampling interval: {self.interval}s\n\n")
        
        try:
            while self.running:
                timestamp = datetime.now()
                
                # Get both CPU and GPU usage from a single tegrastats line
                cpu_usage, gpu_usage = self.get_usage()
                
                # Log the reading
                self.log_reading(timestamp, cpu_usage, gpu_usage)
                
                # Print to console
                cpu_str = f"{cpu_usage:.2f}%" if cpu_usage is not None else "N/A"
                gpu_str = f"{gpu_usage:.2f}%" if gpu_usage is not None else "N/A"
                print(f"[{timestamp.strftime('%H:%M:%S')}] CPU: {cpu_str} | GPU: {gpu_str}")
                
                # Sleep until next reading
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            self.running = False
        
        finally:
            # Stop tegrastats
            if self.tegrastats_process is not None:
                try:
                    # Try to stop tegrastats gracefully
                    subprocess.run(['sudo', 'tegrastats', '--stop'], 
                                 timeout=2, 
                                 capture_output=True)
                except:
                    pass
                # Terminate the process if still running
                try:
                    self.tegrastats_process.terminate()
                    self.tegrastats_process.wait(timeout=2)
                except:
                    try:
                        self.tegrastats_process.kill()
                    except:
                        pass
            
            self.print_averages()
    
    def print_averages(self):
        """Calculate and print average usage statistics."""
        print("\n" + "="*60)
        print("📊 BENCHMARK SUMMARY (JETSON)")
        print("="*60)
        
        if len(self.cpu_readings) == 0 and len(self.gpu_readings) == 0:
            print("No readings collected.")
            return
        
        # Calculate CPU averages
        if len(self.cpu_readings) > 0:
            cpu_avg = sum(self.cpu_readings) / len(self.cpu_readings)
            cpu_min = min(self.cpu_readings)
            cpu_max = max(self.cpu_readings)
            
            print(f"\n🖥️  CPU Usage:")
            print(f"   Average: {cpu_avg:.2f}%")
            print(f"   Minimum: {cpu_min:.2f}%")
            print(f"   Maximum: {cpu_max:.2f}%")
            print(f"   Samples: {len(self.cpu_readings)}")
        else:
            print(f"\n🖥️  CPU Usage: No CPU readings available")
        
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
            if len(self.cpu_readings) > 0:
                cpu_avg = sum(self.cpu_readings) / len(self.cpu_readings)
                cpu_min = min(self.cpu_readings)
                cpu_max = max(self.cpu_readings)
                f.write(f"# CPU Average: {cpu_avg:.2f}% (Min: {cpu_min:.2f}%, Max: {cpu_max:.2f}%)\n")
            if len(self.gpu_readings) > 0:
                gpu_avg = sum(self.gpu_readings) / len(self.gpu_readings)
                gpu_min = min(self.gpu_readings)
                gpu_max = max(self.gpu_readings)
                f.write(f"# GPU Average: {gpu_avg:.2f}% (Min: {gpu_min:.2f}%, Max: {gpu_max:.2f}%)\n")
            if len(self.timestamps) > 1:
                duration = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
                f.write(f"# Duration: {duration:.2f} seconds\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor and log GPU and CPU usage over time on Jetson devices using tegrastats"
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

