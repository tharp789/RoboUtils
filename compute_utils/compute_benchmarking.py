#!/usr/bin/env python3
"""
Script to monitor and log GPU and CPU usage over time.
Logs usage to a file and calculates average usage when the script is terminated.
Supports both standard NVIDIA GPUs and NVIDIA Jetson devices.
"""

import time
import signal
import sys
from datetime import datetime
from pathlib import Path
import psutil
import platform

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

# Try to import Jetson monitoring library
try:
    from jtop import jtop
    JETSON_STATS_AVAILABLE = True
except ImportError:
    JETSON_STATS_AVAILABLE = False
    print("Info: jetson-stats not available. Jetson-specific monitoring will be disabled.")
    print("Install with: pip install jetson-stats")


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
        self.gpu_memory_readings = []
        self.timestamps = []
        
        # Jetson-specific metrics
        self.is_jetson = False
        self.jetson_stats = None
        self.power_readings = []
        self.temperature_readings = []
        self.cpu_per_core_readings = []
        
        # Setup log file
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f"compute_benchmark_{timestamp}.log"
        
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Detect Jetson device
        self.is_jetson = self._detect_jetson()
        
        # Initialize monitoring based on device type
        self.gpu_available = False
        if self.is_jetson and JETSON_STATS_AVAILABLE:
            try:
                self.jetson_stats = jtop()
                self.jetson_stats.start()
                self.gpu_available = True
                jetson_model = self.jetson_stats.board['model'] if hasattr(self.jetson_stats, 'board') else "Jetson"
                print(f"✅ Jetson monitoring enabled ({jetson_model} detected)")
            except Exception as e:
                print(f"⚠️  Jetson monitoring unavailable: {e}")
                self.jetson_stats = None
                # Fall back to standard GPU monitoring
                self._init_standard_gpu()
        else:
            # Use standard GPU monitoring
            self._init_standard_gpu()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        device_type = "Jetson" if self.is_jetson else "Standard"
        print(f"📊 Starting compute benchmark monitoring ({device_type} device)...")
        print(f"📝 Logging to: {self.log_file}")
        print(f"⏱️  Sampling interval: {self.interval}s")
        print(f"🛑 Press Ctrl+C to stop and view averages\n")
    
    def _detect_jetson(self):
        """Detect if running on a Jetson device."""
        # Check for Jetson-specific files/systems
        jetson_indicators = [
            '/proc/device-tree/model',  # Device tree model
            '/sys/firmware/devicetree/base/model',  # Alternative device tree path
        ]
        
        for indicator in jetson_indicators:
            try:
                with open(indicator, 'r') as f:
                    model = f.read().strip()
                    if 'jetson' in model.lower() or 'tegra' in model.lower():
                        return True
            except (FileNotFoundError, PermissionError):
                continue
        
        # Check platform
        if platform.machine().startswith('aarch64'):
            # Could be Jetson, but not definitive
            # Try to check for NVIDIA-specific files
            try:
                import os
                if os.path.exists('/usr/bin/tegrastats'):
                    return True
            except:
                pass
        
        return False
    
    def _init_standard_gpu(self):
        """Initialize standard GPU monitoring (non-Jetson)."""
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
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.running = False
        if self.jetson_stats is not None:
            try:
                self.jetson_stats.close()
            except:
                pass
    
    def get_cpu_usage(self):
        """Get current CPU usage percentage."""
        if self.is_jetson and self.jetson_stats is not None:
            try:
                stats = self.jetson_stats.stats
                if stats:
                    # Get average CPU usage across all cores
                    cpu_total = stats.get('CPU', {})
                    if isinstance(cpu_total, dict):
                        cpu_values = [v for k, v in cpu_total.items() if 'cpu' in k.lower() and isinstance(v, (int, float))]
                        if cpu_values:
                            return sum(cpu_values) / len(cpu_values)
            except Exception as e:
                pass
        
        return psutil.cpu_percent(interval=None)
    
    def get_cpu_per_core(self):
        """Get CPU usage per core (Jetson only)."""
        if self.is_jetson and self.jetson_stats is not None:
            try:
                stats = self.jetson_stats.stats
                if stats:
                    cpu_total = stats.get('CPU', {})
                    if isinstance(cpu_total, dict):
                        cpu_cores = {}
                        for k, v in cpu_total.items():
                            if 'cpu' in k.lower() and isinstance(v, (int, float)):
                                cpu_cores[k] = v
                        return cpu_cores
            except Exception as e:
                pass
        return None
    
    def get_gpu_usage(self):
        """Get current GPU usage percentage."""
        if not self.gpu_available:
            return None
        
        try:
            if self.is_jetson and self.jetson_stats is not None:
                # Use jetson-stats
                stats = self.jetson_stats.stats
                if stats:
                    gpu_info = stats.get('GPU', {})
                    if isinstance(gpu_info, dict):
                        # Try different possible keys for GPU usage
                        for key in ['val', 'usage', 'load', 'utilization']:
                            if key in gpu_info:
                                val = gpu_info[key]
                                if isinstance(val, (int, float)):
                                    return float(val)
                        # If no direct usage, try to calculate from freq
                        if 'frq' in gpu_info or 'freq' in gpu_info:
                            # On Jetson, GPU usage might be inferred from frequency
                            # For now, return None if we can't get direct usage
                            pass
            elif GPU_AVAILABLE:
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
            if self.is_jetson and self.jetson_stats is not None:
                # Use jetson-stats
                stats = self.jetson_stats.stats
                if stats:
                    ram_info = stats.get('RAM', {})
                    if isinstance(ram_info, dict):
                        # Jetson uses unified memory, so GPU memory is part of RAM
                        # Try to get GPU-specific memory if available
                        gpu_ram = ram_info.get('gpu', {})
                        if isinstance(gpu_ram, dict):
                            used = gpu_ram.get('used', 0)
                            total = gpu_ram.get('tot', gpu_ram.get('total', 1))
                            if total > 0:
                                return (used / total) * 100
            elif GPU_AVAILABLE:
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
    
    def get_power_usage(self):
        """Get current power consumption in watts (Jetson only)."""
        if self.is_jetson and self.jetson_stats is not None:
            try:
                stats = self.jetson_stats.stats
                if stats:
                    power_info = stats.get('POM', {})
                    if isinstance(power_info, dict):
                        # Try different possible keys
                        for key in ['5V_IN', '5V', 'power', 'watts']:
                            if key in power_info:
                                val = power_info[key]
                                if isinstance(val, (int, float)):
                                    return float(val)
            except Exception as e:
                pass
        return None
    
    def get_temperature(self):
        """Get current temperature in Celsius (Jetson only)."""
        if self.is_jetson and self.jetson_stats is not None:
            try:
                stats = self.jetson_stats.stats
                if stats:
                    temp_info = stats.get('TEMP', {})
                    if isinstance(temp_info, dict):
                        # Get average temperature or specific sensor
                        temp_values = [v for v in temp_info.values() if isinstance(v, (int, float))]
                        if temp_values:
                            return sum(temp_values) / len(temp_values)
                        # Try common keys
                        for key in ['CPU', 'GPU', 'thermal', 'temp']:
                            if key in temp_info:
                                val = temp_info[key]
                                if isinstance(val, (int, float)):
                                    return float(val)
            except Exception as e:
                pass
        return None
    
    def log_reading(self, timestamp, cpu_usage, gpu_usage, gpu_memory, power=None, temperature=None, cpu_per_core=None):
        """Log a single reading to file and store in memory."""
        # Format log line
        gpu_str = f"{gpu_usage:.2f}%" if gpu_usage is not None else "N/A"
        gpu_mem_str = f"{gpu_memory:.2f}%" if gpu_memory is not None else "N/A"
        
        log_parts = [f"{timestamp.isoformat()}, CPU: {cpu_usage:.2f}%, GPU: {gpu_str}, GPU_Memory: {gpu_mem_str}"]
        
        # Add Jetson-specific metrics
        if power is not None:
            log_parts.append(f"Power: {power:.2f}W")
        if temperature is not None:
            log_parts.append(f"Temp: {temperature:.1f}°C")
        if cpu_per_core is not None and isinstance(cpu_per_core, dict):
            core_str = ", ".join([f"{k}: {v:.1f}%" for k, v in cpu_per_core.items()])
            log_parts.append(f"CPU_Cores: [{core_str}]")
        
        log_line = ", ".join(log_parts) + "\n"
        
        # Write to file
        with open(self.log_file, 'a') as f:
            f.write(log_line)
        
        # Store in memory for average calculation
        self.timestamps.append(timestamp)
        self.cpu_readings.append(cpu_usage)
        if gpu_usage is not None:
            self.gpu_readings.append(gpu_usage)
        if gpu_memory is not None:
            self.gpu_memory_readings.append(gpu_memory)
        if power is not None:
            self.power_readings.append(power)
        if temperature is not None:
            self.temperature_readings.append(temperature)
        if cpu_per_core is not None:
            self.cpu_per_core_readings.append(cpu_per_core)
    
    def run(self):
        """Main monitoring loop."""
        # Write header to log file
        device_type = "Jetson" if self.is_jetson else "Standard"
        with open(self.log_file, 'w') as f:
            f.write(f"# Compute Benchmark Log ({device_type} Device)\n")
            f.write(f"# Started: {datetime.now().isoformat()}\n")
            if self.is_jetson:
                f.write(f"# Format: timestamp, CPU: X%, GPU: Y%, GPU_Memory: Z%, Power: W, Temp: °C, CPU_Cores: [...]\n")
            else:
                f.write(f"# Format: timestamp, CPU: X%, GPU: Y%, GPU_Memory: Z%\n")
            f.write(f"# Sampling interval: {self.interval}s\n\n")
        
        try:
            while self.running:
                timestamp = datetime.now()
                cpu_usage = self.get_cpu_usage()
                gpu_usage = self.get_gpu_usage()
                gpu_memory = self.get_gpu_memory_usage()
                
                # Get Jetson-specific metrics
                power = None
                temperature = None
                cpu_per_core = None
                if self.is_jetson:
                    power = self.get_power_usage()
                    temperature = self.get_temperature()
                    cpu_per_core = self.get_cpu_per_core()
                
                # Log the reading
                self.log_reading(timestamp, cpu_usage, gpu_usage, gpu_memory, power, temperature, cpu_per_core)
                
                # Print to console
                gpu_str = f"{gpu_usage:.2f}%" if gpu_usage is not None else "N/A"
                gpu_mem_str = f"{gpu_memory:.2f}%" if gpu_memory is not None else "N/A"
                console_parts = [f"[{timestamp.strftime('%H:%M:%S')}] CPU: {cpu_usage:.2f}% | GPU: {gpu_str} | GPU Memory: {gpu_mem_str}"]
                
                if power is not None:
                    console_parts.append(f"Power: {power:.2f}W")
                if temperature is not None:
                    console_parts.append(f"Temp: {temperature:.1f}°C")
                
                print(" | ".join(console_parts))
                
                # Sleep until next reading
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            self.running = False
        
        finally:
            if self.jetson_stats is not None:
                try:
                    self.jetson_stats.close()
                except:
                    pass
            self.print_averages()
    
    def print_averages(self):
        """Calculate and print average usage statistics."""
        print("\n" + "="*60)
        device_type = "JETSON" if self.is_jetson else "STANDARD"
        print(f"📊 BENCHMARK SUMMARY ({device_type} DEVICE)")
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
        
        # GPU Memory averages
        if len(self.gpu_memory_readings) > 0:
            gpu_mem_avg = sum(self.gpu_memory_readings) / len(self.gpu_memory_readings)
            gpu_mem_min = min(self.gpu_memory_readings)
            gpu_mem_max = max(self.gpu_memory_readings)
            
            print(f"\n💾 GPU Memory Usage:")
            print(f"   Average: {gpu_mem_avg:.2f}%")
            print(f"   Minimum: {gpu_mem_min:.2f}%")
            print(f"   Maximum: {gpu_mem_max:.2f}%")
            print(f"   Samples: {len(self.gpu_memory_readings)}")
        
        # Jetson-specific metrics
        if self.is_jetson:
            if len(self.power_readings) > 0:
                power_avg = sum(self.power_readings) / len(self.power_readings)
                power_min = min(self.power_readings)
                power_max = max(self.power_readings)
                
                print(f"\n⚡ Power Consumption:")
                print(f"   Average: {power_avg:.2f}W")
                print(f"   Minimum: {power_min:.2f}W")
                print(f"   Maximum: {power_max:.2f}W")
                print(f"   Samples: {len(self.power_readings)}")
            
            if len(self.temperature_readings) > 0:
                temp_avg = sum(self.temperature_readings) / len(self.temperature_readings)
                temp_min = min(self.temperature_readings)
                temp_max = max(self.temperature_readings)
                
                print(f"\n🌡️  Temperature:")
                print(f"   Average: {temp_avg:.1f}°C")
                print(f"   Minimum: {temp_min:.1f}°C")
                print(f"   Maximum: {temp_max:.1f}°C")
                print(f"   Samples: {len(self.temperature_readings)}")
        
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
                gpu_avg = sum(self.gpu_readings) / len(self.gpu_readings)
                gpu_min = min(self.gpu_readings)
                gpu_max = max(self.gpu_readings)
                f.write(f"# GPU Average: {gpu_avg:.2f}% (Min: {gpu_min:.2f}%, Max: {gpu_max:.2f}%)\n")
            if len(self.gpu_memory_readings) > 0:
                gpu_mem_avg = sum(self.gpu_memory_readings) / len(self.gpu_memory_readings)
                gpu_mem_min = min(self.gpu_memory_readings)
                gpu_mem_max = max(self.gpu_memory_readings)
                f.write(f"# GPU Memory Average: {gpu_mem_avg:.2f}% (Min: {gpu_mem_min:.2f}%, Max: {gpu_mem_max:.2f}%)\n")
            if len(self.power_readings) > 0:
                power_avg = sum(self.power_readings) / len(self.power_readings)
                power_min = min(self.power_readings)
                power_max = max(self.power_readings)
                f.write(f"# Power Average: {power_avg:.2f}W (Min: {power_min:.2f}W, Max: {power_max:.2f}W)\n")
            if len(self.temperature_readings) > 0:
                temp_avg = sum(self.temperature_readings) / len(self.temperature_readings)
                temp_min = min(self.temperature_readings)
                temp_max = max(self.temperature_readings)
                f.write(f"# Temperature Average: {temp_avg:.1f}°C (Min: {temp_min:.1f}°C, Max: {temp_max:.1f}°C)\n")
            if len(self.timestamps) > 1:
                duration = (self.timestamps[-1] - self.timestamps[0]).total_seconds()
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

