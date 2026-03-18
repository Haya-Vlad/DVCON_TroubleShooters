"""
FPGA Communication Bridge via USB/UART.
Sends quantized CNN feature data to Zynq-7000 FPGA and receives results.
Uses pyserial for USB-serial communication.
"""

import numpy as np
import struct
import time
from typing import Optional, Dict, List, Tuple


class FPGABridge:
    """
    USB/UART bridge for communicating with the CNN accelerator
    on the Zynq-7000 FPGA board.
    
    Protocol:
    - TX: Send quantized INT8 image data + weights
    - RX: Receive accelerated CNN features
    - Control: Start/stop via register writes
    """

    # Command bytes
    CMD_WRITE_REG    = 0x01
    CMD_READ_REG     = 0x02
    CMD_WRITE_DATA   = 0x03
    CMD_READ_DATA    = 0x04
    CMD_WRITE_WEIGHT = 0x05
    CMD_START        = 0x06
    CMD_STATUS       = 0x07
    CMD_ACK          = 0xAA
    CMD_NACK         = 0x55

    def __init__(self, config=None):
        from taskgraph_edge.config import FPGAConfig
        self.config = config or FPGAConfig()
        self.serial = None
        self.connected = False
        self._connect()

    def _connect(self):
        """Connect to FPGA via USB serial port."""
        try:
            import serial
            self.serial = serial.Serial(
                port=self.config.port,
                baudrate=self.config.baud_rate,
                timeout=self.config.timeout,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            self.connected = True
            print(f"[FPGABridge] Connected to {self.config.port} "
                  f"@ {self.config.baud_rate} baud")
            
            # Read version to verify connection
            version = self.read_register(0x1C)
            if version is not None:
                print(f"[FPGABridge] FPGA version: 0x{version:08X}")
            
        except ImportError:
            print("[FPGABridge] pyserial not installed. Install: pip install pyserial")
            self.connected = False
        except Exception as e:
            print(f"[FPGABridge] Connection failed on {self.config.port}: {e}")
            self.connected = False

    def is_connected(self) -> bool:
        """Check if FPGA is connected."""
        return self.connected and self.serial is not None and self.serial.is_open

    def quantize_data(self, data: np.ndarray) -> np.ndarray:
        """Quantize float32 data to INT8 for FPGA."""
        if not self.config.quantization.enabled:
            return data.astype(np.int8)
        
        scale = self.config.quantization.scale_factor
        quantized = np.clip(
            np.round(data * scale),
            -128, 127
        ).astype(np.int8)
        return quantized

    def dequantize_data(self, data: np.ndarray) -> np.ndarray:
        """Dequantize INT8 data back to float32."""
        scale = self.config.quantization.scale_factor
        return data.astype(np.float32) / scale

    def write_register(self, addr: int, value: int) -> bool:
        """Write a 32-bit value to an FPGA register."""
        if not self.is_connected():
            return False
        
        try:
            # Protocol: CMD(1) + ADDR(4) + DATA(4)
            packet = struct.pack('<BII', self.CMD_WRITE_REG, addr, value)
            self.serial.write(packet)
            
            # Wait for ACK
            response = self.serial.read(1)
            if response and response[0] == self.CMD_ACK:
                return True
            return False
        except Exception as e:
            print(f"[FPGABridge] Write register error: {e}")
            return False

    def read_register(self, addr: int) -> Optional[int]:
        """Read a 32-bit value from an FPGA register."""
        if not self.is_connected():
            return None
        
        try:
            # Protocol: CMD(1) + ADDR(4)
            packet = struct.pack('<BI', self.CMD_READ_REG, addr)
            self.serial.write(packet)
            
            # Read response: ACK(1) + DATA(4)
            response = self.serial.read(5)
            if response and len(response) == 5 and response[0] == self.CMD_ACK:
                value = struct.unpack('<I', response[1:])[0]
                return value
            return None
        except Exception as e:
            print(f"[FPGABridge] Read register error: {e}")
            return None

    def send_weights(self, weights: np.ndarray) -> bool:
        """Send CNN weights to FPGA weight memory."""
        if not self.is_connected():
            return False
        
        # Quantize weights
        q_weights = self.quantize_data(weights)
        weight_bytes = q_weights.tobytes()
        
        try:
            # Send weight data in chunks
            chunk_size = 256
            for i in range(0, len(weight_bytes), chunk_size):
                chunk = weight_bytes[i:i+chunk_size]
                header = struct.pack('<BH', self.CMD_WRITE_WEIGHT, len(chunk))
                self.serial.write(header + chunk)
                
                # Wait for ACK
                response = self.serial.read(1)
                if not response or response[0] != self.CMD_ACK:
                    print(f"[FPGABridge] Weight upload failed at byte {i}")
                    return False
            
            return True
        except Exception as e:
            print(f"[FPGABridge] Send weights error: {e}")
            return False

    def send_image_data(self, image_data: np.ndarray) -> bool:
        """Send quantized image data to FPGA for processing."""
        if not self.is_connected():
            return False
        
        # Quantize
        q_data = self.quantize_data(image_data.flatten())
        data_bytes = q_data.tobytes()
        
        try:
            # Send in chunks
            chunk_size = 512
            total_chunks = (len(data_bytes) + chunk_size - 1) // chunk_size
            
            for i in range(0, len(data_bytes), chunk_size):
                chunk = data_bytes[i:i+chunk_size]
                header = struct.pack('<BHI', self.CMD_WRITE_DATA, len(chunk), i)
                self.serial.write(header + chunk)
                
                # ACK
                response = self.serial.read(1)
                if not response or response[0] != self.CMD_ACK:
                    return False
            
            return True
        except Exception as e:
            print(f"[FPGABridge] Send data error: {e}")
            return False

    def start_compute(self, mode: int = 2) -> bool:
        """
        Start FPGA computation.
        Mode: 0=Conv, 1=Conv+ReLU, 2=Conv+ReLU+MaxPool (full pipeline)
        """
        if not self.is_connected():
            return False
        
        # Write mode to control register
        ctrl_value = (mode << 8) | 0x01  # Start bit
        return self.write_register(0x00, ctrl_value)

    def wait_done(self, timeout: float = 5.0) -> bool:
        """Wait for FPGA computation to complete."""
        if not self.is_connected():
            return False
        
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            status = self.read_register(0x04)
            if status is not None and (status & 0x02):  # Done bit
                return True
            time.sleep(0.01)  # 10ms poll interval
        
        print("[FPGABridge] Timeout waiting for FPGA")
        return False

    def receive_features(self, num_elements: int) -> Optional[np.ndarray]:
        """Receive computed features from FPGA."""
        if not self.is_connected():
            return None
        
        try:
            # Request data
            packet = struct.pack('<BI', self.CMD_READ_DATA, num_elements)
            self.serial.write(packet)
            
            # Read response
            response = self.serial.read(1 + num_elements)
            if response and len(response) == 1 + num_elements:
                if response[0] == self.CMD_ACK:
                    data = np.frombuffer(response[1:], dtype=np.int8)
                    return self.dequantize_data(data)
            
            return None
        except Exception as e:
            print(f"[FPGABridge] Receive error: {e}")
            return None

    def accelerate_cnn(
        self,
        image_patch: np.ndarray,
        weights: np.ndarray,
        mode: int = 2,
    ) -> Optional[np.ndarray]:
        """
        Full CNN acceleration flow:
        1. Send weights
        2. Send image data
        3. Start computation
        4. Wait for completion
        5. Receive results
        """
        if not self.is_connected():
            print("[FPGABridge] Not connected, falling back to CPU")
            return None

        # Step 1: Send weights
        if not self.send_weights(weights):
            return None

        # Step 2: Send image data
        if not self.send_image_data(image_patch):
            return None

        # Step 3: Start
        if not self.start_compute(mode):
            return None

        # Step 4: Wait
        if not self.wait_done():
            return None

        # Step 5: Get results
        # Output size depends on mode
        output_size = image_patch.size // (4 if mode == 2 else 1)
        features = self.receive_features(output_size)

        return features

    def get_performance_stats(self) -> Dict[str, int]:
        """Read performance counters from FPGA."""
        stats = {}
        
        cycle_count = self.read_register(0x10)
        compute_cycles = self.read_register(0x14)
        data_count = self.read_register(0x18)
        
        if cycle_count is not None:
            stats["total_cycles"] = cycle_count
        if compute_cycles is not None:
            stats["compute_cycles"] = compute_cycles
            stats["utilization"] = (compute_cycles / max(1, cycle_count)) * 100
        if data_count is not None:
            stats["data_processed"] = data_count
        
        return stats

    def close(self):
        """Close the serial connection."""
        if self.serial and self.serial.is_open:
            self.serial.close()
            self.connected = False
            print("[FPGABridge] Disconnected")

    def __del__(self):
        self.close()
