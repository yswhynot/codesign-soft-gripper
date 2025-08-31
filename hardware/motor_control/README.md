Motor control scripts for the Dynamixel actuators used to drive tendon motion.

Requirements:
- Python 3.10
- Robotis DynamixelSDK (Python) installed and USB permissions configured

Typical tasks:
- Set IDs / baudrate
- Read/Write goal position / torque enable
- Synchronize multiple motors

Usage example (adjust port and IDs):
```bash
python read_write_multi.py --port /dev/ttyUSB0 --ids 1 2 --baud 1000000
```

Troubleshooting:
- Check `dmesg` for the USB serial device path.
- Ensure the user has permission to access the serial device (e.g., add to `dialout` group on Linux).
