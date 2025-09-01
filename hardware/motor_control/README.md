By xueqian:
Motor control script for the Dynamixel actuators used to drive tendon motion.
You can try on both Linux and MacOS.
### Requirements:
- Python 3.10
- Robotis DynamixelSDK (Python) installed and USB permissions configured
### Installation
```
git clone https://github.com/ROBOTIS-GIT/DynamixelSDK.git
cd DynamixelSDK/python
pip install -e .
```
If you are on Linux, open the port by running open_port.sh (not needed on macOS). Make sure the port matches the one you are using (default: '/dev/ttyUSB0').
### Start
Now it's time to give it a try!
```
python finger_motor.py
```



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

