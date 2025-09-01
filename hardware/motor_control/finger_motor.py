# Import the necessary libraries
from dynamixel_sdk import *  # Uses Dynamixel SDK library
import sys
import termios
import tty

# control mode
ADDR_OPERATING_MODE = 11

MY_DXL = 'X_SERIES'       # X330 (5.0 V recommended), X430, X540, 2X430
# Control table address
ADDR_MX_TORQUE_ENABLE = 64      # Address for torque enable
ADDR_MX_GOAL_POSITION = 116     # Address for goal position
ADDR_MX_GOAL_CURRENT = 102      # Address for goal current
ADDR_MX_PRESENT_POSITION = 132  # Address for present position
ADDR_MX_PRESENT_CURRENT = 126   # Address for present current
ADDR_MX_CURRENT_LIMIT = 38      # Address for current limit

# Protocol version
PROTOCOL_VERSION = 2.0  # Dynamixel XL330 uses protocol 2.0

# Default setting
DEVICENAME = '/dev/ttyUSB0'  # Check which port is being used on your setup
BAUDRATE = 57600            # Adjust to match your Dynamixel baud rate
TORQUE_ENABLE = 1           # Value for enabling torque
TORQUE_DISABLE = 0          # Value for disabling torque
DXL_MINIMUM_POSITION_VALUE = 0         # Refer to the Minimum Position Limit of product eManual
DXL_MAXIMUM_POSITION_VALUE = 4095      # Refer to the Maximum Position Limit of product eManual
DXL_MAXIMUM_CURRENT_VALUE = 1750      # Refer to the Maximum Current Limit of product eManual
DXL_MINIMUM_CURRENT_VALUE = 0      # Refer to the MINIMUM Current Limit of product eManual
DXL_MOVING_STATUS_THRESHOLD = 20       # Dynamixel moving status threshold
STEP_SIZE = 150                         # Step size for position adjustment

# Dynamixel IDs
DXL_IDS = [1, 2]  # List of IDs for chained Dynamixels

# Initialize PortHandler and PacketHandler
portHandler = PortHandler(DEVICENAME)
packetHandler = PacketHandler(PROTOCOL_VERSION)

# Open port
if not portHandler.openPort():
    print("Failed to open the port")
    quit()

if not portHandler.setBaudRate(BAUDRATE):
    print("Failed to set the baudrate")
    quit()

# Disable torque to change Operating Mode
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Torque Disable Error for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Torque Disable Error for ID {DXL_ID}: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Torque disabled for Dynamixel ID {DXL_ID}")
        
target_mode = 5
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE, target_mode)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Failed to set Operating Mode: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Error setting Operating Mode: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Operating Mode set to {target_mode}")
        
# Enable torque for each Dynamixel
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_ENABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Torque Enable Error for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Torque Enable Error for ID {DXL_ID}: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Torque enabled for Dynamixel ID {DXL_ID}")
        
# Read Operating Mode
for DXL_ID in DXL_IDS:
    dxl_operation_mode, dxl_comm_result, dxl_error = packetHandler.read1ByteTxRx(portHandler, DXL_ID, ADDR_OPERATING_MODE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Failed to read Operating Mode for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Error reading Operating Mode for ID {DXL_ID}: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Current Operating Mode for ID {DXL_ID}: {dxl_operation_mode}")

# Zeroing: Read initial positions and set them as zero
zero_positions = {}
dxl_zero_position = [2000, 3000] # u can adjust zero position after building the soft gripper

for DXL_ID in DXL_IDS:
    # set zero position
    dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, dxl_zero_position[DXL_ID - 1])
    if dxl_comm_result != COMM_SUCCESS:
                print(f"Goal Position Write Error for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Goal Position Write Error for ID {DXL_ID}: {packetHandler.getRxPacketError(dxl_error)}")
    time.sleep(1.0)
    # get present position
    dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_POSITION)
    # breakpoint()
    if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
        zero_positions[DXL_ID] = dxl_present_position
        print(f"ID {DXL_ID} Zero Position: {dxl_present_position}")
        continue
    else:
        print(f"Failed to read present position for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
        zero_positions[DXL_ID] = 0
        continue 

present_current = {}
    
# Current positions (relative to zero)
current_positions = zero_positions.copy()

# Function to read a single keypress
def get_key():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

# Control loop
print("Control motors with keys: 'g' for tightening tendon, 'h' for loosening tendon, Press 'e' to exit.")
try:
    while True:
        key = get_key()
        if key == 'e':
            break
        elif key == 'g':
            current_positions[1] = max(DXL_MINIMUM_POSITION_VALUE, current_positions[1] - 100)
            current_positions[2] = max(DXL_MINIMUM_POSITION_VALUE, current_positions[2] - 100)
        elif key == 'h':
            current_positions[1] = min(DXL_MAXIMUM_POSITION_VALUE, current_positions[1] + 100)
            current_positions[2] = min(DXL_MAXIMUM_POSITION_VALUE, current_positions[2] + 100)

        # Send updated positions
        for DXL_ID in DXL_IDS:
            dxl_goal_position = current_positions[DXL_ID]
            dxl_comm_result, dxl_error = packetHandler.write4ByteTxRx(portHandler, DXL_ID, ADDR_MX_GOAL_POSITION, dxl_goal_position)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Goal Position Write Error for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
            elif dxl_error != 0:
                print(f"Goal Position Write Error for ID {DXL_ID}: {packetHandler.getRxPacketError(dxl_error)}")

        # Print updated positions
        for DXL_ID in DXL_IDS:
            print(f"ID {DXL_ID} Position: {current_positions[DXL_ID]}")
            
            dxl_present_current, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_PRESENT_CURRENT)
            dxl_current_limit, dxl_comm_result, dxl_error = packetHandler.read2ByteTxRx(portHandler, DXL_ID, ADDR_MX_CURRENT_LIMIT)
            if dxl_comm_result != COMM_SUCCESS:
                print(f"Failed to read Present Current: {packetHandler.getTxRxResult(dxl_comm_result)}")
                continue
            elif dxl_error != 0:
                print(f"Error reading Present Current: {packetHandler.getRxPacketError(dxl_error)}")
                continue
            # Signed conversion
            if dxl_present_current > 32767:
                dxl_present_current -= 65536
            current_mA = dxl_present_current * 1.0  # Convert to milliamps
            # Display position and current
            print(f"ID {DXL_ID} Position: {current_positions[DXL_ID]}, Current: {current_mA:.2f} mA")

        # get present position
        dxl_present_position, dxl_comm_result, dxl_error = packetHandler.read4ByteTxRx(portHandler, 3, ADDR_MX_PRESENT_POSITION)
        # breakpoint()
        if dxl_comm_result == COMM_SUCCESS and dxl_error == 0:
            zero_positions[3] = dxl_present_position
            print(f"ID {3} Zero Position: {dxl_present_position}")
            continue
        else:
            print(f"Failed to read present position for ID {3}: {packetHandler.getTxRxResult(dxl_comm_result)}")
            zero_positions[3] = 0
            continue
            
except KeyboardInterrupt:
    pass

# Disable torque for each Dynamixel
for DXL_ID in DXL_IDS:
    dxl_comm_result, dxl_error = packetHandler.write1ByteTxRx(portHandler, DXL_ID, ADDR_MX_TORQUE_ENABLE, TORQUE_DISABLE)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Torque Disable Error for ID {DXL_ID}: {packetHandler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"Torque Disable Error for ID {DXL_ID}: {packetHandler.getRxPacketError(dxl_error)}")
    else:
        print(f"Torque disabled for Dynamixel ID {DXL_ID}")

# Close port
portHandler.closePort()