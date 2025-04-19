import socket
import json
import time
import threading
import hid
import struct
import math

class SpaceMouseFrankaController:
    def __init__(self, server_host='localhost', server_port=12345):
        # Server connection settings
        self.server_host = server_host
        self.server_port = server_port
        self.socket = None
        self.connected = False
        
        # Controller state
        self.running = False
        self.gripper_state = 0.0  # 0.0 is closed, 255.0 is open
        
        # Movement settings
        self.translation_scale = 0.0001  # Scale factor for movement (smaller = more precise)
        self.rotation_scale = 0.0001     # Scale factor for rotation
        
        # SpaceMouse device
        self.device = None
        
    def find_spacemouse(self):
        """Find and open the SpaceMouse device"""
        spacemouse = None
        for device in hid.enumerate():
            if device['vendor_id'] == 0x46d:  # 3Dconnexion
                print(f"Found: {device['product_string']}")
                spacemouse = device
                break
                
        if not spacemouse:
            print("SpaceMouse not found.")
            return False
            
        try:
            self.device = hid.device()
            self.device.open_path(spacemouse['path'])
            self.device.set_nonblocking(True)
            print("SpaceMouse connected successfully")
            return True
        except Exception as e:
            print(f"Failed to open SpaceMouse: {e}")
            return False
    
    def connect_to_server(self):
        """Attempt to connect to the robot simulation server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_host, self.server_port))
            self.connected = True
            print(f"Connected to server at {self.server_host}:{self.server_port}")
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """Disconnect from the server"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        self.connected = False
        print("Disconnected from server")
        
        if self.device:
            try:
                self.device.close()
            except:
                pass
            print("SpaceMouse disconnected")
    
    def send_control_data(self, delta_pos, rot, grip):
        """Send controller data to the server"""
        if not self.connected:
            return False
        
        try:
            # Create data packet
            data = {
                'delta_pos': delta_pos,
                'rot': rot,
                'grip': grip
            }
            
            # Convert to JSON and send
            json_data = json.dumps(data)
            self.socket.sendall(json_data.encode('utf-8'))
            return True
        except Exception as e:
            print(f"Failed to send data: {e}")
            self.connected = False
            return False
    
    def process_spacemouse_input(self):
        """Process SpaceMouse input and return movement deltas and rotation"""
        # Default values (no movement)
        delta_pos = [0.0, 0.0, 0.0]
        rot = [0.0, 0.0, 0.0]
        left_pressed = False
        right_pressed = False
        
        # Read data from SpaceMouse
        data = self.device.read(13)
        if not data:
            return delta_pos, rot, left_pressed, right_pressed
        
        report_id = data[0]
        
        if report_id == 1:  # Translation
            tx = struct.unpack('<h', bytes(data[1:3]))[0]
            ty = struct.unpack('<h', bytes(data[3:5]))[0]
            tz = struct.unpack('<h', bytes(data[5:7]))[0]
            
            # Apply deadzone and scale
            deadzone = 50
            if abs(tx) < deadzone: tx = 0
            if abs(ty) < deadzone: ty = 0
            if abs(tz) < deadzone: tz = 0
            
            # Convert to delta position (with inverted axes as needed for intuitive control)
            delta_pos = [
                tx * self.translation_scale,   # X axis
                -ty * self.translation_scale,  # Y axis (inverted)
                -tz * self.translation_scale   # Z axis (inverted)
            ]
            
        elif report_id == 2:  # Rotation
            rx = struct.unpack('<h', bytes(data[1:3]))[0]
            ry = struct.unpack('<h', bytes(data[3:5]))[0]
            rz = struct.unpack('<h', bytes(data[5:7]))[0]
            
            # Apply deadzone and scale
            deadzone = 50
            if abs(rx) < deadzone: rx = 0
            if abs(ry) < deadzone: ry = 0
            if abs(rz) < deadzone: rz = 0
            
            # Scale rotation values
            rot = [
                rx * self.rotation_scale,
                ry * self.rotation_scale,
                rz * self.rotation_scale
            ]
            
        elif report_id == 3:  # Buttons
            buttons = data[1]
            left_pressed = bool(buttons & 0x01)
            right_pressed = bool(buttons & 0x02)
            
        return delta_pos, rot, left_pressed, right_pressed
    
    def controller_loop(self):
        """Main controller loop - reads SpaceMouse and sends data to server"""
        try:
            print("Reading SpaceMouse input... Press Ctrl+C to exit.")
            
            # For tracking button state changes
            last_left_pressed = False
            last_right_pressed = False
            
            # Main control loop
            while self.running:
                # Process SpaceMouse input
                delta_pos, rot, left_pressed, right_pressed = self.process_spacemouse_input()
                
                # Handle gripper control based on button presses
                if left_pressed and not last_left_pressed:
                    # Left button pressed - open gripper
                    self.gripper_state = min(255.0, self.gripper_state + 25.0)
                    print(f"Opening gripper: {self.gripper_state}")
                
                if right_pressed and not last_right_pressed:
                    # Right button pressed - close gripper
                    self.gripper_state = max(0.0, self.gripper_state - 25.0)
                    print(f"Closing gripper: {self.gripper_state}")
                
                # Update button state tracking
                last_left_pressed = left_pressed
                last_right_pressed = right_pressed
                
                # Print movement data occasionally (for debugging)
                if any(abs(v) > 0 for v in delta_pos) or any(abs(v) > 0 for v in rot):
                    print(f"Movement: {delta_pos}, Rotation: {rot}, Gripper: {self.gripper_state}")
                
                # Send data to server
                if self.connected:
                    if not self.send_control_data(delta_pos, rot, self.gripper_state):
                        print("Connection lost, attempting to reconnect...")
                        if self.connect_to_server():
                            print("Reconnected successfully")
                        else:
                            time.sleep(1)  # Wait before retry
                else:
                    if self.connect_to_server():
                        print("Connected successfully")
                    else:
                        time.sleep(1)  # Wait before retry
                
                # Sleep to maintain update rate
                time.sleep(0.01)  # 100 Hz update rate
                
        except Exception as e:
            print(f"Controller error: {e}")
            import traceback
            traceback.print_exc()
    
    def start(self):
        """Start the controller"""
        if self.running:
            print("Controller is already running")
            return
        
        # Find and connect to SpaceMouse
        if not self.find_spacemouse():
            print("Failed to find SpaceMouse. Controller not started.")
            return
        
        # Connect to robot server
        self.connect_to_server()
        
        # Start controller
        self.running = True
        
        # Start controller thread
        self.controller_thread = threading.Thread(target=self.controller_loop)
        self.controller_thread.daemon = True
        self.controller_thread.start()
        print("Controller started")
    
    def stop(self):
        """Stop the controller"""
        self.running = False
        self.disconnect()
        
        # Wait for thread to end
        if hasattr(self, 'controller_thread') and self.controller_thread.is_alive():
            self.controller_thread.join(timeout=1.0)
        
        print("Controller stopped")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SpaceMouse controller for Franka arm")
    parser.add_argument('--host', type=str, default='localhost', help='Server hostname')
    parser.add_argument('--port', type=int, default=12345, help='Server port')
    
    args = parser.parse_args()
    
    try:
        controller = SpaceMouseFrankaController(args.host, args.port)
        controller.start()
        
        print("Controller running. Press Ctrl+C to exit.")
        # Keep main thread alive until keyboard interrupt
        while controller.running:
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopping controller...")
    finally:
        if 'controller' in locals():
            controller.stop()