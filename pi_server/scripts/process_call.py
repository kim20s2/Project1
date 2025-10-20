import subprocess
import time
import sys
import signal
import os

class ProcessManager:
    def __init__(self):
        self.rtsp_server = None
        self.f_estimator = None
        self.p_estimator = None
        self.recorder = None
        self.xml_mixer = None # Added for the xml_mix method

    # --- Start Methods (Same as original) ---

    def rtsp_start(self, target):
        if target == 'file':
            rtsp_name = 'rtsp_file'
        elif target == 'a6700':
            rtsp_name = 'rtsp_a6700'
        elif target == 'webcam':
            rtsp_name = 'rtsp_webcam'

        self.rtsp_server = subprocess.Popen(
            [sys.executable, '-m', f'scripts.gstreamer.{rtsp_name}'],
            stdout=subprocess.PIPE
        )
        print(f"RTSP Server (PID: {self.rtsp_server.pid}) started.")

    def f_estimator_start(self):
        self.f_estimator = subprocess.Popen(
            [sys.executable, '-m', 'scripts.face_est.face_est_main',
            '--camera', 'rtsp://127.0.0.1:8554/test'], 
            stdout=subprocess.DEVNULL, 
        )
        print(f"Face Estimator (PID: {self.f_estimator.pid}) started.")

    def p_estimator_start(self):
        self.p_estimator = subprocess.Popen(
            [sys.executable, '-m', 'scripts.pose_est.pose_est_main',
            '--hef', 'models/vit_pose_small.hef',
            '--camera', 'rtsp://127.0.0.1:8554/test',
            '--conf', '0.4',
            '--width', '192',
            '--height', '256'], 
            stdout=subprocess.DEVNULL, 
            preexec_fn=os.setsid # Make it a process group leader
        )
        print(f"Pose Estimator (PID: {self.p_estimator.pid}) started.")

    def recorder_start(self):
        self.recorder = subprocess.Popen(
            [sys.executable, '-m', 'scripts.gstreamer.recorder'],
            stdout=subprocess.DEVNULL
        )
        print(f"Recorder (PID: {self.recorder.pid}) started.")

    def xml_mix(self):
        self.xml_mixer = subprocess.Popen(
            [sys.executable, '-m', 'scripts.xml_mix'],
            stdout=subprocess.DEVNULL
        )
        print(f"XML Mixer (PID: {self.xml_mixer.pid}) started.")

    # --- Helper method for handling termination logic ---
    
    def _terminate_process(self, process_attr_name, process_name_str, use_pgkill=False):
        """
        Helper function to safely terminate and verify a process.

        :param process_attr_name: The name of the process attribute stored in self (e.g., 'rtsp_server')
        :param process_name_str: A string name for the process for logging (e.g., 'RTSP Server')
        :param use_pgkill: Whether to kill the entire process group (for p_estimator)
        """
        process = getattr(self, process_attr_name)
        
        if process is None:
            print(f"{process_name_str} is not running or has already been terminated.")
            return

        # Check if the process has already terminated
        if process.poll() is not None:
            print(f"{process_name_str} (PID: {process.pid}) has already terminated (Exit code: {process.poll()}).")
            setattr(self, process_attr_name, None)
            return

        print(f"Attempting to terminate {process_name_str} (PID: {process.pid})...")
        
        try:
            if use_pgkill:
                # If the process was started with preexec_fn=os.setsid (like p_estimator),
                # we must send a signal to the process group (PGID) to terminate child processes.
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGTERM) # Send SIGTERM to the group instead of SIGINT
                print(f"Sent SIGTERM to process group {pgid}.")
            else:
                # In the normal case, send SIGINT to the process
                process.send_signal(signal.SIGINT)
                print("Sent SIGINT.")

            # Wait 5 seconds for termination
            process.wait(timeout=5)
            print(f"{process_name_str} (PID: {process.pid}) terminated successfully (Exit code: {process.returncode}).")

        except subprocess.TimeoutExpired:
            print(f"{process_name_str} (PID: {process.pid}) did not respond within 5 seconds. Attempting force kill (SIGKILL)...")
            try:
                if use_pgkill:
                    pgid = os.getpgid(process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    process.kill() # Send SIGKILL
                
                process.wait(timeout=2) # Wait for OS to clean up
                print(f"{process_name_str} (PID: {process.pid}) was forcibly killed.")
            except Exception as e:
                print(f"Error during force kill of {process_name_str}: {e}")

        except (ProcessLookupError, PermissionError) as e:
            # If the process disappeared right as we sent the signal
            print(f"Error terminating {process_name_str} (PID: {process.pid}) (Process not found or permission error): {e}")
        
        except Exception as e:
            # (e.g., os.getpgid failed, etc.)
            print(f"Unexpected error while terminating {process_name_str} (PID: {process.pid}): {e}")
        
        finally:
            # Whether successful or not, remove from the manager
            setattr(self, process_attr_name, None)

    # --- Modified Finish Methods ---

    def rtsp_server_finish(self):
        self._terminate_process('rtsp_server', 'RTSP Server')

    def f_estimator_finish(self):
        self._terminate_process('f_estimator', 'Face Estimator')

    def p_estimator_finish(self):
        # p_estimator used os.setsid, so use use_pgkill=True
        self._terminate_process('p_estimator', 'Pose Estimator', use_pgkill=True)

    def recorder_finish(self):
        self._terminate_process('recorder', 'Recorder')

    def xml_mixer_finish(self):
        # Added termination method for xml_mix
        self._terminate_process('xml_mixer', 'XML Mixer')

    # --- Convenience method to terminate all processes ---
    
    def finish_all(self):
        print("\n--- Initiating shutdown of all processes ---")
        # First, terminate dependents (estimators, recorder)
        self.f_estimator_finish()
        self.p_estimator_finish()
        self.recorder_finish()
        self.xml_mixer_finish()
        
        # Finally, terminate the stream source (RTSP server)
        self.rtsp_server_finish()
        print("--- All processes shut down complete ---")