# Phase 2: Robotics Tools Setup Documentation

Since we are adapting to a native Windows workflow for Phase 3-8 (to extract the bag seamlessly using Python) while still supporting a full ROS 2 Foxy installation for future use, here is the documentation of our setup steps.

## 1. Native Windows Python Environment
**Reason:** The fastest and most reliable way to extract data from ROS 2 bags (`.db3`) without needing a heavy ROS 2 installation is to use pure Python. We are using the `rosbags` library which parses SQLite3 databases directly.
**Virtual Environment:** You already have a virtual environment at `c:\Users\sanmi\Desktop\projects\RL\MuJoCo\.venv`. We will reuse this environment to keep all project dependencies isolated and clean.

**Command to install data extraction dependencies:**
```powershell
& "c:\Users\sanmi\Desktop\projects\RL\MuJoCo\.venv\Scripts\python.exe" -m pip install rosbags opencv-python pandas numpy scipy matplotlib open3d pyyaml
```
*We will run this automatically.*

## 2. Installing ROS 2 Foxy in WSL (Manual Step)
**Reason:** You requested ROS 2 Foxy. Foxy requires **Ubuntu 20.04**, but your current WSL has Ubuntu 22.04. Installing WSL instances often triggers a Windows UAC (Admin) prompt that cannot be bypassed by automation scripts. 

**Steps for you to run (Optional right now, but required for RViz2 later):**
1. Open a new PowerShell terminal as Administrator.
2. Run the following command to install the correct Ubuntu version:
   ```powershell
   wsl --install -d Ubuntu-20.04
   ```
3. Set up your username and password when prompted.
4. Once inside the Ubuntu 20.04 terminal, follow the standard ROS 2 Foxy installation steps for Ubuntu:
   ```bash
   sudo apt update && sudo apt install locales
   sudo locale-gen en_US en_US.UTF-8
   sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
   export LANG=en_US.UTF-8

   sudo apt install software-properties-common
   sudo add-apt-repository universe

   sudo apt update && sudo apt install curl -y
   sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

   echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

   sudo apt update
   sudo apt install ros-foxy-desktop python3-colcon-common-extensions -y
   ```

## Next Steps
While you install Ubuntu 20.04 in the background (if you choose to do so now), I am proceeding with **Phase 3 & 4 (Inspect the ROS Bag & Detect Sensors)** natively on Windows using your virtual environment. All outputs will be saved to your Windows dataset directory.
