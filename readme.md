<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="left">
  <a href="https://github.com/dipi-unimore/mininet-gym-icaart-2026">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h2 align="center">MininetGym</h3>

  <p>
    This project aims to provide a basic framework for DDoS mitigation using reinforcement learning (Deep and not).
    The network is implemented using Mininet (based on Software-Defined networking).
    The design of the solution is inspired by the work 
  </p>
  <h3 align="center">
    <strong ><em> <a href="https://www.sciencedirect.com/science/article/pii/S235271102500278X">"MininetGym: A modular SDN-based simulation environment for reinforcement learning in cybersecurity"</a></em></strong> <br>
    by <strong><a href="https://finix77.github.io/index.html">Finistrella S.</a></strong> at al.</h5>


  <p>
    <a href="https://github.com/dipi-unimore/mininet-gym-icaart-2026/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/dipi-unimore/mininet-gym-icaart-2026/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

## About The Project
A Modular SDN-based Simulation Environment for Reinforcement Learning in Cybersecurity
Real-time traffic generation and flow monitoring via Mininet and Custom Gym environments for traffic classification and DoS attack detection.

![Schema Screen Shot][schema-screenshot]
**Framework Architecture:** The schema illustrates the framework architecture. A deep explanation can be found in the section on the cited **[mininet-gym-paper]**.

![Product Screen Shot][product-screenshot]
**Web Interface for Flow Monitoring**: Real-time visualization of network traffic and DDoS attack metrics registered per second during a simulation.


---

## Built With

* [Python](https://www.python.org/)
* [Mininet](http://mininet.org/)
* [OpenDayLight (ODL)](https://www.opendaylight.org/)
* [Gymnasium / OpenAI Gym](https://gymnasium.farama.org/)
* [PyTorch](https://pytorch.org/) / [NumPy](https://numpy.org/)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

This section provides instructions on how to set up and run the MininetGym project on a clean Ubuntu 20.04+ system.

### Prerequisites

Ensure you have the following installed on your system:
* **Python 3.11** or later
* **Mininet**: A network emulator that creates a network of virtual hosts, switches, controllers, and links.
* **OpenDayLight (ODL)**: A modular open-source platform for Software-Defined Networking (SDN). Java 1.8.0 or later is required for ODL.

### Installation

Follow these steps to get your development environment set up.

1.  **Install Mininet, hping3, and System Dependencies**

    The project requires the **`hping3`** tool for network attack simulation. Make sure it is installed along with Mininet.

    ```bash
    sudo apt update && sudo apt upgrade -y
    sudo apt-get install mininet python3-venv git -y
    # Installazione di hping3
    sudo apt-get install -y hping3
    ```
    
    **Verifica e Pulizia:**
    Verification and Cleanup: Run a simple Mininet test and clean the environment to ensure that `hping3` is recognized by the virtual hosts.
    
    sudo mn --test pingall
    sudo mn -c           
   
    You can verify the `hping3` installation with: `hping3 --help` or `which hping3`.

2.  **Clone the Repository**
    Create a new directory for your project, navigate into it, and clone your repository.

    ```bash
    mkdir MininetGym
    cd MininetGym
    git clone [https://github.com/dipi-unimore/mininet-gym.git](https://github.com/dipi-unimore/mininet-gym.git)
    cd mininet-gym
    ```

3.  **Create and Activate a Python Virtual Environment**
    It is crucial to use a virtual environment to manage project dependencies and avoid conflicts with system packages. This also prevents the `externally-managed-environment` error.

    ```bash
    # Create a new virtual environment named 'venv'
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate
    ```
    You will see `(venv)` prepended to your terminal prompt, indicating that the virtual environment is active.

4.  **Install Python Dependencies**
    With your virtual environment activated, install the required Python packages using `requirements.txt`.

    ```bash
    pip install -r requirements.txt
    ```
    If you encounter an error related to `ale-py`, you may need to update the version in the `requirements.txt` file (e.g., to `ale-py>=0.11.0`).

5. **Troubleshooting: libcudnn.so Errors**
    If you encounter an error like `ImportError: libcudnn.so.9: cannot open shared object file: No such file or directory`, it means that your PyTorch installation is configured to use NVIDIA GPU acceleration but cannot find the necessary libraries. You have two options to resolve this:

    Option A: Install the CPU-only version of PyTorch (Recommended)
    This is the simplest and safest solution, especially if you do not have a dedicated NVIDIA GPU. It avoids the need for any CUDA or cuDNN libraries.

    ```bash
    pip uninstall torch
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    ```    

    Option B: Install the Full CUDA Toolkit and cuDNN (If you have an NVIDIA GPU)
    If your machine has a compatible NVIDIA GPU and you want to use it for training, you\'ll need to install the full CUDA Toolkit and the cuDNN library. This is a more complex process and is not covered in detail in this guide. You should refer to NVIDIA\'s official documentation for instructions on how to install the appropriate versions of CUDA and cuDNN for your system.




6.  **Run a Sample**
    Mininet requires root privileges to create virtual network devices. 
    
    ```bash
    sudo python3 main.py
    ```
    This command will start a Mininet simulation, run your environment, and begin the training process.

7.  **OpenDayLight (ODL) Controller Setup (Optional)**
    For installing the OpenDayLight controller, follow the instructions provided in the [ODL-Ubuntu22-installation] guide. This project was developed with ODL Karaf version 0.8.4.

    You might also use a Docker container for ODL. To start an OpenDayLight controller container:
    ```bash
    docker run -d -t -v ~/.m2:/root/.m2/ -p 6633:6633 -p 8101:8101 -p 8181:8181 --net=bridge --hostname=ovsdb-cluster-node-1 --name=opendaylight opendaylight/opendaylight:0.18.2 [https://github.com/sfuhrm/docker-opendaylight](https://github.com/sfuhrm/docker-opendaylight)
    ```
    To connect via SSH to the ODL controller inside the Docker container on a virtual machine (e.g., 192.168.1.226):
    ```bash
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null admin@192.168.1.226 -p 8101
    ```

    Ensure your `JAVA_PATH` is correctly set, especially if you are running ODL directly and not via Docker.
    ```bash
    echo 'export JAVA_PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin/java' >> ~/.bashrc
    source ~/.bashrc
    ```
    Adjust the path to your Java installation accordingly.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

*This section is not yet filled out.*

## Roadmap

*This section is not yet filled out.*

## Contributing

*This section is not yet filled out.*

## License

Distributed under the MIT License – see the `LICENSE.txt` file for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

*This section is not yet filled out.*

## Acknowledgments

*This section is not yet filled out.*

---
[contributors-shield]: https://img.shields.io/github/contributors/dipi-unimore/mininet-gym.svg?style=for-the-badge
[contributors-url]: https://github.com/dipi-unimore/mininet-gym/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/dipi-unimore/mininet-gym.svg?style=for-the-badge
[forks-url]: https://github.com/dipi-unimore/mininet-gym/network/members
[stars-shield]: https://img.shields.io/github/stars/dipi-unimore/mininet-gym.svg?style=for-the-badge
[stars-url]: https://github.com/dipi-unimore/mininet-gym/stargazers
[issues-shield]: https://img.shields.io/github/issues/dipi-unimore/mininet-gym.svg?style=for-the-badge
[issues-url]: https://github.com/dipi-unimore/mininet-gym/issues
[license-shield]: https://img.shields.io/github/license/dipi-unimore/mininet-gym.svg?style=for-the-badge
[license-url]: https://github.com/dipi-unimore/mininet-gym/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/salvo-finistrella-970034237
[mininet-gym-paper]: https://www.sciencedirect.com/science/article/pii/S235271102500278X
[product-screenshot]: images/screenshot.png
[schema-screenshot]: images/architecture.png
[ODL-Ubuntu22-installation]: https://docs.opendaylight.org/en/stable-fluorine/downloads.html
