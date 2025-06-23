<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <a href="https://github.com/dipi-unimore/mininet-gym">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">MininetGym</h3>

  <p align="center">
    Reinforcement learning Mininet OpenDayLight
    This project aims to provide a basic framework for DDoS mitigation using reinforcement learning (Deep and not).
    The network is implemented using Mininet (based on Software defined networking).
    The design of the solution is inspired by the work "???" by Salvo Finistrella and others here.
    <br />
    <a href="https://github.com/dipi-unimore/mininet-gym"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://github.com/dipi-unimore/mininet-gym/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/dipi-unimore/mininet-gym/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
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

[![Product Screen Shot][product-screenshot]]
[![Schema Screen Shot][schema-screenshot]]

Reinforcement learning Mininet OpenDayLight

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

This section provides instructions on how to set up and run the MininetGym project.

### Prerequisites

Ensure you have the following installed on your system:
* **Python 3.11**
* **Mininet**: A network emulator that creates a network of virtual hosts, switches, controllers, and links.
* **OpenDayLight (ODL)**: A modular open-source platform for Software-Defined Networking (SDN).
* **Java**: Required for OpenDayLight. Version 1.8.0 or later is recommended.

### Installation

Follow these steps to get your development environment set up:

1.  **Mininet Installation**
    Mininet is crucial for this simulation framework. It is recommended to install Mininet on a clean Ubuntu x.x LTS system. For detailed instructions, refer to the official Mininet documentation or a guide specific to your OS version. A common installation method involves:
    ```bash
    git clone [https://github.com/mininet/mininet](https://github.com/mininet/mininet)
    mininet/util/install.sh -a
    ```

2.  **OpenDayLight (ODL) Controller Installation**
    For installing OpenDayLight controller, follow the instructions provided in the [ODL-Ubuntu22-installation] guide. This project was developed with ODL Karaf version 0.8.4 and Java 1.8.0. This is not mandatory, it depends by environment configuration.

    You might also use a Docker container for ODL. To start an OpenDayLight controller container:
    ```bash
    docker run -d -t -v ~/.m2:/root/.m2/ -p 6633:6633 -p 8101:8101 -p 8181:8181 --net=bridge --hostname=ovsdb-cluster-node-1 --name=opendaylight opendaylight/opendaylight:0.18.2 [https://github.com/sfuhrm/docker-opendaylight](https://github.com/sfuhrm/docker-opendaylight)
    ```
    To connect via SSH to the ODL controller inside the Docker container on a virtual machine (e.g., 192.168.1.226):
    ```bash
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null admin@192.168.1.226 -p 8101
    ```
    (Note: Replace `admin` with your ODL username and `192.168.1.226` with your VM's IP if different).

3.  **Set JAVA_PATH (if using local ODL installation)**
    Ensure your `JAVA_PATH` is correctly set, especially if you are running ODL directly and not via Docker:
    ```bash
    echo 'export JAVA_PATH=/usr/lib/jvm/java-21-openjdk-amd64/bin/java' >> ~/.bashrc
    source ~/.bashrc
    ```
    Adjust the path to your Java installation accordingly.

4.  **Create and Activate a Python Virtual Environment**
    It's highly recommended to use a virtual environment to manage project dependencies.
    ```bash
    # Create a virtual environment named 'MYenv11' using Python 3.11
    python3.11 -m venv MYenv11
    # Alternatively, if you have 'virtualenv' installed:
    # virtualenv --python="/usr/bin/python3.11" "MYenv11"

    # Activate the virtual environment
    source MYenv11/bin/activate
    ```
    You will see `(MYenv11)` prepended to your terminal prompt, indicating that the virtual environment is active.

5.  **Install Python Dependencies**
    With your virtual environment activated, install the required Python packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    This will install the following libraries:
    * `Flask==3.0.3`
    * `Flask-SocketIO==5.3.0`
    * `Flask-Cors==4.0.0`
    * `eventlet==0.36.1`
    * `python-engineio==4.9.1`
    * `python-socketio==5.11.0`
    * `requests==2.32.3`
    * `numpy==1.26.4`
    * `pandas==2.2.2`
    * `matplotlib==3.8.4`
    * `scikit-learn==1.4.2`
    * `scipy==1.13.1`
    * `gymnasium==0.29.1`
    * `stable-baselines3==2.3.0`
    * `sb3-contrib==2.3.0`
    * `Mininet==2.3.0`
    * `colorama==0.4.6`
    * `pyyaml==6.0.1`
    * `lxml==5.2.2`
    * `beautifulsoup4==4.12.3`

    Your environment is now set up!

6.  **Create configuration**
    Starting from base_config.yaml, create your own configuration, following the instruction of the paper (not published yet)

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
[product-screenshot]: images/screenshot.png
[schema-screenshot]: images/schema.png
[ODL-Ubuntu22-installation]: https://docs.opendaylight.org/en/stable-fluorine/downloads.html
[ODL-karaf-0.8.4]: https://docs.opendaylight.org/en/stable-fluorine/downloads.html
[use-different-python-version-with-virtualenv]: https://stackoverflow.com/questions/35579976/how-to-use-a-different-python-version-with-virtualenv