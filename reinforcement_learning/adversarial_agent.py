# adversarial_agents.py
import time, random, threading, traceback
from colorama import Fore
from utility.constants import LONG_ATTACK, NORMAL, SHORT_ATTACK, TRAFFIC_TYPE_ID_MAPPING, TrafficTypes
from utility.my_log import information, debug, error
from mininet.net import Mininet
from mininet.node import Host

def generate_random_traffic(net: Mininet):
    """
    Function to generate random traffic between hosts in the Mininet network.
    Args:
    - net: Mininet network object
    """
    information(Fore.CYAN + "Environment "+Fore.WHITE) 
    if net.hosts is None :
        information(Fore.WHITE + "No hosts created\n")
        return TrafficTypes.NONE,None,None
    
    traffic_type = random.choice(net.traffic_types)
    return generate_traffic(net, traffic_type, 1)

def choose_task_type(net_env):
    """
    Choose the task type based on the attack likely.
    Args:        attack_likely (float): Probability of choosing an attack (0 to 0.5).
    Returns:        str: The chosen task type ("normal", "short_attack", or "long_attack").
    """
    attack_likely = max(0, min(net_env.attack_likely, 0.5))
    
    rand = random.random()
    now = time.time()
    no_attack_timeout = 15
    total_lenght_short_attack = 15
    total_lenght_long_attack = 50
    percentage_likelyhood_short_attack = 0.66
    
    if now < net_env.last_long_attack_timestamp or now < net_env.last_short_attack_timestamp:
        return NORMAL
    elif rand < attack_likely * percentage_likelyhood_short_attack: 
        net_env.last_short_attack_timestamp = time.time() + total_lenght_short_attack + no_attack_timeout
        net_env.attack_likely=net_env.init_attack_likely
        percentage_likelyhood_short_attack -= 0.07
        return SHORT_ATTACK
    elif rand < attack_likely: 
        net_env.last_long_attack_timestamp = time.time() + total_lenght_long_attack + no_attack_timeout
        net_env.attack_likely=net_env.init_attack_likely
        percentage_likelyhood_short_attack += 0.1
        return LONG_ATTACK
    else:
        return NORMAL


def continuous_traffic_generation(net_env, show_normal_traffic = True):
    """
    Continuously iterate over hosts and assign tasks (normal traffic or attack).
    This function runs in a separate thread and stops when the stop_event is set.

    Args:
        net: Mininet network object.
    """
    net = net_env.net
    net_env.host_tasks = {}
    net_env.host_threads = []

    for host in net.hosts:
        thread = threading.Thread(
            target=host_task,
            args=(host, net_env, show_normal_traffic),
            daemon=True
        )
        thread.start()
        net_env.host_threads.append(thread)
   
    debug("All host threads started")
    
def host_task(host: Host, net_env, show_normal_traffic):
    """
    Function to assign and execute a task for a single host during gym_type = 'Attack'
    Args:
        host: Host object from Mininet.
        net_env: Network environment object.
    """
    net = net_env.net
    host_tasks = net_env.host_tasks
    
    while not net_env.stop_event.is_set():
        try: 
            while net_env.pause_event.is_set(): 
                time.sleep(1)
                continue
            
            # Skip if the host is already performing a task
            if host_tasks.get(host.name, {}).get("end_time", 0) > time.time():
                ht_old = host_tasks.get(host.name, {})
                task_type = ht_old.get("task_type")
                destination = ht_old.get("destination")
                traffic_type = ht_old.get("traffic_type") if task_type == "normal" else "attack"
                debug(Fore.BLUE + f"{host.name} is YET assigned {task_type} {traffic_type} targeting {destination}\n" + Fore.WHITE)
                time.sleep(random.uniform(1, 3))
                continue

            # Assign a new task to the host
            task_type = choose_task_type(net_env)
            destination = random.choice([h for h in net.hosts if h != host])
            start_time = time.time()
            task_duration = 0

            try:
                debug("Inizio assegnazione task per host: " + host.name)
                if task_type == "normal": 
                    net_env.attack_likely *=  1.1           
                    traffic_type = random.choice(net.traffic_types)
                    task_duration = 0 if traffic_type=='none' else random.uniform(2, 5)
                elif task_type == SHORT_ATTACK:
                    task_duration = 5
                elif task_type == LONG_ATTACK:
                    task_duration = 30
                    
                debug(Fore.GREEN + f"{host.name} is assigned {task_type} targeting {destination.name}\n" + Fore.WHITE)

                # Store the task details for the host
                host_tasks[host.name] = {
                    "task_type": task_type,
                    "traffic_type": traffic_type if task_type == NORMAL else "attack",
                    "destination": destination.name,
                    "end_time": start_time + task_duration,
                }
                net_env.host_tasks = host_tasks
                        
                if task_type == NORMAL and task_duration>0:
                    color = Fore.GREEN if show_normal_traffic else Fore.BLACK
                    generate_normal_traffic(host, destination, traffic_type, duration = task_duration, color = color)
                elif task_type == SHORT_ATTACK or task_type == LONG_ATTACK:                    
                    if not launch_dos_attack_hping3(attacker=host, victim=destination, duration=task_duration):
                        error(f"Error while {host.name} is assigned long_attack targeting {destination.name}. hping3 process stopped unexpectedly on {host.name} at second 0.")
                    else:
                        debug(f"{host.name} is attacking {destination.name} for duration {task_duration}")
                debug(f"Task completion for host: {host.name}")
            except Exception as e:
                error(Fore.RED + f"Error while {host.name} is assigned {task_type} targeting {destination.name}\n" + Fore.WHITE)
                task_duration = 1
                host_tasks[host.name] = {
                    "task_type": NORMAL,
                    "traffic_type": "none",
                    "destination": "",
                    "end_time": start_time + task_duration,
                }
                net_env.host_tasks = host_tasks
            finally:
                time.sleep(task_duration+random.uniform(net_env.params.wait_after_read, net_env.params.wait_after_read + 3))
        except Exception as e:
            error(Fore.RED + f"Error in {host.name}\n{traceback.format_exc()}\n" + Fore.WHITE)
        
    debug(f"{host.name} thread finished")

def generate_traffic(net: Mininet, traffic_type, task_duration: None):  
    """
    Generate traffic type among to random two hosts.
    Args:
        net: Network.
        traffic_type: Type of traffic ("tcp", "udp", "ping", "none").
    """
    if traffic_type == TrafficTypes.NONE:
        information(Fore.WHITE + "No traffic generated\n")
        return TrafficTypes.NONE, None, None
    
    src_host = random.choice(net.hosts)
    dst_host = random.choice([host for host in net.hosts if host != src_host])
    task_duration = random.uniform(2, 5) if task_duration is None else task_duration
    return generate_normal_traffic(src_host, dst_host, traffic_type, task_duration)


def is_actual_error(output: str, traffic_type: str) -> bool:
    """
    Determine if output contains actual errors (not benign warnings).
    
    Args:
        output: Command output string
        traffic_type: Type of traffic being generated
        
    Returns:
        True if there's an actual error, False otherwise
    """
    if not output:
        return False
    
    output_lower = output.lower()
    
    # Benign messages that should be IGNORED (these are normal)
    benign_patterns = [
        "warning: ack of last datagram failed",  # Normal UDP behavior
        "waiting for server threads",             # Normal iperf shutdown
        "interrupt again to force quit",          # Normal iperf message
        "server report:",                         # Normal iperf output
    ]
    
    # Check if this is just a benign message
    for pattern in benign_patterns:
        if pattern in output_lower:
            # This line contains a benign pattern
            # Check if there are OTHER lines with real errors
            lines = output.split('\n')
            for line in lines:
                line_lower = line.lower()
                # Skip lines with benign patterns
                if any(bp in line_lower for bp in benign_patterns):
                    continue
                # Check for real errors in other lines
                if any(err in line_lower for err in ["error", "failed", "unable", "invalid"]):
                    return True
            return False  # Only benign warnings found
    
    # Real error patterns (these indicate actual problems)
    real_errors = [
        "connection refused",
        "no route to host",
        "network is unreachable",
        "bind failed",
        "address already in use",
        "connect failed",
        "unable to connect",
        "command not found",
        "permission denied",
        "failed to",
        "cannot",
        "fatal",
    ]
    
    # Check for real errors
    for error_pattern in real_errors:
        if error_pattern in output_lower:
            return True
    
    # Check for generic error/failed but exclude benign contexts
    if ("error" in output_lower or "failed" in output_lower) and not any(bp in output_lower for bp in benign_patterns):
        # Additional check: iperf statistics lines often contain "failed" in context that's OK
        if "[" in output and "]" in output and "datagram" in output_lower:
            return False  # This is just iperf statistics
        return True
    
    return False


def is_serious_warning(output: str) -> bool:
    """
    Determine if output contains warnings worth logging.
    
    Args:
        output: Command output string
        
    Returns:
        True if there's a serious warning, False otherwise
    """
    if not output:
        return False
    
    output_lower = output.lower()
    
    # Benign warnings to ignore
    benign_warnings = [
        "warning: ack of last datagram failed",
        "waiting for server threads",
    ]
    
    for pattern in benign_warnings:
        if pattern in output_lower:
            return False
    
    # Serious warnings
    serious_warnings = [
        "unreachable",
        "timeout",
        "packet loss",
        "congestion",
    ]
    
    for warning in serious_warnings:
        if warning in output_lower:
            return True
    
    return "warning" in output_lower


# Port management to avoid conflicts
_used_ports = set()
_port_lock = threading.Lock()

def get_available_port(min_port=10000, max_port=60000):
    """
    Get an available port number, avoiding recently used ports.
    Thread-safe port allocation.
    """
    with _port_lock:
        max_attempts = 100
        for _ in range(max_attempts):
            port = random.randint(min_port, max_port)
            if port not in _used_ports:
                _used_ports.add(port)
                # Clean up old ports if the set gets too large
                if len(_used_ports) > 1000:
                    _used_ports.clear()
                return port
        # Fallback: just return a random port and hope for the best
        return random.randint(min_port, max_port)

def release_port(port):
    """Release a port back to the pool after a delay."""
    def delayed_release():
        time.sleep(5)  # Wait for port to be fully released by OS
        with _port_lock:
            _used_ports.discard(port)
    
    threading.Thread(target=delayed_release, daemon=True).start()


def wait_for_host_ready(host, timeout=5.0):
    """
    Wait for a host's shell to be ready for commands.
    
    Args:
        host: Mininet host object
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if host is ready, False if timeout
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # Check if shell exists and is not waiting
            if host.shell and not host.waiting:
                return True
        except AttributeError:
            # Host might not be fully initialized
            pass
        time.sleep(0.1)
    
    return False


def is_host_busy(host):
    """
    Check if a host is currently busy executing a command.
    
    Args:
        host: Mininet host object
        
    Returns:
        True if host is busy, False if available
    """
    try:
        return host.waiting if hasattr(host, 'waiting') else False
    except:
        return True  # Assume busy if we can't determine


def generate_normal_traffic(src_host, dst_host, traffic_type, duration: int, color = None):
    """
    Generate normal traffic between two hosts.
    Args:
        src_host: Source host object.
        dst_host: Destination host object.
        traffic_type: Type of traffic ("tcp", "udp", "ping", "http").
        duration: Duration of traffic generation in seconds.
        color: Color for logging (optional).
    """
    server_output = ''
    client_output = ''
    port_number = None
    
    try:
        # Wait for hosts to be ready (with reasonable timeout)
        max_wait_time = 10.0  # Maximum 10 seconds wait for hosts to be ready
        
        debug(f"Waiting for {src_host.name} to be ready...")
        if not wait_for_host_ready(src_host, timeout=max_wait_time):
            error(f"Source host {src_host.name} still busy after {max_wait_time}s - attempting anyway")
            # Don't return None - try to proceed anyway, it might work
        
        debug(f"Waiting for {dst_host.name} to be ready...")
        if not wait_for_host_ready(dst_host, timeout=max_wait_time):
            error(f"Destination host {dst_host.name} still busy after {max_wait_time}s - attempting anyway")
            # Don't return None - try to proceed anyway
        if traffic_type == TrafficTypes.PING:
            # Generate ping traffic
            client_output = src_host.cmd(f"ping -c {int(duration)} {dst_host.IP()} 2>&1 &")
            
            if color is not Fore.BLACK:
                color = Fore.GREEN if color is None else color
                information(Fore.WHITE + f"{src_host.name} " + color + f"is pinging "+ Fore.WHITE + f"{dst_host.name}\n")    
            else:
                debug(Fore.WHITE + f"{src_host.name} " + color + f"is pinging "+ Fore.WHITE + f"{dst_host.name}\n")  

        elif traffic_type == TrafficTypes.UDP:
            # Generate UDP traffic using iperf with unique port
            port_number = get_available_port()
            
            # Kill any stale iperf processes on this host first
            # Use shorter timeout here since we already waited above
            if wait_for_host_ready(dst_host, timeout=1.0):
                dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                time.sleep(0.05)
            else:
                debug(f"{dst_host.name} busy during cleanup, skipping pkill")
            
            # Start server with stderr/stdout redirected to avoid mixing outputs
            # Wait briefly for dst_host to be ready
            wait_for_host_ready(dst_host, timeout=2.0)
            
            server_cmd = f"timeout {duration+2}s iperf -s -u -p {port_number} >/tmp/iperf_server_{dst_host.name}_{port_number}.log 2>&1 &"
            try:
                dst_host.cmd(server_cmd)
            except AssertionError:
                error(f"Failed to start server on {dst_host.name} - host still busy")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            time.sleep(0.2)  # Give server time to bind to port
            
            # Verify server started
            if wait_for_host_ready(dst_host, timeout=1.0):
                check = dst_host.cmd(f"pgrep -f 'iperf.*{port_number}'")
                if not check.strip():
                    error(f"Failed to start iperf server on {dst_host.name}:{port_number}")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
            
            # Start client (use background with process monitoring)
            # Wait for src_host to be ready
            wait_for_host_ready(src_host, timeout=2.0)
            
            client_pid_file = f"/tmp/iperf_client_{src_host.name}_{port_number}.pid"
            client_log_file = f"/tmp/iperf_client_{src_host.name}_{port_number}.log"
            
            try:
                src_host.cmd(f"bash -c 'timeout {duration+1}s iperf -c {dst_host.IP()} -u -p {port_number} -t {duration} >{client_log_file} 2>&1 & echo $! >{client_pid_file}'")
            except AssertionError:
                error(f"Failed to start client on {src_host.name} - host still busy")
                # Clean up server
                if wait_for_host_ready(dst_host, timeout=1.0):
                    dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            # Wait for client to finish (with timeout)
            max_wait = duration + 2
            for i in range(int(max_wait * 2)):  # Check every 0.5s
                time.sleep(0.5)
                if wait_for_host_ready(src_host, timeout=0.5):
                    pid_check = src_host.cmd(f"test -f {client_pid_file} && kill -0 $(cat {client_pid_file}) 2>/dev/null && echo 'running' || echo 'done'")
                    if 'done' in pid_check:
                        break
            
            # Read outputs
            if wait_for_host_ready(src_host, timeout=2.0):
                client_output = src_host.cmd(f"cat {client_log_file} 2>/dev/null")
                src_host.cmd(f"rm -f {client_log_file} {client_pid_file} 2>/dev/null")
            
            time.sleep(0.5)
            if wait_for_host_ready(dst_host, timeout=2.0):
                server_output = dst_host.cmd(f"cat /tmp/iperf_server_{dst_host.name}_{port_number}.log 2>/dev/null")
                dst_host.cmd(f"rm -f /tmp/iperf_server_{dst_host.name}_{port_number}.log 2>/dev/null")
            
            if color is not Fore.BLACK:
                color = Fore.YELLOW if color is None else color
                information(Fore.WHITE + f"{src_host.name} " + color + f"is sending UDP traffic to "+ Fore.WHITE + f"{dst_host.name}:{port_number}\n")  
            else:
                debug(Fore.WHITE + f"{src_host.name} " + color + f"is sending UDP traffic to "+ Fore.WHITE + f"{dst_host.name}:{port_number}\n")  

        elif traffic_type == TrafficTypes.TCP:
            # Generate TCP traffic using iperf with unique port
            port_number = get_available_port()
            
            # Kill any stale iperf processes on this host first
            if wait_for_host_ready(dst_host, timeout=1.0):
                dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                time.sleep(0.05)
            else:
                debug(f"{dst_host.name} busy during cleanup, skipping pkill")
            
            # Start server with output to file
            wait_for_host_ready(dst_host, timeout=2.0)
            
            server_cmd = f"timeout {duration+2}s iperf -s -p {port_number} >/tmp/iperf_server_{dst_host.name}_{port_number}.log 2>&1 &"
            try:
                dst_host.cmd(server_cmd)
            except AssertionError:
                error(f"Failed to start server on {dst_host.name} - host still busy")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            time.sleep(0.3)  # TCP needs more setup time
            
            # Verify server started
            if wait_for_host_ready(dst_host, timeout=1.0):
                check = dst_host.cmd(f"pgrep -f 'iperf.*{port_number}'")
                if not check.strip():
                    error(f"Failed to start iperf server on {dst_host.name}:{port_number}")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
            
            # Start client in background with monitoring
            wait_for_host_ready(src_host, timeout=2.0)
            
            client_pid_file = f"/tmp/iperf_client_{src_host.name}_{port_number}.pid"
            client_log_file = f"/tmp/iperf_client_{src_host.name}_{port_number}.log"
            
            try:
                src_host.cmd(f"bash -c 'timeout {duration+1}s iperf -c {dst_host.IP()} -p {port_number} -t {duration} >{client_log_file} 2>&1 & echo $! >{client_pid_file}'")
            except AssertionError:
                error(f"Failed to start client on {src_host.name} - host still busy")
                # Clean up server
                if wait_for_host_ready(dst_host, timeout=1.0):
                    dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            # Wait for client to finish
            max_wait = duration + 2
            for i in range(int(max_wait * 2)):
                time.sleep(0.5)
                if wait_for_host_ready(src_host, timeout=0.5):
                    pid_check = src_host.cmd(f"test -f {client_pid_file} && kill -0 $(cat {client_pid_file}) 2>/dev/null && echo 'running' || echo 'done'")
                    if 'done' in pid_check:
                        break
            
            # Read outputs
            if wait_for_host_ready(src_host, timeout=2.0):
                client_output = src_host.cmd(f"cat {client_log_file} 2>/dev/null")
                src_host.cmd(f"rm -f {client_log_file} {client_pid_file} 2>/dev/null")
            
            time.sleep(0.5)
            if wait_for_host_ready(dst_host, timeout=2.0):
                server_output = dst_host.cmd(f"cat /tmp/iperf_server_{dst_host.name}_{port_number}.log 2>/dev/null")
                dst_host.cmd(f"rm -f /tmp/iperf_server_{dst_host.name}_{port_number}.log 2>/dev/null")
            
            if color is not Fore.BLACK:
                color = Fore.MAGENTA if color is None else color
                information(Fore.WHITE + f"{src_host.name} " + color + f"is sending TCP traffic to "+ Fore.WHITE + f"{dst_host.name}:{port_number}\n")  
            else:
                debug(Fore.WHITE + f"{src_host.name} " + color + f"is sending TCP traffic to "+ Fore.WHITE + f" {dst_host.name}:{port_number}\n")
        
        # Check for ACTUAL errors (not benign warnings)
        if server_output:
            if is_actual_error(server_output, traffic_type):
                error(Fore.RED + f"Error in server {dst_host.name} while receiving {traffic_type} traffic from {src_host.name}:\n{server_output}\n" + Fore.WHITE)
            elif is_serious_warning(server_output):
                debug(Fore.YELLOW + f"Warning in server {dst_host.name} while receiving {traffic_type} traffic from {src_host.name}:\n{server_output[:200]}...\n" + Fore.WHITE)
            else:
                # Benign output - just debug log it
                debug(f"Server {dst_host.name} completed normally")
        
        if client_output:
            if is_actual_error(client_output, traffic_type):
                error(Fore.RED + f"Error in client {src_host.name} while sending {traffic_type} traffic to {dst_host.name}:\n{client_output}\n" + Fore.WHITE)
            elif is_serious_warning(client_output):
                debug(Fore.YELLOW + f"Warning in client {src_host.name} while sending {traffic_type} traffic to {dst_host.name}:\n{client_output[:200]}...\n" + Fore.WHITE)
            else:
                # Benign output - just debug log it
                debug(f"Client {src_host.name} completed normally")
        
        return traffic_type, src_host, dst_host
        
    except Exception as e:
        error(Fore.RED + f"Exception in generate_normal_traffic: {type(e).__name__}: {str(e)}\n{traceback.format_exc()}" + Fore.WHITE)
        return TrafficTypes.NONE, None, None
    finally:
        # Release the port after use
        if port_number is not None:
            release_port(port_number)


global_is_dos_attack_active = False

def launch_dos_attack_hping3(attacker: Host, victim: Host, duration: int = 15):
    """
    Launch a DoS attack from the attacker to the victim.
    
    Args:
        attacker: Attacking host object.
        victim: Victim host object.
        duration: Duration of the attack in seconds.
    """
    information(Fore.RED + f"{attacker.name} is attacking {victim.name} for duration {duration}\n" + Fore.WHITE)
    victim_ip = victim.IP()

    # Ensure `hping3` is installed
    hping_check = attacker.cmd("which hping3")
    if not hping_check.strip():
        msg = (f"{attacker.name} does not have hping3 installed.\n"
               f"Install hping3 on vm with mn:\n"
               f"  sudo apt-get update\n"
               f"  sudo apt-get install -y hping3\n"
               f"Verify: hping3 --help\n"
               f"Restart Mininet: sudo mn -c")
        error(msg)
        raise Exception(msg)        
    
    # Start hping3 in the background with a timeout
    attack_cmd = f"timeout {duration}s hping3 --flood --udp {victim_ip} > /dev/null 2>&1 &"
    attacker.cmd(attack_cmd)
    
    # Monitor the attack
    for sec in range(duration):
        process_check = attacker.cmd("pgrep -f hping3")        
        if not process_check.strip():
            error(Fore.RED + f"hping3 process stopped unexpectedly on {attacker.name} at second {sec}.\n" + Fore.WHITE)
            return False
        else:
            debug(f"hping3 still running on {attacker.name} at second {sec}")
        
        time.sleep(1)

    # Stop the attack
    attacker.cmd("killall hping3 2>/dev/null")
    information(Fore.YELLOW + f"{attacker.name} has stopped attacking {victim.name}\n" + Fore.WHITE)
    return True

def test_dos_attack(net):
    """Test DoS attack functionality"""
    attacker, victim = net.hosts[0], net.hosts[1]
    if launch_dos_attack_hping3(attacker, victim, duration=5):
        information(Fore.GREEN + f"DoS attack test from {attacker.name} to {victim.name} completed successfully.\n" + Fore.WHITE)
    else:
        error(Fore.RED + f"DoS attack test from {attacker.name} to {victim.name} failed.\n" + Fore.WHITE)
    time.sleep(2) 
    net.stop()