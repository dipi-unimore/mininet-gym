# adversarial_agents.py
import time, random, threading, traceback
from colorama import Fore
from utility.constants import LONG_ATTACK, NORMAL, SHORT_ATTACK, TrafficTypes
from utility.my_log import information, debug, error
from mininet.net import Mininet
from mininet.node import Host

from utility.network_attacks import AttackType, launch_attack_smart, launch_udp_flood, launch_http_flood, launch_icmp_flood, launch_tcp_syn_flood

global_is_dos_attack_active = False


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


def continuous_traffic_generation(net_env, options = {"show_normal_traffic": True}):
    """
    Continuously iterate over hosts and assign tasks (normal traffic or attack).
    This function runs in a separate thread and stops when the stop_event is set.

    Args:
        net: Mininet network object.
    """
    net = net_env.net
    net_env.host_tasks = {}
    net_env.host_threads = []
    only_one_can_attack = options.get("only_one_can_attack", False)

    for host in net.hosts:
        if only_one_can_attack:
            time.sleep(0.5)  # Stagger thread starts for attack-only mode
        thread = threading.Thread(
            target=host_task,
            args=(host, net_env, options),
            daemon=True
        )
        thread.start()
        net_env.host_threads.append(thread)
   
    debug("All host threads started")
    
def host_task(host: Host, net_env, options):
    """
    Function to assign and execute a task for a single host during gym_type = 'Attack'
    Args:
        host: Host object from Mininet.
        net_env: Network environment object.
    """
    net = net_env.net
    host_tasks = net_env.host_tasks
    show_normal_traffic = options.get("show_normal_traffic", False)
    generate_only_attacks = options.get("send_only_attacks", False)
    generate_only_normal_traffic = options.get("send_only_normal_traffic", False)
    generate_only_tcp_traffic = options.get("send_only_tcp_traffic", False)
    generate_only_udp_traffic = options.get("send_only_udp_traffic", False)
        
    
    while (not hasattr(net_env,'stop_event') or not net_env.stop_event.is_set()):
        try: 
            while hasattr(net_env,'pause_event') and net_env.pause_event.is_set(): 
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
            
            #Skip if is generate only attacks and there is an ongoing attack in the network
            if generate_only_attacks:
                ongoing_attack = any(
                    ht.get("task_type") in [SHORT_ATTACK, LONG_ATTACK] and ht.get("end_time", 0) > time.time()
                    for ht in host_tasks.values()
                )
                if ongoing_attack:
                    debug(Fore.BLUE + f"{host.name} is waiting since there is an ongoing attack in the network\n" + Fore.WHITE)
                    time.sleep(random.uniform(1, 3))
                    continue

            # Assign a new task to the host
            if generate_only_attacks:
                task_type = random.choice([SHORT_ATTACK, LONG_ATTACK])
            elif generate_only_normal_traffic or (generate_only_tcp_traffic or generate_only_udp_traffic):
                task_type = NORMAL
            else:
                task_type = choose_task_type(net_env)
            destination = random.choice([h for h in net.hosts if h != host])
            start_time = time.time()
            task_duration = 0

            try:
                debug("Inizio assegnazione task per host: " + host.name)
                if task_type == "normal": 
                    net_env.attack_likely *=  1.1  
                    if generate_only_tcp_traffic:
                        traffic_type = TrafficTypes.TCP
                    elif generate_only_udp_traffic:
                        traffic_type = TrafficTypes.UDP
                    else:         
                        traffic_type = random.choice(net.traffic_types)
                    task_duration = 0 if traffic_type=='none' else random.uniform(2, 5)
                elif task_type == SHORT_ATTACK:
                    task_duration = 5
                elif task_type == LONG_ATTACK:
                    task_duration = 30
                    
                debug(Fore.GREEN + f"{host.name} is assigned {task_type} targeting {destination.name}\n" + Fore.WHITE)

                available_attack_types = [
                        AttackType.UDP_FLOOD,
                        AttackType.SYN_FLOOD,
                        AttackType.ICMP_FLOOD,
                    ]

                # Store the task details for the host
                #host_tasks[host.name] = {
                task_info = {
                    "task_type": task_type,
                    "traffic_type": traffic_type if task_type == NORMAL else "attack",
                    "destination": destination.name,
                    "end_time": start_time + task_duration,
                }
                if task_type != NORMAL:
                    attack_method = random.choice(available_attack_types)
                    task_info.update({"attack_subtype": attack_method["name"]})  # Add start_time for debugging
                
                # Start delayed registration thread (0.01s delay for task to start)
                registration_thread = threading.Thread(
                    target=delayed_task_registration,
                    args=(net_env, host.name, task_info, 0.03),
                    daemon=True
                )
                registration_thread.start()                
                
                #net_env.host_tasks = host_tasks
                        
                if task_type == NORMAL and task_duration>0:
                    color = Fore.GREEN if show_normal_traffic else Fore.BLACK                    
                    generate_normal_traffic(host, destination, traffic_type, duration = task_duration, color = color)
                elif task_type == SHORT_ATTACK or task_type == LONG_ATTACK: 
                    success = launch_attack_smart(
                        attacker=host,
                        victim=destination,
                        duration=task_duration,
                        attack_type=attack_method
                    )                                                         
                    if not success:
                        error(f"Error while {host.name} is assigned {task_type} targeting {destination.name}.")
                    else:
                        debug(f"{host.name} has finished attacking {destination.name} after duration {task_duration}")
                        time.sleep(0.5)  # Ensure attack process has finished to propagate
                    # Reset task both for error and success
                    task_duration = 0
                    host_tasks[host.name] = {
                        "task_type": NORMAL,
                        "traffic_type": "none",
                        "destination": "",
                        "end_time": start_time + task_duration,
                    }
                    net_env.host_tasks = host_tasks
                    continue                        
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

def delayed_task_registration(net_env, host_name, task_info, delay=0.1):
    """
    Register a task in host_tasks after a delay to ensure the process has started.
    
    Args:
        net_env: Network environment object
        host_name: Name of the host
        task_info: Dictionary with task information
        delay: Delay in seconds before registration (default 0.5s for attacks)
    """
    time.sleep(delay)
    net_env.host_tasks[host_name] = task_info
    debug(f"Task registered for {host_name}: {task_info['task_type']}")

def host_task_async(host: Host, net_env, options):
    """
    Function to assign and execute a task for a single host during gym_type = 'Attack'
    """
    net = net_env.net
    host_tasks = net_env.host_tasks
    show_normal_traffic = options.get("show_normal_traffic", True)
    generate_only_attacks = options.get("send_only_attacks", False)
    generate_only_normal_traffic = options.get("send_only_normal_traffic", False)
    generate_only_tcp_traffic = options.get("send_only_tcp_traffic", False)
    generate_only_udp_traffic = options.get("send_only_udp_traffic", False)
    
    while (not hasattr(net_env,'stop_event') or not net_env.stop_event.is_set()):
        try: 
            while hasattr(net_env,'pause_event') and net_env.pause_event.is_set(): 
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
            
            # Skip if is generate only attacks and there is an ongoing attack in the network
            if generate_only_attacks:
                ongoing_attack = any(
                    ht.get("task_type") in [SHORT_ATTACK, LONG_ATTACK] and ht.get("end_time", 0) > time.time()
                    for ht in host_tasks.values()
                )
                if ongoing_attack:
                    debug(Fore.BLUE + f"{host.name} is waiting since there is an ongoing attack in the network\n" + Fore.WHITE)
                    time.sleep(random.uniform(1, 3))
                    continue

            # Assign a new task to the host
            if generate_only_attacks:
                task_type = random.choice([SHORT_ATTACK, LONG_ATTACK])
            elif generate_only_normal_traffic or (generate_only_tcp_traffic or generate_only_udp_traffic):
                task_type = NORMAL
            else:
                task_type = choose_task_type(net_env)
            
            destination = random.choice([h for h in net.hosts if h != host])
            task_duration = 0

            try:
                debug("Inizio assegnazione task per host: " + host.name)
                
                # Determine traffic type and duration
                if task_type == "normal": 
                    net_env.attack_likely *= 1.1  
                    if generate_only_tcp_traffic:
                        traffic_type = TrafficTypes.TCP
                    elif generate_only_udp_traffic:
                        traffic_type = TrafficTypes.UDP
                    else:         
                        traffic_type = random.choice(net.traffic_types)
                    task_duration = 0 if traffic_type == 'none' else random.uniform(2, 5)
                elif task_type == SHORT_ATTACK:
                    task_duration = 5
                elif task_type == LONG_ATTACK:
                    task_duration = 30
                    
                debug(Fore.GREEN + f"{host.name} is assigned {task_type} targeting {destination.name}\n" + Fore.WHITE)

                # ===== CRITICAL CHANGE: Store task info AFTER successful start =====
                
                # Initialize start_time here (before task execution)
                start_time = time.time()
                
                if task_type == NORMAL and task_duration > 0:
                    # For normal traffic, store immediately (starts quickly)
                    host_tasks[host.name] = {
                        "task_type": task_type,
                        "traffic_type": traffic_type,
                        "destination": destination.name,
                        "end_time": start_time + task_duration,
                        "start_time": start_time,  # Add start_time for debugging
                    }
                    net_env.host_tasks = host_tasks
                    
                    color = Fore.GREEN if show_normal_traffic else Fore.BLACK                    
                    generate_normal_traffic(host, destination, traffic_type, duration=task_duration, color=color)
                    
                elif task_type == SHORT_ATTACK or task_type == LONG_ATTACK:
                    # For attacks, start the attack FIRST, then store task info
                    attack_started = launch_dos_attack_hping3_async(
                        attacker=host, 
                        victim=destination, 
                        duration=task_duration
                    )
                    
                    if not attack_started:
                        error(f"Error while {host.name} is assigned {task_type} targeting {destination.name}.")
                        # Set a placeholder task to prevent immediate retry
                        task_duration = 1
                        host_tasks[host.name] = {
                            "task_type": NORMAL,
                            "traffic_type": "none",
                            "destination": "",
                            "end_time": time.time() + task_duration,
                            "start_time": time.time(),
                        }
                        net_env.host_tasks = host_tasks
                    else:
                        # Attack successfully started - NOW store the task info
                        actual_start_time = time.time()  # Use actual start time
                        host_tasks[host.name] = {
                            "task_type": task_type,
                            "traffic_type": "attack",
                            "destination": destination.name,
                            "end_time": actual_start_time + task_duration,
                            "start_time": actual_start_time,
                        }
                        net_env.host_tasks = host_tasks
                        debug(f"Attack task stored for {host.name} at {actual_start_time}")
                        
                debug(f"Task completion for host: {host.name}")
                
            except Exception as e:
                error(Fore.RED + f"Error while {host.name} is assigned {task_type} targeting {destination.name}\n{traceback.format_exc()}\n" + Fore.WHITE)
                task_duration = 1
                host_tasks[host.name] = {
                    "task_type": NORMAL,
                    "traffic_type": "none",
                    "destination": "",
                    "end_time": time.time() + task_duration,
                    "start_time": time.time(),
                }
                net_env.host_tasks = host_tasks
            finally:
                time.sleep(task_duration + random.uniform(net_env.params.wait_after_read, net_env.params.wait_after_read + 3))
                
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
            
            # Kill any stale iperf processes on this port
            if wait_for_host_ready(dst_host, timeout=1.0):
                dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                time.sleep(0.1)
            
            # Start server with output to file
            wait_for_host_ready(dst_host, timeout=2.0)
            
            server_log = f"/tmp/iperf_server_{dst_host.name}_{port_number}.log"
            server_cmd = f"timeout {duration+3}s iperf -s -p {port_number} >{server_log} 2>&1 &"
            
            try:
                dst_host.cmd(server_cmd)
            except AssertionError:
                error(f"Failed to start server on {dst_host.name} - host still busy")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            # CRITICAL: Wait longer for TCP server to fully bind
            time.sleep(0.5)  # Increased from 0.3
            
            # Verify server started and is listening
            if wait_for_host_ready(dst_host, timeout=1.0):
                # Check process exists
                check_process = dst_host.cmd(f"pgrep -f 'iperf.*-s.*{port_number}'").strip()
                if not check_process:
                    error(f"iperf server process not found on {dst_host.name}:{port_number}")
                    # Check server log for error
                    server_output = dst_host.cmd(f"cat {server_log} 2>/dev/null")
                    error(f"Server log: {server_output}")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
                
                # Check if port is actually listening
                check_port = dst_host.cmd(f"netstat -tln | grep ':{port_number}' || ss -tln | grep ':{port_number}'").strip()
                if not check_port:
                    error(f"Port {port_number} not listening on {dst_host.name}")
                    # Check server log
                    server_output = dst_host.cmd(f"cat {server_log} 2>/dev/null")
                    error(f"Server log: {server_output}")
                    # Kill the process and retry with different port
                    dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
                
                debug(f"Server confirmed listening on {dst_host.name}:{port_number}")
            
            # Start client in background with monitoring
            wait_for_host_ready(src_host, timeout=2.0)
            
            client_pid_file = f"/tmp/iperf_client_{src_host.name}_{port_number}.pid"
            client_log_file = f"/tmp/iperf_client_{src_host.name}_{port_number}.log"
            
            # Test connection first with a quick ping
            connectivity_test = src_host.cmd(f"ping -c 1 -W 1 {dst_host.IP()}").strip()
            if "1 received" not in connectivity_test and "1 packets received" not in connectivity_test:
                error(f"No connectivity between {src_host.name} and {dst_host.name}")
                dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            try:
                # Start client with better error handling
                client_cmd = (
                    f"bash -c 'timeout {duration+1}s iperf -c {dst_host.IP()} "
                    f"-p {port_number} -t {duration} -i 1 "
                    f">{client_log_file} 2>&1 & echo $! >{client_pid_file}'"
                )
                src_host.cmd(client_cmd)
            except AssertionError:
                error(f"Failed to start client on {src_host.name} - host still busy")
                # Clean up server
                if wait_for_host_ready(dst_host, timeout=1.0):
                    dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                release_port(port_number)
                return TrafficTypes.NONE, None, None
            
            # Give client a moment to start
            time.sleep(0.3)
            
            # Verify client started
            if wait_for_host_ready(src_host, timeout=1.0):
                client_pid = src_host.cmd(f"cat {client_pid_file} 2>/dev/null").strip()
                if client_pid:
                    client_check = src_host.cmd(f"ps -p {client_pid} -o pid=").strip()
                    if not client_check:
                        error(f"iperf client failed to start on {src_host.name}")
                        client_output = src_host.cmd(f"cat {client_log_file} 2>/dev/null")
                        error(f"Client log: {client_output[:300]}")
                        # Clean up
                        dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
                        release_port(port_number)
                        return TrafficTypes.NONE, None, None
                    else:
                        debug(f"Client started with PID {client_pid}")
            
            # Wait for client to finish
            max_wait = duration + 2
            for i in range(int(max_wait * 2)):  # Check every 0.5s
                time.sleep(0.5)
                if wait_for_host_ready(src_host, timeout=0.5):
                    pid_check = src_host.cmd(f"test -f {client_pid_file} && kill -0 $(cat {client_pid_file}) 2>/dev/null && echo 'running' || echo 'done'")
                    if 'done' in pid_check:
                        break
            
            # Small delay before reading outputs
            time.sleep(0.3)
            
            # Read outputs
            if wait_for_host_ready(src_host, timeout=2.0):
                client_output = src_host.cmd(f"cat {client_log_file} 2>/dev/null")
                src_host.cmd(f"rm -f {client_log_file} {client_pid_file} 2>/dev/null")
            
            time.sleep(0.5)
            if wait_for_host_ready(dst_host, timeout=2.0):
                server_output = dst_host.cmd(f"cat {server_log} 2>/dev/null")
                dst_host.cmd(f"rm -f {server_log} 2>/dev/null")
            
            # Ensure cleanup
            dst_host.cmd(f"pkill -9 -f 'iperf.*{port_number}' 2>/dev/null")
            
            # Check if data was actually transferred
            if client_output:
                # Look for transfer statistics in client output
                if "0 Bytes" in client_output or "0.00 Bytes" in client_output:
                    error(f"TCP transfer failed - 0 bytes transferred from {src_host.name} to {dst_host.name}:{port_number}")
                    error(f"Client output: {client_output[:400]}")
                    if server_output:
                        error(f"Server output: {server_output[:400]}")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
                elif "Connection refused" in client_output:
                    error(f"Connection refused on {dst_host.name}:{port_number}")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
                elif "No route to host" in client_output:
                    error(f"No route to {dst_host.name}")
                    release_port(port_number)
                    return TrafficTypes.NONE, None, None
            
            if color is not Fore.BLACK:
                color = Fore.MAGENTA if color is None else color
                information(Fore.WHITE + f"{src_host.name} " + color + f"sent TCP traffic to "+ Fore.WHITE + f"{dst_host.name}:{port_number}\n")  
            else:
                debug(Fore.WHITE + f"{src_host.name} " + color + f"sent TCP traffic to "+ Fore.WHITE + f" {dst_host.name}:{port_number}\n")
        
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



