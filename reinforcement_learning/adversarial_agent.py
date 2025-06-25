# adversarial_agents.py
import time, random, threading, json as jsonlib, traceback
from colorama import Fore, Back, Style
from utility.my_log import set_log_level, information, debug, error, notify_client
from utility.network_configurator import create_network
from utility.params import Params
from mininet.net import Mininet
from mininet.node import Host


# def generate_random_traffic_for_duration(net, duration=60):
#     """
#     Function to generate random traffic between hosts in the Mininet network.
#     Args:
#     - net: Mininet network object
#     - duration: Total duration to run the traffic generation
#     """
#     end_time = time.time() + duration
#     while time.time() < end_time:
#         if net.hosts is None:
#             continue
#         generate_random_traffic(net)
        
#         # Wait for a random interval before generating the next traffic
#         time.sleep(random.uniform(1, 3))

def generate_random_traffic(net: Mininet):
    """
    Function to generate random traffic between hosts in the Mininet network.
    Args:
    - net: Mininet network object
    """
    # Randomly choose a type of traffic to generate
    information(Fore.CYAN + "\nEnvironment "+Fore.WHITE) 
    if net.hosts is None :
        information(Fore.WHITE + "No hosts created\n")
        return 0,None,None
    
    traffic_type = random.choice(net.traffic_types)
    return generate_traffic(net,traffic_type,1)

def choose_task_type(net_env):
    """
    Choose the task type based on the attack probability.
    Args:        attack_probability (float): Probability of choosing an attack (0 to 0.5).
    Returns:        str: The chosen task type ("normal", "short_attack", or "long_attack").
    """
    # Ensure attack_probability is within bounds
    attack_probability = max(0, min(net_env.attack_probability, 0.5))
    
    # Generate a random number to decide the task
    rand = random.random()
    now = time.time()
    no_attack_timeout = 15
    total_lenght_short_attack = 15
    total_lenght_long_attack = 50
    percentage_likelyhood_short_attack = 0.66
    if now < net_env.last_long_attack_timestamp or now < net_env.last_short_attack_timestamp: #to check last time an attack generated
        return "normal"
    elif rand < attack_probability * percentage_likelyhood_short_attack: 
        net_env.last_short_attack_timestamp = time.time() + total_lenght_short_attack + no_attack_timeout
        net_env.attack_probability=net_env.init_attack_probability
        percentage_likelyhood_short_attack -= 0.07
        return "short_attack"
    elif rand < attack_probability: 
        net_env.last_long_attack_timestamp = time.time() + total_lenght_long_attack + no_attack_timeout
        net_env.attack_probability=net_env.init_attack_probability
        percentage_likelyhood_short_attack += 0.1
        return "long_attack"
    else:
        return "normal"


def continuous_traffic_generation(net_env, stop_event, pause_event, show_normal_traffic = True):
    """
    Continuously iterate over hosts and assign tasks (normal traffic or attack).
    This function runs in a separate thread and stops when the stop_event is set.

    Args:
        net: Mininet network object.
        stop_event: threading.Event to signal when to stop the function.
        attack_probability (float): Probability of generating an attack.
    """
    net = net_env.net
    net_env.host_tasks = {}
    threads = []
    for host in net.hosts:
        thread = threading.Thread(
            target=host_task,
            args=(host, net_env, stop_event, pause_event, show_normal_traffic),
            daemon=True
        )
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete when the stop event is set.
    for thread in threads:
        thread.join()
    
    debug("All host threads finished")
    
def host_task(host: Host, net_env, stop_event, pause_event, show_normal_traffic):
    """
    Function to assign and execute a task for a single host during gym_type = 'Attack'
    Args:
        host: Host object from Mininet.
        net_env: Network environment object.
        stop_event: threading.Event to signal when to stop the function.
        attack_probability: Probability of generating an attack.
    """
    net = net_env.net
    host_tasks = net_env.host_tasks  # Shared dictionary for host tasks
    while not stop_event.is_set():
        try: 
            while pause_event.is_set():
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
                if task_type == "normal": 
                    net_env.attack_probability+=net_env.attack_probability*0.01 #add a 1 percent to likelyhood               
                    traffic_type = random.choice(net.traffic_types)
                    task_duration = 0 if traffic_type=='none' else random.uniform(2, 5)  # Random short traffic duration.
                elif task_type == "short_attack":
                    task_duration = 5  # Short attack duration.
                elif task_type == "long_attack":
                    task_duration = 30  # Long attack duration.
                    
                # Log task assignment
                debug(Fore.GREEN + f"{host.name} is assigned {task_type} targeting {destination.name}\n" + Fore.WHITE)

                # Store the task details for the host
                host_tasks[host.name] = {
                    "task_type": task_type,
                    "traffic_type": traffic_type if task_type == "normal" else "attack",
                    "destination": destination.name,
                    "end_time": start_time + task_duration,
                }
                net_env.host_tasks = host_tasks
                        
                if task_type == "normal" and task_duration>0:
                    color = Fore.GREEN if show_normal_traffic else Fore.BLACK
                    generate_normal_traffic(host, destination, traffic_type, duration = task_duration, color = color)
                elif task_type == "short_attack" or task_type == "long_attack":
                    launch_dos_attack(host, destination, duration=task_duration)
                
            except Exception as e:
                # Handle errors (e.g., host is busy, network issues)
                error(Fore.RED + f"Error while {host.name} is assigned {task_type} targeting {destination.name}\n" + Fore.WHITE)
                task_duration = 1
                host_tasks[host.name] = {
                    "task_type": "normal",
                    "traffic_type": "none",
                    "destination": "",
                    "end_time": start_time + task_duration,
                }
                net_env.host_tasks = host_tasks
            finally:
                # Wait a bit before assigning a new task
                time.sleep(task_duration+random.uniform(0, 3))
        except Exception as e:
            # Handle errors (e.g., host is busy, network issues)
            error(Fore.RED + f"Error in {host.name}\n{traceback.format_exc()}\n" + Fore.WHITE)
        
    debug(f"{host.name} thread finished")

def generate_traffic(net: Mininet, traffic_type, task_duration: None):  
    """
    Generate traffic type among to random two hosts.
    Args:
        net: Network.
        traffic_type: Type of traffic ("tcp", "udp", "ping", "http").
    """
    if traffic_type == "none":
        information(Fore.WHITE + "No traffic generated\n")
        return 0,None,None
    
    # Randomly select two hosts
    src_host = random.choice(net.hosts)
    dst_host = random.choice([host for host in net.hosts if host != src_host])
    task_duration = random.uniform(2, 5) if task_duration is None else task_duration
    return generate_normal_traffic(src_host, dst_host, traffic_type, task_duration)
    
def generate_normal_traffic(src_host, dst_host, traffic_type, duration: int, color = None):
    """
    Generate normal traffic between two hosts.
    Args:
        src_host: Source host object.
        dst_host: Destination host object.
        traffic_type: Type of traffic ("tcp", "udp", "ping", "http").
    """        
    if traffic_type == "ping":
        # Generate ping traffic
        src_host.cmd(f"ping -c 4 {dst_host.IP()} &")  # Non-blocking ping
        if color is not Fore.BLACK:
            color = Fore.GREEN if color is None else color
            information(color + f"{src_host.name} is pinging {dst_host.name}\n"+Fore.WHITE )  
        else:
            debug(f"{src_host.name} is pinging {dst_host.name}\n")
        return 1,src_host,dst_host
    
    elif traffic_type == "udp":
        # Generate UDP traffic using iperf
        port_number = random.uniform(1025,40000)
        dst_host.cmd(f"timeout {duration+1}s iperf -s -u -p {port_number} &")  # Start UDP server on destination host
        src_host.cmd(f"iperf -c {dst_host.IP()} -u -p {port_number} -t {duration} &")  # Client-side UDP traffic
        if color is not Fore.BLACK:
            color = Fore.YELLOW if color is None else color
            information(color + f"{src_host.name} is sending UDP traffic to {dst_host.name}\n" + Fore.WHITE)  
        else:
            debug(f"{src_host.name} is sending UDP traffic to  {dst_host.name}\n")
        return 2,src_host,dst_host
    
    elif traffic_type == "tcp":
        # Generate TCP traffic using iperf
        port_number = int(random.uniform(1025,40000))
        server= dst_host.cmd(f"timeout {duration+2}s iperf -s -p {port_number} &")  # Start TCP server on destination host
        #print(f"server {server}")
        time.sleep(0.5)  # Ensure server is ready
        client = src_host.cmd(f"iperf -c {dst_host.IP()} -p {port_number} -t {duration} &")  # Client-side TCP traffic
        #print(f"client {client}")
        if color is not Fore.BLACK:
            color = Fore.MAGENTA if color is None else color
            information(color + f"{src_host.name} is sending TCP traffic to {dst_host.name}\n"+ Fore.WHITE )  
        else:
            debug(f"{src_host.name} is sending TCP traffic to  {dst_host.name}\n")
        return 3,src_host,dst_host
        
    elif traffic_type == "http":
        src_host.cmd(f"curl -s {dst_host.IP()} &")  # HTTP traffic.
        if color is not Fore.BLACK:
            color = Fore.CYAN if color is None else color
            information(color + f"{src_host.name} is sending HTTP requests to {dst_host.name}\n" + Fore.WHITE)
        else:
            debug(f"{src_host.name} is sending HTTP requests to  {dst_host.name}\n")
        return 4,src_host,dst_host
    
def print_traffic(traffic_type, src_host_name, dst_host_name):   
    if traffic_type == "none":
        information(Fore.WHITE + "No traffic generated\n")
        
    if traffic_type == "ping":
        # Generate ping traffic
        information(Fore.GREEN + f"{src_host_name} is pinging {dst_host_name}\n"+Fore.WHITE )  
    
    elif traffic_type == "udp":
        # Generate UDP traffic using iperf
        information(Fore.YELLOW + f"{src_host_name} is sending UDP traffic to {dst_host_name}\n" + Fore.WHITE)  
    
    elif traffic_type == "tcp":
        # Generate TCP traffic using iperf
        information(Fore.MAGENTA + f"{src_host_name} is sending TCP traffic to {dst_host_name}\n"+ Fore.WHITE )  
    elif traffic_type == "http":
        # Generate TCP traffic using iperf
        information(Fore.OR + f"{src_host_name} is sending HTTP traffic to {dst_host_name}\n"+ Fore.WHITE )  
 
 
 
# def start_random_traffic_thread(net, duration=60):
#     """
#     Start the random traffic generation in a separate thread.
#     """
#     traffic_thread = threading.Thread(target=generate_random_traffic, args=(net, duration))
#     traffic_thread.start()
#     return traffic_thread


def launch_dos_attack(attacker, victim, duration):
    """
    Launch a DoS attack from the attacker to the victim. https://github.com/geraked/miniattack/blob/master/miniattack/net.py
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
        msg = f"{attacker.name} does not have hping3 installed.\nInstall hping3 on vm with mn: sudo apt-get update\nsudo apt-get install -y hping3\nVerify the installation: hping3 --help\nRestart Mininet: sudo mn -c"
        error(msg)
        raise Exception(msg)        

    # Record start and end times
    # start_time = time.time()
    # end_time = start_time + duration
    # while time.time() < end_time:
    #     # Execute the attack
    #     #attack_output = attacker.cmd(f"hping3 -S --flood -p 80 {victim_ip} &")
    #     attack_output = attacker.cmd(f"hping3 --flood --udp {victim_ip} &")
    #     if not attack_output.strip():
    #         error(Fore.RED+f"ERROR: No output from hping3 on {attacker.name}. Is hping3 working?\n"+Fore.WHITE)
    #         time.sleep(0.5)
    #     elif "TCP" in attack_output:
    #         msg = f"{attacker.name} was not able to attack {victim_ip}"
    #         error(msg)
    #         raise Exception(msg)               
    #     else:
    #         debug(f"hping3 started on {attacker.name} targeting {victim.name}:\n{attack_output}\n")
    #         time.sleep(1)  # Prevent overwhelming the system
    
    # Start hping3 in the background with a timeout
    attack_cmd = f"timeout {duration}s hping3 --flood --udp {victim_ip} > /dev/null 2>&1 &"
    attacker.cmd(attack_cmd)
    for sec in range(duration):
        # Check if the hping3 process is running on the attacker
        process_check = attacker.cmd("pgrep -f hping3")        
        if not process_check.strip():
            error(Fore.RED + f"hping3 process stopped unexpectedly on {attacker.name} at second {sec}.\n" + Fore.WHITE)
            raise Exception(msg)   
        else:
            debug(f"hping3 still running on {attacker.name} at second {sec}")
        
        time.sleep(1)

    # Stop the attack
    attacker.cmd("killall hping3")
    information(Fore.YELLOW + f"{attacker.name} has stopped attacking {victim.name}\n" + Fore.WHITE)



if __name__ == '__main__':
    set_log_level('info')
    net_params = { 
            'num_hosts':3,
            'num_switches':1,
            'num_iot':0,
            'controller': {
                'ip':'192.168.1.226',
                'port':6633,
                'usr':'admin',
                'pwd':'admin'
            }
        }

    net_params = jsonlib.loads(jsonlib.dumps(net_params), object_hook=Params)

    launch_dos_attack('10.0.0.1', 80, 60)
    # In each episode Randomly select attacker and bengin user 
    
    for i in range(episode_count=100):
        print("Episode "+str(i))
        no_of_hosts = 8
        net = create_network(net_params)
        attacking_host_id = random.randint(0, no_of_hosts - 2) # select a random host in between 1 and no_of_hosts - 1
        attacking_host = net.hosts[attacking_host_id]

        benign_host_id = random.choice([i for i in range(0, no_of_hosts - 2) if i not in [attacking_host_id]])
        benign_host = net.hosts[benign_host_id]
        print("host" + str(attacking_host_id) + " is attacking and host" + str(benign_host_id) + " is sending normal requests")

        # Create seperate threads for attacker and benign user
        # t1 = threading.Thread(target=ddos_benign, args=(benign_host,))
        # t2 = threading.Thread(target=ddos_flood, args=(attacking_host,)) 
    
        # t1.start()
        # t2.start()
        
        # t1.join()
        # t2.join()
