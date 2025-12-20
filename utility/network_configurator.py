from threading import Thread
from typing import Any, Dict, List
from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch, Host, Switch
from mininet.link import TCLink
from mininet.cli import CLI
from utility.my_log import set_log_level, information, debug, error
from utility.my_files import drop_privileges, regain_root
from utility.network_flows import initialize_monitoring
from utility.params import Params
from mininet.clean import cleanup
from types import SimpleNamespace
import numpy as np, subprocess, os, re, time, requests, json as jsonlib, math

#https://brianlinkletter.com/2015/04/how-to-use-miniedit-mininets-graphical-user-interface/

def add_default_flows(switch):
    #os.system(f"ovs-ofctl add-flow {switch} priority=100,in_port=1,actions=output:2")
    #os.system(f"ovs-ofctl add-flow {switch} priority=100,in_port=2,actions=output:1")
    os.system(f"ovs-ofctl add-flow {switch} priority=1,actions=normal")
    
def start_cli(net):
    CLI(net)  # Run Mininet CLI
    
def create_host(name=None):
    return Host(name=name)

#def create_network(num_hosts=2, num_switches=1, num_iot=0, controller_ip='127.0.0.1', controller_port=6633):
def create_network(params = { 
            'num_hosts':10,
            'num_switches':1,
            'num_iot':1,
            'controller': {
                'ip':'192.168.1.226',
                'port':6633,
                'usr':'admin',
                'pwd':'admin'
            }
        }, server_user = "server_user"):   
    
    cleanup()  # Clean up any existing Mininet processes
    net = Mininet(controller=RemoteController, link=TCLink, switch=OVSKernelSwitch)

    information('*** Adding controller\n')
    #controller = net.addController('c0', controller=RemoteController, ip=params.controller.ip, port=params.controller.port)
    #controller.usr = params.controller.usr
    #controller.pwd = params.controller.pwd

    information(f'*** Adding {params.num_switches} switches\n')
    switches = []
    for i in range(params.num_switches):
        switch = net.addSwitch(f's{i+1}', cls=OVSKernelSwitch ,protocols='OpenFlow13')
        switches.append(switch)

    information(f'*** Adding {params.num_hosts} hosts\n')
    hosts = [net.addHost(f'h{i+1}', ip=f'10.0.0.{i+1}/24') for i in range(params.num_hosts)]  # Same subnet for all hosts

    information(f'*** Adding IoT {params.num_iot} devices\n')
    iot_devices = [net.addHost(f'iot{i+1}', ip=f'10.0.0.{params.num_hosts + i + 1}/24') for i in range(params.num_iot)]  # Same subnet

    information(f'*** Creating links on {params.num_switches} switches \n')
    links = {}
    for switch in switches:
        for host in hosts:
            link  = net.addLink(host, switch)  # Connect each host to a switch
            #links[host.name] = { "link" : link, "switch": switch, "sw_port" : 0}
            links[host.name] = { "link" : link } #link.intf2.name is sw_port
        for iot_device in iot_devices:
            link = net.addLink(iot_device, switch)  # Connect each IoT device to a switch
            #links[iot_device.name] = { "link" : link, "switch": switch, "sw_port" : 0}
            links[iot_device.name] = { "link" : link }
    net.hosts_links = links 
    net.blocked_hosts = []       

    information('*** Starting network\n')
    net.start()

    information('*** Configuring OpenFlow 1.0 and 1.3 protocol for each switch\n')
    for switch in switches:
        switch.cmd('ovs-vsctl set bridge %s protocols=OpenFlow10,OpenFlow13' % switch.name)
        add_default_flows(switch)

    # information('*** Running ping test\n')
    # net.pingAll()  # Test network connectivity

    if params.start_cli:
        information('*** Starting CLI in a separate thread\n')
        cli_thread = Thread(target=start_cli, args=(net,))
        cli_thread.start()  # Start the CLI in a new thread, non-blocking

    net.switches = switches
    #net.controller = controller
    # net.devices = hosts
    # net.devices.extend(iot_devices)
    net.total_packets_received = 0
    net.total_bytes_received = 0
    net.total_packets_transmitted = 0
    net.total_bytes_transmitted = 0
    net.traffic_types = params.traffic_types
    drop_privileges(server_user)
    
    # 2. Initialize monitoring ONCE (waits for network to be ready)
    if not initialize_monitoring(net, bridge_name="s1"):
        print("Failed to initialize monitoring")
        net.stop()
        exit(1)
    
    return net

def stop(net: Mininet):
    regain_root()
    if net is not None and hasattr(net, 'stop'):
        information('*** Stopping Mininet network\n')
        net.stop()


#Define functions to get network statistics from OpenDaylight controller        
def get_data_controller(controller,switches):
    response = get_response(controller,'')
    if response.status_code == 200:
        #information(response.status_code)
        json = response.json()
        y = jsonlib.dumps(json).replace("flow-node-inventory:","").replace("-","_").replace("opendaylight_port_statistics:","").replace("flow_capable_node_connector_","")
        x = jsonlib.loads(y, object_hook=lambda d: SimpleNamespace(**d))
        #print(f"json {json}")
        #nodes=json.get('nodes', [{}])
        nodes = x.nodes.node
        for node in nodes:
            #print(type(node))
            id =  int(node.description.replace("s",""))
            switches[id-1].data=node            
    elif response.status_code == 404:
        error(f"Error retrieving data: status code 404 (Not Found)")
        #return None  # Return None for a missing switch
    else:
        error(f"Error retrieving data: status code {response.status_code}")
        #return None  # Return None for other errors    
        
def get_response(controller,path):
    url = f"http://{controller.ip}:8181/restconf/operational/opendaylight-inventory:nodes{path}"
    return requests.get(url, auth=(controller.usr, controller.pwd), headers={"Accept": "application/json", "Content-type": "application/json"})


def get_packet_rate_byte_count_from_switch(controller,switch_id):
    table=switch_id-1
    response = get_response(controller,f'/node/{switch_id}/flow-node-inventory:table/{table}')
    if response.status_code == 200:
        json = response.json()
 
        # Find all flows in the response
        flows = json.get('flow-node-inventory:table', [{}])[0].get('flow', [])
        #print(f"flows {flows}")
        if not flows:
            print(f"No flows found for switch {switch_id}.")
            return 0
        
        # Initialize packet rate counters
        total_packet_count = 0
        total_byte_count = 0
        
        # Extract packet count from all flow statistics
        for flow in flows:
            # Access the flow statistics
            flow_stats = flow.get('opendaylight-flow-statistics:flow-statistics', {})
            #print(f"flow_stats {flow_stats}")            
            packet_count = flow_stats.get('packet-count', 0)
            total_packet_count += packet_count
            
            byte_count = flow_stats.get('byte-count', 0)
            total_byte_count += byte_count

        #print(f"Total packet rate for switch {switch_id}: {total_packet_count}")
        #return total_packet_count
        return np.array([total_packet_count, total_byte_count], dtype=np.float32)
    else:
        print(f"Error retrieving packet rate: {response.status_code}")
        return 0


def get_latency_from_controller(controller,switch_id):
    # Use OpenDaylight stats to estimate latency (e.g., round-trip time, delays)
    response = get_response(controller,f'/node/{switch_id}')
    
    if response.status_code == 200:
        # Example of estimating latency (this is placeholder logic)
        stats = response.json()
        # Placeholder logic: Use statistics to estimate latency
        latency = 50  # Example placeholder for real calculations based on controller stats
        return latency
    else:
        error(f"Error retrieving latency: {response.status_code}")
        return 0

def get_connection_status(controller,switch_id):
    response = get_response(controller,f'/node/{switch_id}/flow-node-inventory:flow')
    if response.status_code == 200:
        flow_stats = response.json()
        # Count the number of active flows (indicating established connections)
        active_connections = len(flow_stats['flow-node-inventory:flow'])
        return active_connections
    else:
        error(f"Error retrieving connection status: {response.status_code}")
        return 0

def get_switch_id_by_name(switch_name):
    return switch_name.replace("s", "openflow:")

def get_data_switch_by_id(controller,switch_id):
    response = get_response(controller,f'/node/{switch_id}')
    data = {}
    if response.status_code == 200:
        json = response.json()
        #print(f"json {json}")
        nodes=json.get('node', [{}])
        data["id"]=nodes[0].get('id', {})
        data["hardware"]=nodes[0].get('flow-node-inventory:hardware')
        data["description"]=nodes[0].get('flow-node-inventory:description')
        data["software"]=nodes[0].get('flow-node-inventory:software')
        data["nodes_connector"]=nodes[0].get('node-connector', [{}])
        data["tables"]=nodes[0].get('flow-node-inventory:table', [{}])
        data["group_features"]=nodes[0].get('group-features', {})
        data["switch_features"]=nodes[0].get('opendaylight-group-statistics:group-features', {})
        data["port_number"]=nodes[0].get('flow-node-inventory:port-number')
        return data
    elif response.status_code == 404:
        error(f"Error retrieving data for switch {switch_id}: status code 404 (Not Found)")
        return None  # Return None for a missing switch
    else:
        error(f"Error retrieving data switch {switch_id}: status code {response.status_code}")
        return None  # Return None for other errors    
# Finished function used with SDN controller

#function to get stats from hosts
def get_packet_rate( host):
    command = f"sudo tcpdump -i {host.name}-eth0 -c 1000"
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode

def get_cpu_usage( host):
    output = host.cmd("mpstat 1 1")
    if len(output.strip().split('\n')) < 4:
        print("Unexpected mpstat output format.")
        return  0.0
    try:
        idle  = "".join(ch for ch in output.split('   ')[-1] if ch.isnumeric() or ch == '.')
        #print(f"h1 idle:  {idle}")
        cpu_usage = 100-float(idle)
        #print(f"h1 cpu usage:  {format(100-float(idle),'.2f')}")
        return cpu_usage
    except (ValueError, IndexError):
        print(f"Error parsing CPU usage from command: '{output}'")
        return  0.0
    
    
def get_latency( host1, host2):
    try:
        # Execute the ping command
        result = host1.cmd(f"ping -c 4 {host2.IP()}")
        time.sleep(0.1)  # Add a delay to avoid overlapping commands
        # Split the output and extract the average latency
        # Handle cases where the output might not be formatted as expected
        lines = result.split('\n')
        if len(lines) < 2:
            # If the output doesn't have enough lines, return a default high latency
            print(f"Error: Insufficient ping output between {host1} and {host2}.")
            return float('inf')

        # Try to extract the average latency from the ping result
        # lines[-2] penultima riga, perchè nell'output c'è una riga vuota
        # [-3] terzultimo elemento è avg, avrei min [-4] e max [-2]
        avg_latency = lines[-2].split('/')[-3]
        return float(avg_latency)

    except (IndexError, ValueError) as e:
        # Handle cases where the expected values are not found or can't be converted to float
        print(f"Error: Failed to get latency between {host1} and {host2}: {e}")
        return float('inf')  # Return a high latency if something goes wrong


def get_connection_status( host):
    connections = host.cmd("netstat -an | grep ESTABLISHED | wc -l")
    #print(f"connections:  {connections}")
    return int(connections)
#end of functions to get network statistics from hosts

def format_bytes(num_bytes: int | float | None) -> str:
    """
    Formats a number of bytes into a human-readable string (K, M, G, T, P)
    to maintain a maximum of 3 significant digits, adjusting decimal precision.

    This is based on the decimal (base 1000) system, commonly used for data rates.

    Args:
        num_bytes: The number of bytes (integer or float).

    Returns:
        The formatted string with the appropriate unit.

    Examples:
        34         -> "34"
        1257       -> "1.26K" (1.257 rounded to 1.26)
        20304      -> "20.3K"
        43993939   -> "44.0M"
        1000000000 -> "1.00G"
    """
    # 1. Handle null/zero/negative input
    if num_bytes is None or num_bytes == 0:
        return "0"

    num = abs(num_bytes)

    units = ["", "K", "M", "G", "T", "P"]
    base = 1000
    
    # 2. Determine the unit index (i)
    try:
        # Calculate the power of 1000 (log base 1000)
        i = 0 if num < base else int(math.floor(math.log(num, base)))
    except ValueError:
        # Handle cases where log(num) might be invalid (though covered by num < base)
        return str(num)

    unit_index = min(len(units) - 1, i)
    unit = units[unit_index]
    
    # 3. Calculate the value in the chosen unit
    value = num / (base ** unit_index)
    precision = 0

    # 4. Determine precision based on the value to keep max ~3 significant figures
    if unit_index > 0:
        if value < 10:
            # e.g., 1.25K (3 significant figures) -> 2 decimals
            precision = 2
        elif value < 100:
            # e.g., 20.3K (3 significant figures) -> 1 decimal
            precision = 1
        else:
            # e.g., 999K (3 significant figures) -> 0 decimals
            precision = 0

    # 5. Format the value and handle overflow (e.g., 999.9K -> 1.0M)
    formatted_value = f"{value:.{precision}f}"
    
    # Check if rounding caused an overflow to 1000 (e.g., 999.9 to 1000.0)
    if float(formatted_value) >= 1000 and unit_index < len(units) - 1:
        # If it overflows, re-calculate using the next unit
        next_unit_index = unit_index + 1
        next_unit = units[next_unit_index]
        value = num / (base ** next_unit_index)
        
        # Recalculate precision for the new, smaller value
        if value < 10:
            precision = 2
        elif value < 100:
            precision = 1
        else:
            precision = 0
            
        formatted_value = f"{value:.{precision}f}"
        return f"{formatted_value}{next_unit}"

    return f"{formatted_value}{unit}"

# def get_host_switch_and_port_by_host_name(net: Mininet, host_name):
#     """
#     Retrieves the host object, the connected switch, and the OpenFlow port number 
#     on the switch for the specified host.
#     """
#     host = net.get(host_name)
    
#     # 1. Get the first (and usually only) network interface object of the host
#     # This object is of type Mininet.Interface
#     host_intf = host.intfs[0]
    
#     # Check if the interface is actually connected
#     if not host_intf.link:
#         raise Exception(f"Host {host_name} interface {host_intf.name} is not connected to a link.")

#     # 2. Get the Link object connected to this interface
#     link = host_intf.link
    
#     return get_host_switch_and_port_by_link(link=link, host=host)
    
  
# def get_host_switch_and_port_by_link(link: os.link, host: Host):  
#     # 3. Determine the Switch Interface and the Switch Node
#     # The Link object holds two interfaces (intf1 and intf2).
#     # We find which one belongs to the switch.
    
#     # switch_intf is the interface connected to the link that IS NOT the host's interface
#     if link.intf1.node == host:
#         switch_intf = link.intf2
#     else:
#         # This branch handles cases where the host is connected as intf2
#         switch_intf = link.intf1
    
#     port_num = switch_intf.name  # The OpenFlow port number on the switch

#     # 4. Get the Switch object
#     switch = switch_intf.node
    
#     # 5. Check if the switch object is actually a switch (and not another host)
#     if not switch.isSwitch:
#          raise Exception(f"Error: {switch.name} is not a switch.")
            
#     return host, switch, port_num

def get_switch_name_by_host_name(net: Mininet, host_name:str) -> str:
    switch_port_name = net.hosts_links[host_name]["link"].intf2.name
    items = switch_port_name.split("-")
    return items[0]

def get_switch_port_by_host_name(net: Mininet, host_name:str) -> str:
    switch_port_name = net.hosts_links[host_name]["link"].intf2.name
    items = switch_port_name.split("-")
    return items[1].replace("eth","")



#action
def attach_link(net: Mininet, host_name:str) -> bool:
    """Enable link host - switch."""
    try:
        switch_name = get_switch_name_by_host_name(net, host_name)        
        information(f"--- 🛑 Attaching Link: {host_name} <--> {switch_name} ---")
        net.configLinkStatus(host_name, switch_name, 'down')
        debug("Link attached.")        
        return True
    except Exception as e:
        error(f"Error: attaching link: {e}")
        return False

def detach_link(net: Mininet, host_name:str) -> bool:
    """Disable link host - switch."""
    try:
        switch_name = get_switch_name_by_host_name(net, host_name)         
        information(f"--- 🛑 Detaching Link: {host_name} <--> {switch_name} ---")
        net.configLinkStatus(host_name, switch_name, 'down')
        debug("Link detached.")        
        return True
    except Exception as e:
        error(f"Error: detaching link: {e}")
        return False   


# --- Block Function ---

def block_flow_drop(net: Mininet, host_name: str) -> bool:
    """Adds a DROP rule to block IP traffic originating from the specified host."""
    try:
        # Retrieve necessary network components
        switch_name = get_switch_name_by_host_name(net, host_name) 
        sw_port_num = get_switch_port_by_host_name(net, host_name)  
        switch = net.get(switch_name)
        host = net.get(host_name)
        # Host IP is required for the OpenFlow match
        host_ip = host.IP()
        switch_name = switch.name
        
        # Rule: High priority (65535), matches incoming IP traffic on the host's port 
        # with the host's IP as source, and sets the action to drop.
        command_drop = f"ovs-ofctl add-flow {switch_name} priority=65535,in_port={sw_port_num},ip,nw_src={host_ip},actions=drop"
        
        information(f"--- 🚫 Blocking Flow from {host_name} ({host_ip}) on {switch_name} port {sw_port_num} ---")
        output = switch.cmd(command_drop)
        if output != "":
            error(f"Error adding DROP rule {host_name}: {output}")
            return False
        debug(f"DROP rule added successfully {host_name}. Flow blocked.")
        net.blocked_hosts.append(host)
        
        return True
    except Exception as e:
        error(f"Error adding DROP rule {host_name}: {e}")
        return False

# --- Unblock Function ---

def unblock_flow_delete(net: Mininet, host_name: str) -> bool:
    """Removes the previously added DROP rule to unblock the IP traffic from the host."""
    try:
        # Retrieve necessary network components
        switch_name = get_switch_name_by_host_name(net, host_name) 
        sw_port_num = get_switch_port_by_host_name(net, host_name)   
        switch = switch = net.get(switch_name)
        host = net.get(host_name)
        
        host_ip = host.IP()
        switch_name = switch.name
        
        # The match specification MUST exactly match the one used in 'add-flow', 
        # but without 'priority' or 'actions=drop'.
        rule_match = f"in_port={sw_port_num},ip,nw_src={host_ip}"
        
        # The del-flows command removes the flow matching the criteria
        command_delete = f"ovs-ofctl del-flows {switch_name} '{rule_match}'"
        
        information(f"--- 🔓 Unblocking Flow from {host_name} ({host_ip}) on {switch_name} ---")
        output = switch.cmd(command_delete)
        if output != "":
            error(f"Error removing DROP rule: {output}")
            return False
        debug(f"DROP rule removed successfully {host_name}. Flow unblocked.")
        net.blocked_hosts.remove(host)
        
        return True
    except Exception as e:
        print(f"Error removing DROP rule: {e}")
        return False
    
#communication actions. 
def comunicate_no_traffic_detected():
    return "NO traffic detected"

def comunicate_ping_traffic_detected():
    return "PING traffic detected"

def comunicate_tcp_traffic_detected():
    return "TCP traffic detected"

def comunicate_udp_traffic_detected():
    return "UDP traffic detected"

def comunicate_normal_traffic_detected():
    return "NORMAL traffic detected"

def comunicate_attack_detected():
    return "ATTACK detected"

def comunicate_in_attack_detected():
    return "ATTACK IN detected"

def comunicate_out_attack_detected():
    return "ATTACK OUT detected"
#communication functions end

def test_link_actions(config):
  
    #if you want to test the attack:
    net = create_network(config.env_params.net_params, server_user = config.server_user)  
    detach_link(net, "h1")
    time.sleep(1)
    attach_link(net, "h1")
    time.sleep(1)
    block_flow_drop(net, "h1")
    time.sleep(1)
    unblock_flow_delete(net, "h1")

if __name__ == '__main__':
    set_log_level('info')  # Set Mininet log level to display information
    net_params = { 
            'num_hosts':10,
            'num_switches':1,
            'num_iot':1,
            'controller': {
                'ip':'192.168.1.226',
                'port':6633,
                'usr':'admin',
                'pwd':'admin'
            }
        }
    net_params = jsonlib.loads(jsonlib.dumps(net_params), object_hook=Params)
    net = create_network(params = net_params)
    
    detach_link(net, "h1")
    time.sleep(1)
    attach_link(net, "h1")
    time.sleep(1)
    block_flow_drop(net, "h1")
    time.sleep(1)
    unblock_flow_delete(net, "h1")
    pass
    
    get_data_controller(net.controller, net.switches)
    
    for sw in net.switches:
        data = sw.data
        print(f"id {data.id}")
        for nc in data.node_connector:
            print(f"{nc.id}")
            print(f"rec {nc.statistics.packets.received}ptk - tra {nc.statistics.packets.transmitted}ptk")  
            print(f"rec {nc.statistics.bytes.received}byte - tra {nc.statistics.bytes.transmitted}byte")        
        
    host1 = net.hosts[0]
    host2 = net.hosts[1]
    sw1 = net.switches[0]
    switch_id=get_switch_id_by_name(sw1.name)

    while True:     
        time.sleep(3)
        sw1.data=get_data_switch_by_id(net.controller,switch_id)
        for table in sw1.data.get('tables'): 
            stats = table.get('opendaylight-flow-table-statistics:flow-table-statistics')
            if stats.get('active-flows') > 0: 
                print(f"table {table.get('id')} ")
                print(f"flows {stats.get('active-flows')} packets-matched {stats.get('packets-matched')} ")
