from mininet.net import Mininet
from mininet.node import RemoteController, OVSKernelSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from utility.my_log import set_log_level, information, debug, error, notify_client
from utility.my_files import drop_privileges, regain_root
from utility.params import Params
from mininet.clean import cleanup
from threading import Thread
import numpy as np
import subprocess
import os, re
import time
import requests
#from lxml import etree
import json as jsonlib
from types import SimpleNamespace

#https://brianlinkletter.com/2015/04/how-to-use-miniedit-mininets-graphical-user-interface/

def add_default_flows(switch):
    #os.system(f"ovs-ofctl add-flow {switch} priority=100,in_port=1,actions=output:2")
    #os.system(f"ovs-ofctl add-flow {switch} priority=100,in_port=2,actions=output:1")
    os.system(f"ovs-ofctl add-flow {switch} priority=1,actions=normal")
    
def start_cli(net):
    CLI(net)  # Run Mininet CLI

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
        switch = net.addSwitch(f's{i+1}', protocols='OpenFlow13')
        switches.append(switch)

    information(f'*** Adding {params.num_hosts} hosts\n')
    hosts = [net.addHost(f'h{i+1}', ip=f'10.0.0.{i+1}/24') for i in range(params.num_hosts)]  # Same subnet for all hosts

    information(f'*** Adding IoT {params.num_iot} devices\n')
    iot_devices = [net.addHost(f'iot{i+1}', ip=f'10.0.0.{params.num_hosts + i + 1}/24') for i in range(params.num_iot)]  # Same subnet

    information(f'*** Creating links on {params.num_switches} switches \n')
    for switch in switches:
        for host in hosts:
            net.addLink(host, switch)  # Connect each host to a switch
        for iot_device in iot_devices:
            net.addLink(iot_device, switch)  # Connect each IoT device to a switch

    information('*** Starting network\n')
    net.start()

    information('*** Configuring OpenFlow 1.0 and 1.3 protocol for each switch\n')
    for switch in switches:
        switch.cmd('ovs-vsctl set bridge %s protocols=OpenFlow10,OpenFlow13' % switch.name)
        add_default_flows(switch)

    # info('*** Running ping test\n')
    # net.pingAll()  # Test network connectivity

    # information('*** Starting CLI in a separate thread\n')
    # cli_thread = Thread(target=start_cli, args=(net,))
    # cli_thread.start()  # Start the CLI in a new thread, non-blocking

    net.switches = switches
    #net.controller = controller
    net.host_devices = hosts
    net.iot_devices = iot_devices
    net.total_packets_received = 0
    net.total_bytes_received = 0
    net.total_packets_transmitted = 0
    net.total_bytes_transmitted = 0
    net.traffic_types = params.traffic_types
    drop_privileges(server_user)
    return net

def stop(net: Mininet):
    regain_root()
    net.stop()

def get_host_by_name(net,name):
    for host in net.hosts:
        if host.name == name:
            return host
    return None

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
import re

import re

def parse_ovs_flows(flow_data, net=None):
    """
    Parse the output of 'ovs-dpctl dump-flows' to extract packets, bytes, and flow details.
    """
    flows = []
    total_packets = 0
    total_bytes = 0

    for line in flow_data.splitlines():
        try:
            in_port_match = re.search(r'in_port\((\d+)\)', line)
            src_match = re.search(r'src=([\w:]+)', line)
            dst_match = re.search(r'dst=([\w:]+)', line)
            packets_match = re.search(r'packets:(\d+)', line)
            bytes_match = re.search(r'bytes:(\d+)', line)
            used_match = re.search(r'used:([\w\d\.]+)', line)
            actions_match = re.search(r'actions:(.*)', line)

            if not (in_port_match and src_match and dst_match and packets_match and bytes_match and actions_match):
                continue  # skip if any critical info is missing

            in_port = int(in_port_match.group(1))
            src_mac = src_match.group(1)
            dst_mac = dst_match.group(1)
            packets = int(packets_match.group(1))
            bytes_ = int(bytes_match.group(1))
            used = used_match.group(1) if used_match else "N/A"
            actions = actions_match.group(1).strip()

            total_packets += packets
            total_bytes += bytes_

            src_name = dst_name = None
            if net:
                src_host = next((host for host in net.hosts if host.MAC() == src_mac), None)
                dst_host = next((host for host in net.hosts if host.MAC() == dst_mac), None)
                src_name = src_host.name if src_host else None
                dst_name = dst_host.name if dst_host else None

            flows.append({
                "in_port": in_port,
                "src": src_mac,
                "dst": dst_mac,
                "src_name": src_name,
                "dst_name": dst_name,
                "packets": packets,
                "bytes": bytes_,
                "used": used,
                "actions": actions,
            })

        except Exception as e:
            print(f"Warning: Failed to parse flow line: {line}\nError: {e}")
            continue

    return {
        "packets": {
            "received": total_packets,
            "transmitted": total_packets,  # assumed symmetry
        },
        "bytes": {
            "received": total_bytes,
            "transmitted": total_bytes,
        },
        "flows": flows,
    }


    
def get_data_controller_flow(net):
    # Get the raw output from ovs-dpctl dump-flows
    f = os.popen('ovs-dpctl dump-flows ovs-system')
    flows_raw = f.read()
    f.close()
    # Parse the flows
    parsed_flows = parse_ovs_flows(flows_raw, net)
    return parsed_flows     
        
def get_data_controller(controller,switches):
    response = get_response(controller,'')
    if response.status_code == 200:
        #info(response.status_code)
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
    time.sleep(1)
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
