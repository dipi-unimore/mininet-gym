import re
import time
import traceback
from typing import Any, Dict, List, Optional
from mininet.node import Host, Switch
from mininet.net import Mininet
from utility.my_log import information, debug, error

# Global flag to track if monitoring flows have been installed
_monitoring_flows_installed = {}

# --- Utility for installing Monitoring Flows ---

def install_monitoring_flows(ovs_switch: Switch, net: Mininet, force: bool = False):
    """
    Installs OpenFlow monitoring flows (priority=2) on the OVS bridge.
    Should be called ONCE during network setup, not repeatedly.
    
    Args:
        ovs_switch: The OVS switch object
        net: The Mininet network object
        force: If True, reinstall flows even if already installed
    """
    bridge_name = ovs_switch.name
    
    # Check if flows already installed for this bridge
    if not force and bridge_name in _monitoring_flows_installed:
        debug(f"Monitoring flows already installed for {bridge_name}, skipping...")
        return True
    
    if ovs_switch is None or net.hosts is None:
        error("Cannot install monitoring flows: switch or hosts not found.")
        return False

    MONITORING_PRIORITY = 2
    blocked_hosts = getattr(net, 'blocked_hosts', [])
    
    success_count = 0
    total_hosts = 0
    
    for host in net.hosts:
        if not isinstance(host, Host):
            continue
            
        total_hosts += 1
        
        try:
            # Wait a moment for host to be fully initialized
            if not hasattr(host, 'MAC'):
                error(f"Host {getattr(host, 'name', 'Unknown')} not fully initialized (no MAC method)")
                continue
                
            host_mac = host.MAC()
            
            # Validate MAC address
            if not host_mac or host_mac == "00:00:00:00:00:00":
                error(f"Invalid MAC address for host {host.name}: {host_mac}")
                continue
            
            # Flow 1: Traffic OUT from the host (match on dl_src)
            command_out = (
                f'ovs-ofctl add-flow {bridge_name} '
                f'priority={MONITORING_PRIORITY},dl_src={host_mac},actions=NORMAL'
            )
            
            # Flow 2: Traffic IN to the host (match on dl_dst)
            command_in = (
                f'ovs-ofctl add-flow {bridge_name} '
                f'priority={MONITORING_PRIORITY},dl_dst={host_mac},actions=NORMAL'
            )

            if host not in blocked_hosts:
                result_out = ovs_switch.cmd(command_out)
                if result_out and ("error" in result_out.lower() or "failed" in result_out.lower()):
                    error(f"Failed to install OUT flow for {host.name}: {result_out.strip()}")
                    continue
                    
            result_in = ovs_switch.cmd(command_in)
            if result_in and ("error" in result_in.lower() or "failed" in result_in.lower()):
                error(f"Failed to install IN flow for {host.name}: {result_in.strip()}")
                continue
            
            debug(f"Installed P2 flows for {host.name} ({host_mac})")
            success_count += 1
                    
        except AttributeError as e:
            error(f"Failed to install monitoring flows for {getattr(host, 'name', 'Unknown')}. "
                  f"AttributeError: {type(e).__name__} - {str(e) or 'Object not fully initialized'}")
            debug(f"Traceback: {traceback.format_exc()}")
        except Exception as e:
            error(f"Failed to install monitoring flows for {getattr(host, 'name', 'Unknown')}. "
                  f"Error type: {type(e).__name__}, Message: {str(e) or 'Empty exception message'}")
            debug(f"Traceback: {traceback.format_exc()}")
    
    if success_count == total_hosts:
        information(f"Successfully installed monitoring flows for all {total_hosts} hosts on {bridge_name}")
        _monitoring_flows_installed[bridge_name] = True
        return True
    elif success_count > 0:
        information(f"Partially installed monitoring flows: {success_count}/{total_hosts} hosts on {bridge_name}")
        _monitoring_flows_installed[bridge_name] = True
        return True
    else:
        error(f"Failed to install monitoring flows for any host on {bridge_name}")
        return False

def wait_for_network_ready(net: Mininet, bridge_name: str, timeout: float = 10.0) -> bool:
    """
    Wait for the network to be fully initialized before attempting operations.
    
    Args:
        net: Mininet network object
        bridge_name: Name of the OVS bridge
        timeout: Maximum time to wait in seconds
        
    Returns:
        True if network is ready, False if timeout
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check if switch exists
            ovs_switch = next((s for s in net.switches if s.name == bridge_name and isinstance(s, Switch)), None)
            if not ovs_switch:
                debug(f"Switch {bridge_name} not found, waiting...")
                time.sleep(0.5)
                continue
            
            # Check if hosts are initialized with MACs
            all_hosts_ready = True
            for host in net.hosts:
                if isinstance(host, Host):
                    try:
                        mac = host.MAC()
                        if not mac or mac == "00:00:00:00:00:00":
                            all_hosts_ready = False
                            break
                    except:
                        all_hosts_ready = False
                        break
            
            if not all_hosts_ready:
                debug(f"Hosts not fully initialized, waiting...")
                time.sleep(0.5)
                continue
            
            # Try a simple command to verify OVS is responsive
            result = ovs_switch.cmd(f'ovs-ofctl show {bridge_name}')
            if result and "error" not in result.lower():
                information(f"Network ready after {time.time() - start_time:.2f}s")
                return True
                
        except Exception as e:
            debug(f"Network not ready: {type(e).__name__}")
        
        time.sleep(0.5)
    
    error(f"Network not ready after {timeout}s timeout")
    return False

# --- Parsing Functions ---

def _get_mac_to_name_map(net: Mininet) -> Dict[str, str]:
    """Creates a MAC -> Host Name map for host resolution."""
    mac_to_name: Dict[str, str] = {}
    if net and net.hosts:
        for host in net.hosts:
            if isinstance(host, Host):
                try:
                    mac = host.MAC()
                    if mac:
                        mac_to_name[mac.lower()] = host.name
                except Exception as e:
                    error(f"Failed to get MAC for host {getattr(host, 'name', 'Unknown')}: {type(e).__name__}")
    return mac_to_name

def _get_port_to_host_map(net: Mininet, bridge_name: str) -> Dict[str, str]:
    """Creates a map from OVS port name to Host name."""
    port_to_host: Dict[str, str] = {}
    if net and net.switches:
        ovs_switch = next((s for s in net.switches if s.name == bridge_name and isinstance(s, Switch)), None)
        if ovs_switch:
            for intf in ovs_switch.intfList():
                if intf.name.startswith(bridge_name + '-eth'):
                    try:
                        port_number = intf.name.split('eth')[-1]
                        host_index = int(port_number) - 1
                        
                        if 0 <= host_index < len(net.hosts):
                            host_name = net.hosts[host_index].name
                            port_to_host[intf.name] = host_name
                        else:
                            debug(f"Port {intf.name} index {host_index} out of range for {len(net.hosts)} hosts")
                    except (ValueError, IndexError) as e:
                        debug(f"Failed to map port {intf.name}: {type(e).__name__}")
                        continue
    return port_to_host

def parse_flow_ovs_ofctl(flow_data: str, net: Mininet) -> Dict[str, Any]:
    """Parses ovs-ofctl dump-flows output."""
    mac_to_name = _get_mac_to_name_map(net)
    
    FLOW_PATTERN = re.compile(
        r'.*?n_packets=(?P<packets>\d+).*?'
        r'n_bytes=(?P<bytes>\d+).*?'
        r'priority=(?P<priority>\d+).*?'
        r'(?:dl_src=(?P<src_mac>[\w:]+).*?)?' 
        r'(?:dl_dst=(?P<dst_mac>[\w:]+).*?)?'
        r'.*?'
    )
    
    flows: List[Dict[str, Any]] = []
    lines = flow_data.strip().replace('\r', '').split('\n')
    
    total_received_packets = 0
    total_transmitted_packets = 0
    total_received_bytes = 0
    total_transmitted_bytes = 0
    
    for line in lines:
        if 'priority=2' not in line: 
            continue 
        
        match = FLOW_PATTERN.search(line)
        if match:
            data = match.groupdict()
            
            packets = int(data['packets'])
            bytes_ = int(data['bytes'])
            src_mac = data.get('src_mac')
            dst_mac = data.get('dst_mac')
            
            is_dl_src = src_mac is not None
            is_dl_dst = dst_mac is not None
            
            src_mac_final = src_mac or 'N/A'
            dst_mac_final = dst_mac or 'N/A'
            
            if is_dl_src:
                total_transmitted_packets += packets
                total_transmitted_bytes += bytes_
                src_name = mac_to_name.get(src_mac_final.lower(), 'Unknown')
                dst_name = 'N/A'
            elif is_dl_dst:
                total_received_packets += packets
                total_received_bytes += bytes_
                dst_name = mac_to_name.get(dst_mac_final.lower(), 'Unknown')
                src_name = 'N/A'
            else:
                continue

            flows.append({
                "src_name": src_name,
                "dst_name": dst_name,
                "packets": packets,
                "bytes": bytes_,
                "direction": "TX" if is_dl_src else "RX"
            })

    return {
        "packets": {"received": total_received_packets, "transmitted": total_transmitted_packets},
        "bytes": {"received": total_received_bytes, "transmitted": total_transmitted_bytes},
        "flows": flows,
        "description": "OpenFlow Table Flows (Priority 2)"
    }

def parse_flow_ovs_dpctl(flow_data: str, net: Mininet) -> Dict[str, Any]:
    """Parses ovs-dpctl dump-flows output."""
    mac_to_name = _get_mac_to_name_map(net)
    
    FLOW_PATTERN = re.compile(
        r'eth\(src=(?P<src_mac>[\w:]+),dst=(?P<dst_mac>[\w:]+)\).*?'
        r'packets:(?P<packets>\d+).*?'
        r'bytes:(?P<bytes>\d+).*?'
        r'used:(?P<used>[\w\d\.]+).*?' 
    )
    
    flows: List[Dict[str, Any]] = []
    lines = flow_data.strip().replace('\r', '').split('\n')
    
    total_packets = 0
    total_bytes = 0
    
    for line in lines:
        if 'eth(src=' not in line: 
            continue
        
        match = FLOW_PATTERN.search(line)
        if match:
            data = match.groupdict()
            
            try:
                packets = int(data['packets'])
                bytes_ = int(data['bytes'])
                src_mac_final = data['src_mac'].lower()
                dst_mac_final = data['dst_mac'].lower()
                
                src_name = mac_to_name.get(src_mac_final, 'Unknown')
                dst_name = mac_to_name.get(dst_mac_final, 'Unknown')

                total_packets += packets
                total_bytes += bytes_

                flows.append({
                    "src_name": src_name,
                    "dst_name": dst_name,
                    "packets": packets,
                    "bytes": bytes_,
                })
            except (ValueError, KeyError):
                continue

    return {
        "packets": {"received": total_packets, "transmitted": total_packets},
        "bytes": {"received": total_bytes, "transmitted": total_bytes},
        "flows": flows,
        "description": "Kernel Datapath Flows"
    }

def parse_flow_ovs_ports(flow_data: str, net: Mininet, bridge_name: str) -> Dict[str, Any]:
    """Parses ovs-ofctl dump-ports output."""
    if not hasattr(net, 'port_to_host_map'):
        net.port_to_host_map = _get_port_to_host_map(net, bridge_name)
    
    PORT_RX_PATTERN = re.compile(r'port\s+"?(?P<port_name>[\w\-]+)"?:\s+rx pkts=(?P<rx_pkts>\d+),\s+bytes=(?P<rx_bytes>\d+).*?')
    PORT_TX_PATTERN = re.compile(r'\s+tx pkts=(?P<tx_pkts>\d+),\s+bytes=(?P<tx_bytes>\d+).*?')
    
    separated_flows: List[Dict[str, Any]] = []
    
    total_rx_packets = 0
    total_tx_packets = 0
    total_rx_bytes = 0
    total_tx_bytes = 0
    
    lines = flow_data.strip().replace('\r', '').split('\n')
    current_port_data: Dict[str, Any] = {}

    for line in lines:
        rx_match = PORT_RX_PATTERN.search(line)
        tx_match = PORT_TX_PATTERN.search(line)

        if rx_match:
            if current_port_data:
                debug(f"Skipping incomplete port data for {current_port_data.get('port_name', 'Unknown')}")
                current_port_data = {}

            data = rx_match.groupdict()
            port_name = data['port_name']
            
            if port_name == 'local':
                current_port_data = {}
                continue
                 
            rx_pkts = int(data['rx_pkts'])
            rx_bytes = int(data['rx_bytes'])

            host_name = net.port_to_host_map.get(port_name, 'Unknown')
            
            current_port_data = {
                "port_name": port_name,
                "host_name": host_name,
                "rx_pkts": rx_pkts,
                "rx_bytes": rx_bytes,
                "tx_pkts": 0,
                "tx_bytes": 0,
            }
            total_rx_packets += rx_pkts
            total_rx_bytes += rx_bytes

        elif tx_match and current_port_data:
            data = tx_match.groupdict()
            tx_pkts = int(data['tx_pkts'])
            tx_bytes = int(data['tx_bytes'])
            
            current_port_data["tx_pkts"] = tx_pkts
            current_port_data["tx_bytes"] = tx_bytes
            
            total_tx_packets += tx_pkts
            total_tx_bytes += tx_bytes
            
            tx_flow = {
                "src_name": current_port_data["host_name"],
                "dst_name": "N/A",
                "packets": current_port_data["tx_pkts"], 
                "bytes": current_port_data["tx_bytes"],     
                "port_name": current_port_data["port_name"],
                "direction": "TX"
            }
            
            rx_flow = {
                "src_name": "N/A",
                "dst_name": current_port_data["host_name"], 
                "packets": current_port_data["rx_pkts"],
                "bytes": current_port_data["rx_bytes"],
                "port_name": current_port_data["port_name"],
                "direction": "RX"
            }

            separated_flows.append(tx_flow)
            separated_flows.append(rx_flow)
            
            current_port_data = {}
            
    return {
        "packets": {"received": total_rx_packets, "transmitted": total_tx_packets},
        "bytes": {"received": total_rx_bytes, "transmitted": total_tx_bytes},
        "flows": separated_flows,
        "description": "Port Statistics"
    }

# --- Main Data Collection Function ---

def initialize_monitoring(net: Mininet, bridge_name: str = "s1", wait_timeout: float = 10.0) -> bool:
    """
    Initialize network monitoring by waiting for network readiness and installing flows.
    Call this ONCE after starting the Mininet network.
    
    Args:
        net: Mininet network object
        bridge_name: Name of the OVS bridge
        wait_timeout: Maximum time to wait for network to be ready
        
    Returns:
        True if initialization successful, False otherwise
    """
    information(f"Initializing network monitoring for bridge {bridge_name}...")
    
    # Wait for network to be ready
    if not wait_for_network_ready(net, bridge_name, timeout=wait_timeout):
        error("Network initialization timeout - cannot install monitoring flows")
        return False
    
    # Find the switch
    ovs_switch = next((s for s in net.switches if s.name == bridge_name and isinstance(s, Switch)), None)
    if ovs_switch is None:
        error(f"Could not find OVS switch named '{bridge_name}'")
        return False
    
    # Install monitoring flows once
    return install_monitoring_flows(ovs_switch, net, force=False)

def get_data_flow(net: Mininet, bridge_name: str = "s1", max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Collects flow statistics from the OVS bridge.
    Does NOT install flows - call initialize_monitoring() first!
    
    Args:
        net: Mininet network object
        bridge_name: Name of the OVS bridge
        max_retries: Maximum number of retry attempts
        
    Returns:
        Dictionary with flow statistics, or None if failed
    """
    # Check if monitoring flows have been installed
    if bridge_name not in _monitoring_flows_installed:
        error(f"Monitoring flows not installed for {bridge_name}. Call initialize_monitoring() first!")
        return None
    
    ovs_switch = None
    if net and hasattr(net, 'switches'):
        ovs_switch = next((s for s in net.switches if s.name == bridge_name and isinstance(s, Switch)), None)
    
    if ovs_switch is None:
        error(f"Could not find an OVS switch named '{bridge_name}'")
        return None
    
    for attempt in range(max_retries):
        start_time = time.time()
        
        try:
            command_ports = f'ovs-ofctl dump-ports {bridge_name}'
            
            time.sleep(0.1)  # Small delay to avoid overwhelming OVS
            flows_raw = ovs_switch.cmd(command_ports)
            
            # Validate output
            if not flows_raw or flows_raw.strip() == "":
                error(f"Command returned empty output (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
            
            if "No such device" in flows_raw or "error" in flows_raw.lower() or "failed" in flows_raw.lower():
                error(f"Command failed: {flows_raw.strip()} (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
            
            # Parse the output
            parsed_data = parse_flow_ovs_ports(flows_raw, net, bridge_name)
            
            if 'error' in parsed_data:
                error(f"Parsing failed (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
                return None
            
            end_time = time.time()
            debug(f"Successfully retrieved flow data in {end_time - start_time:.4f}s")
            return parsed_data
            
        except Exception as e:
            exc_type = type(e).__name__
            exc_msg = str(e) if str(e) else "Empty exception message"
            error(f"Error retrieving flow data (attempt {attempt + 1}/{max_retries}): "
                  f"{exc_type}: {exc_msg}")
            debug(f"Traceback: {traceback.format_exc()}")
            
            if attempt < max_retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
    
    error(f"Failed to retrieve flow data after {max_retries} attempts")
    return None