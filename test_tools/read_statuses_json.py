import json

def read_statuses_json(file_path):
    """
    Reads a JSON file containing statuses and returns the data as a Python object.

    Args:
        file_path (str): The path to the JSON file.
        
    Returns:
        dict: The data from the JSON file.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def count_bytes_over(statuses, threshold):
    """
    Counts the number of bytes in the statuses that have a value over a given threshold.
    """
    id = 0
    count_bt_total = 0
    count_bt_hosts_received = 0
    count_bt_hosts_sent = 0
    count_bt_received_different_from_sent = 0
    count_bt_host_data_different_from_all = 0
    
    for status in statuses:
        id += 1
        bytes_ = status.get('bytes', 0)
        if bytes_ > threshold:
            count_bt_total += 1
        hostStatusesStructured = status.get('hostStatusesStructured', [])
        receivedBytes = 0
        transmittedBytes = 0
        for host_name, host_status in hostStatusesStructured.items():
            receivedBytesPerHost = host_status.get('receivedBytes', 0)
            receivedBytes += receivedBytesPerHost
            if receivedBytesPerHost > threshold:
                count_bt_hosts_received += 1
                print(f"Id {id} Host {host_name} received bytes over threshold: {receivedBytesPerHost}")    
            transmittedBytesPerHost = host_status.get('transmittedBytes', 0)
            transmittedBytes += transmittedBytesPerHost
            if transmittedBytesPerHost > threshold:
                count_bt_hosts_sent += 1
                print(f"Id {id} Host {host_name} transmitted bytes over threshold: {transmittedBytesPerHost}")
        if (receivedBytes - transmittedBytes)**2 > 10000000000:
            count_bt_received_different_from_sent += 1
        if (receivedBytes - bytes_)**2 > 10000000000 or (transmittedBytes - bytes_)**2 > 10000000000:
            count_bt_host_data_different_from_all   += 1
    return count_bt_total, count_bt_hosts_received, count_bt_hosts_sent, count_bt_received_different_from_sent, count_bt_host_data_different_from_all


def count_packets_over(statuses, threshold):
    """
    Counts the number of packets in the statuses that have a value over a given threshold.
    """
    id = 0
    count_pkt_total = 0
    count_pkt_hosts_received = 0
    count_pkt_hosts_sent = 0
    count_pkt_received_different_from_sent = 0
    count_pkt_host_data_different_from_all = 0
    
    for status in statuses:
        id += 1
        packets = status.get('packets', 0)
        if packets > threshold:
            count_pkt_total += 1
        hostStatusesStructured = status.get('hostStatusesStructured', [])
        receivedPackets = 0
        transmittedPackets = 0
        for host_name, host_status in hostStatusesStructured.items():
            receivedPacketsPerHost = host_status.get('receivedPackets', 0)
            receivedPackets += receivedPacketsPerHost
            if receivedPacketsPerHost > threshold:
                count_pkt_hosts_received += 1
                print(f"Id {id} Host {host_name} received packets over threshold: {receivedPacketsPerHost}")    
            transmittedPacketsPerHost = host_status.get('transmittedPackets', 0)
            transmittedPackets += transmittedPacketsPerHost
            if transmittedPacketsPerHost > threshold:
                count_pkt_hosts_sent += 1
                print(f"Id {id} Host {host_name} transmitted packets over threshold: {transmittedPacketsPerHost}")
        if (receivedPackets - transmittedPackets)**2 > 10000:
            count_pkt_received_different_from_sent += 1
        if (receivedPackets - packets)**2 > 10000 or (transmittedPackets - packets)**2 > 10000:
            count_pkt_host_data_different_from_all   += 1
    return count_pkt_total, count_pkt_hosts_received, count_pkt_hosts_sent, count_pkt_received_different_from_sent, count_pkt_host_data_different_from_all

#main function to test the read_statuses_json function
if __name__ == "__main__":
    # Example usage
    file_path = '_training/attacks_ho/20260725-172548_1_5_5/statuses.json'  # Replace with your JSON file path
    statuses = read_statuses_json(file_path)
    
    is_pkt = True
    
    if is_pkt:
        pkt_threshold = 200000 # Replace with your desired threshold
        packet_count, pkt_hosts_received_count, pkt_hosts_sent_count, pkt_received_different_from_sent_count, host_data_different_pkt_from_all_count = count_packets_over(statuses, pkt_threshold)
        print(f"Number of packets over {pkt_threshold}: {packet_count}")
        print(f"Number of hosts that received packets over {pkt_threshold}: {pkt_hosts_received_count}")
        print(f"Number of hosts that transmitted packets over {pkt_threshold}: {pkt_hosts_sent_count}")
        print(f"Number of hosts with received packets different from transmitted packets: {pkt_received_different_from_sent_count}")
        print(f"Number of hosts with host data different from all packets: {host_data_different_pkt_from_all_count}")
    else:
        bt_threshold = 2000000000 # Replace with your desired threshold
        byte_count, bt_hosts_received_count, bt_hosts_sent_count, bt_received_different_from_sent_count, host_data_different_bt_from_all_count = count_bytes_over(statuses, bt_threshold)
        print(f"Number of bytes over {bt_threshold}: {byte_count}")
        print(f"Number of hosts that received bytes over {bt_threshold}: {bt_hosts_received_count}")
        print(f"Number of hosts that transmitted bytes over {bt_threshold}: {bt_hosts_sent_count}")
        print(f"Number of hosts with received bytes different from transmitted bytes: {bt_received_different_from_sent_count}")
        print(f"Number of hosts with host data different from all bytes: {host_data_different_bt_from_all_count}")    