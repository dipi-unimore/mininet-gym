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

def count_packets_over(statuses, threshold):
    """
    Counts the number of packets in the statuses that have a value over a given threshold.
    """
    id = 0
    count_total = 0
    count_hosts_received = 0
    count_hosts_sent = 0
    count_received_different_from_sent = 0
    count_host_data_different_from_all = 0
    for status in statuses:
        id += 1
        packets = status.get('packets', 0)
        if packets > threshold:
            count_total += 1
        hostStatusesStructured = status.get('hostStatusesStructured', [])
        receivedPackets = 0
        transmittedPackets = 0
        for host_name, host_status in hostStatusesStructured.items():
            receivedPacketsPerHost = host_status.get('receivedPackets', 0)
            receivedPackets += receivedPacketsPerHost
            if receivedPacketsPerHost > threshold:
                count_hosts_received += 1
                print(f"Id {id} Host {host_name} received packets over threshold: {receivedPacketsPerHost}")    
            transmittedPacketsPerHost = host_status.get('transmittedPackets', 0)
            transmittedPackets += transmittedPacketsPerHost
            if transmittedPacketsPerHost > threshold:
                count_hosts_sent += 1
                print(f"Id {id} Host {host_name} transmitted packets over threshold: {transmittedPacketsPerHost}")
        if (receivedPackets - transmittedPackets)**2 > 10000:
            count_received_different_from_sent += 1
        if (receivedPackets - packets)**2 > 10000 or (transmittedPackets - packets)**2 > 10000:
            count_host_data_different_from_all   += 1
    return count_total, count_hosts_received, count_hosts_sent, count_received_different_from_sent, count_host_data_different_from_all

#main function to test the read_statuses_json function
if __name__ == "__main__":
    # Example usage
    file_path = '_training/attacks_ho/20260723-185141_1_5_5/statuses.json'  # Replace with your JSON file path
    statuses = read_statuses_json(file_path)
    #print(statuses)
    threshold = 200000 # Replace with your desired threshold
    packet_count, hosts_received_count, hosts_sent_count, received_different_from_sent_count, host_data_different_from_all_count = count_packets_over(statuses, threshold)
    print(f"Number of packets over {threshold}: {packet_count}")
    print(f"Number of hosts that received packets over {threshold}: {hosts_received_count}")
    print(f"Number of hosts that transmitted packets over {threshold}: {hosts_sent_count}")
    print(f"Number of hosts with received packets different from transmitted packets: {received_different_from_sent_count}")
    print(f"Number of hosts with host data different from all packets: {host_data_different_from_all_count}")