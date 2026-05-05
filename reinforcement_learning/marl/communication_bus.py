import collections

class CommunicationBus:
    """
    Manages the message queues for all agents, handling direct messages and broadcasts.
    """
    def __init__(self):
        # Use a defaultdict to automatically create a list for new agents
        self.messages = collections.defaultdict(list)

    def send_message(self, sender_id: str, receiver_id: str, message: str):
        """
        Sends a message from one agent to a specific receiver.

        Args:
            sender_id (str): The ID of the agent sending the message.
            receiver_id (str): The ID of the agent receiving the message.
            message (str): The content of the message.
        """
        self.messages[receiver_id].append({"sender": sender_id, "content": message})

    def broadcast(self, sender_id: str, message: str):
        """
        Broadcasts a message from the sender to all other registered agents.

        Args:
            sender_id (str): The ID of the agent sending the message.
            message (str): The content of the message.
        """
        # A simple broadcast logic; you might need to manage a list of agent IDs
        # For now, it sends to all keys in the messages dictionary
        for receiver_id in self.messages.keys():
            if receiver_id != sender_id:
                self.send_message(sender_id, receiver_id, message)
    
    def get_messages(self, agent_id: str) -> list:
        """
        Retrieves all messages for a specific agent and clears the queue.

        Args:
            agent_id (str): The ID of the agent.

        Returns:
            list: A list of messages for the agent.
        """
        received_messages = self.messages[agent_id]
        self.messages[agent_id] = []  # Clear the queue after retrieval
        return received_messages
        
    def clear_messages(self):
        """
        Clears all message queues, typically at the start of a new episode.
        """
        self.messages.clear()