"""
Network Packet Simulator

A discrete-event simulator modeling packet transmission through a network with
routers, demonstrating queueing behavior, routing, and delay analysis.

This simulator models realistic network behavior including:
- Transmission delays (time to send packets)
- Propagation delays (time for signals to travel)
- Router processing delays
- Queueing at hosts and routers
- FIFO queue management

Author: Connor (pospsl)
"""

import csv
import heapq
from typing import List, Union, Dict


class Node:
    """
    Base class for network devices (hosts and routers).
    
    Attributes:
        node_id (str): Unique identifier for this node
        output_queue (List[Packet]): FIFO queue of packets waiting to be transmitted
        routing_table (Dict): Maps destination nodes to next-hop nodes
    """

    def __init__(self, node_id: str):
        self.node_id = node_id
        self.output_queue = []
        self.routing_table = {}

    def add_route(self, destination: 'Node', next_hop: 'Node'):
        """
        Add a routing entry.
        
        Args:
            destination: Final destination node
            next_hop: Next node in the path to destination
        """
        self.routing_table[destination] = next_hop


class Packet:
    """
    Represents a network packet.
    
    Each packet has a unique ID, source, destination, and next hop.
    Packets are created with auto-incrementing IDs for tracking.
    
    Attributes:
        packet_id (int): Unique packet identifier
        source (Node): Originating node
        destination (Node): Final destination node
        next_hop (Node): Immediate next destination
    """

    _counter: int = 0

    def __init__(self, source: Node, destination: Node, next_hop: Union[Node, None] = None):
        self.packet_id = Packet._counter
        Packet._counter += 1
        self.source = source
        self.destination = destination
        self.next_hop = next_hop if next_hop is not None else destination

    def __str__(self):
        return f'P[{self.packet_id},{self.source.node_id}->{self.destination.node_id}] next={self.next_hop.node_id}'


class Host(Node):
    """
    A host (end device) that sends and receives packets.
    
    Hosts have only an output queue for packets waiting to be transmitted.
    Examples: computers, servers, smartphones.
    """

    def __init__(self, node_id: str):
        super().__init__(node_id)

    def __str__(self):
        return f'{self.node_id:2s} queue={[p.packet_id for p in self.output_queue]}'


class Router(Node):
    """
    A router that forwards packets between network segments.
    
    Routers process incoming packets, consult routing tables, and forward
    packets toward their destinations. Processing takes time (processing_delay).
    
    Attributes:
        processing_delay (int): Time (ticks) required to process a packet
    """

    def __init__(self, node_id: str, processing_delay: int = 0):
        super().__init__(node_id)
        self.processing_delay = processing_delay

    def next_hop(self, destination: Node) -> Node:
        """
        Determine the next hop for a packet destined for the given node.
        
        Args:
            destination: Final destination node
            
        Returns:
            Next hop node from routing table
        """
        if self == destination:
            return self
        return self.routing_table[destination]

    def __str__(self):
        return f'{self.node_id:2s} out={[p.packet_id for p in self.output_queue]}'


class Event:
    """
    Represents a discrete event in the simulation.
    
    Events are processed in chronological order by the simulator.
    Each event has a type, target node, optional packet, and execution time.
    
    Event Types:
        ENQUEUE: Packet arrives in a node's queue
        TRANSMIT: Packet finishes transmission
        PROPAGATE: Packet travels through physical medium
        RECEIVE: Packet arrives at a node
        PROCESSING: Router processes packet to determine routing
    """

    ENQUEUE = 0
    TRANSMIT = 1
    PROPAGATE = 2
    RECEIVE = 3
    PROCESSING = 4
    
    _counter = 0

    def __init__(self, event_type: int, target_node: Node, packet: Packet = None, time: int = None):
        assert 0 <= event_type <= 4, "Invalid event type"
        self.target_node = target_node
        self.event_type = event_type
        self.time = time
        self.packet = packet
        self.event_id = Event._counter
        Event._counter += 1

    def type_to_str(self) -> str:
        """Convert event type to string representation."""
        event_names = {
            Event.ENQUEUE: 'ENQUEUE',
            Event.TRANSMIT: 'TRANSMIT',
            Event.PROPAGATE: 'PROPAGATE',
            Event.RECEIVE: 'RECEIVE',
            Event.PROCESSING: 'PROCESSING'
        }
        return event_names.get(self.event_type, 'UNKNOWN')

    def __str__(self):
        return f'{self.time:4d} {self.type_to_str():12s} {self.target_node.node_id} pkt={str(self.packet)}'


class Simulator:
    """
    Discrete-event network simulator.
    
    Simulates packet transmission through a network, tracking all delays
    (transmission, propagation, processing, queueing) and generating
    detailed performance metrics.
    
    Attributes:
        transmission_delay (int): Time to transmit a packet (ticks)
        propagation_delay (int): Time for signal to travel between nodes (ticks)
        clock (int): Current simulation time
        nodes (Dict[str, Node]): All nodes in the network
        event_queue (List): Priority queue of future events
        packet_log (Dict): Performance data for each packet
    """

    def __init__(self, transmission_delay: int = 10, propagation_delay: int = 1):
        self.event_queue: List = []
        self.transmission_delay = transmission_delay
        self.propagation_delay = propagation_delay
        self.clock = 0
        self.nodes: Dict[str, Node] = {}
        self.packet_log: Dict[int, Dict] = {}

    def schedule_event_after(self, event: Event, delay: int):
        """
        Schedule an event to occur after a specified delay.
        
        Args:
            event: Event to schedule
            delay: Delay in ticks from current time
        """
        event.time = self.clock + delay
        heapq.heappush(self.event_queue, (event.time, event.event_id, event))

    def run(self, verbose: bool = False):
        """
        Run the simulation until all events are processed.
        
        Args:
            verbose: If True, print each event as it's processed
        """
        if verbose:
            print('Starting simulation')
            
        while len(self.event_queue) > 0:
            self.clock, _, event = heapq.heappop(self.event_queue)
            
            if verbose:
                print(f'{str(event)}')
                
            self.handle_event(event)

    def handle_event(self, event: Event):
        """
        Process a single event according to its type.
        
        This method implements the core network behavior:
        - ENQUEUE: Add packet to queue, start transmission if idle
        - TRANSMIT: Complete transmission, propagate packet
        - PROPAGATE: Packet travels through medium
        - RECEIVE: Packet arrives, potentially trigger processing
        - PROCESSING: Router determines next hop
        
        Args:
            event: Event to process
        """
        node = event.target_node
        packet = event.packet

        # Initialize packet log entry if needed
        if packet is not None and packet.packet_id not in self.packet_log:
            self.packet_log[packet.packet_id] = {
                "source": packet.source.node_id,
                "queue_src": None,
                "transmit_src": None,
                "receive_c": None,
                "transmit_c": None,
                "receive_d": None,
            }

        if event.event_type == Event.ENQUEUE:
            self._handle_enqueue(node, packet)
        elif event.event_type == Event.TRANSMIT:
            self._handle_transmit(node, packet)
        elif event.event_type == Event.PROPAGATE:
            self._handle_propagate(node, packet)
        elif event.event_type == Event.PROCESSING:
            self._handle_processing(node, packet)
        elif event.event_type == Event.RECEIVE:
            self._handle_receive(node, packet)

    def _handle_enqueue(self, node: Node, packet: Packet):
        """Handle ENQUEUE event: packet arrives in node's output queue."""
        node_was_idle = (len(node.output_queue) == 0)
        node.output_queue.append(packet)

        # Log queue arrival time at source
        if node == packet.source:
            self.packet_log[packet.packet_id]["queue_src"] = self.clock

        # If node was idle, start transmitting immediately
        if node_was_idle:
            new_event = Event(Event.TRANSMIT, node, packet)
            self.schedule_event_after(new_event, self.transmission_delay)

    def _handle_transmit(self, node: Node, packet: Packet):
        """Handle TRANSMIT event: packet finishes transmission."""
        if node.output_queue:
            node.output_queue.pop(0)

        # Log transmission completion time
        if node == packet.source:
            self.packet_log[packet.packet_id]["transmit_src"] = self.clock
        elif isinstance(node, Router):
            self.packet_log[packet.packet_id]["transmit_c"] = self.clock

        # Determine next hop
        if isinstance(node, Host):
            next_node = packet.next_hop
        elif isinstance(node, Router):
            next_node = node.next_hop(packet.destination)
        else:
            next_node = packet.destination

        # Schedule propagation to next node
        new_event = Event(Event.PROPAGATE, next_node, packet)
        self.schedule_event_after(new_event, self.propagation_delay)

        # If more packets waiting, schedule next transmission
        if len(node.output_queue) > 0:
            next_packet = node.output_queue[0]
            next_event = Event(Event.TRANSMIT, node, next_packet)
            self.schedule_event_after(next_event, self.transmission_delay)

    def _handle_propagate(self, node: Node, packet: Packet):
        """Handle PROPAGATE event: packet travels through physical medium."""
        new_event = Event(Event.RECEIVE, node, packet)
        self.schedule_event_after(new_event, 0)

    def _handle_processing(self, node: Node, packet: Packet):
        """Handle PROCESSING event: router processes packet and determines routing."""
        if isinstance(node, Router):
            packet.next_hop = node.next_hop(packet.destination)

            node_was_idle = (len(node.output_queue) == 0)
            node.output_queue.append(packet)

            # If router was idle, start transmitting immediately
            if node_was_idle:
                new_event = Event(Event.TRANSMIT, node, packet)
                self.schedule_event_after(new_event, self.transmission_delay)

    def _handle_receive(self, node: Node, packet: Packet):
        """Handle RECEIVE event: packet arrives at a node."""
        if isinstance(node, Router):
            # Packet arrived at router
            self.packet_log[packet.packet_id]["receive_c"] = self.clock

            # Schedule processing
            new_event = Event(Event.PROCESSING, node, packet)
            self.schedule_event_after(new_event, node.processing_delay)

        elif node == packet.destination:
            # Packet arrived at final destination
            self.packet_log[packet.packet_id]["receive_d"] = self.clock

    def new_host(self, node_id: str) -> Host:
        """
        Create and register a new host.
        
        Args:
            node_id: Unique identifier for the host
            
        Returns:
            The created host
            
        Raises:
            Exception: If node_id already exists
        """
        if node_id in self.nodes:
            raise Exception(f'Node {node_id} already exists')
        node = Host(node_id)
        self.nodes[node_id] = node
        return node

    def new_router(self, node_id: str, processing_delay: int = 0) -> Router:
        """
        Create and register a new router.
        
        Args:
            node_id: Unique identifier for the router
            processing_delay: Time (ticks) required to process packets
            
        Returns:
            The created router
            
        Raises:
            Exception: If node_id already exists
        """
        if node_id in self.nodes:
            raise Exception(f'Node {node_id} already exists')
        router = Router(node_id, processing_delay=processing_delay)
        self.nodes[node_id] = router
        return router

    def export_results(self, filename: str):
        """
        Export simulation results to CSV file.
        
        Generates a CSV with detailed timing information for each packet,
        including queueing delays and end-to-end latency.
        
        Args:
            filename: Output CSV file path
            
        CSV Columns:
            - Seq num: Packet ID
            - Source: Source host
            - Queue @ source: Time packet entered source queue
            - Transmit @ source: Time transmission completed at source
            - Receive @ C: Time packet received at router
            - Transmit @ C: Time transmission completed at router
            - Receive @ D: Time packet received at destination
            - Queue Delay @ source: Time spent waiting in source queue
            - Queue Delay @ C: Time spent waiting in router queue
            - Total Queueing Delay: Sum of all queueing delays
            - End-to-End Delay: Total time from source to destination
        """
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                "Seq num.",
                "Source",
                "Queue @ source",
                "Transmit @ source",
                "Receive @ C",
                "Transmit @ C",
                "Receive @ D",
                "Queue Delay @ source",
                "Queue Delay @ C",
                "Total Queueing Delay",
                "End-to-End Delay"
            ])

            # Find router to get processing delay
            router = None
            for n in self.nodes.values():
                if isinstance(n, Router):
                    router = n
                    break
            proc_delay = router.processing_delay if router else 0

            # Write data for each packet
            for pkt_id, data in sorted(self.packet_log.items()):
                q_src = data.get("queue_src")
                t_src = data.get("transmit_src")
                r_c = data.get("receive_c")
                t_c = data.get("transmit_c")
                r_d = data.get("receive_d")

                # Calculate queueing delay at source
                if q_src is not None and t_src is not None:
                    start_t_src = t_src - self.transmission_delay
                    q_delay_src = max(0, start_t_src - q_src)
                else:
                    q_delay_src = ""

                # Calculate queueing delay at router
                if r_c is not None and t_c is not None:
                    start_t_c = t_c - self.transmission_delay
                    ready_c = r_c + proc_delay
                    q_delay_c = max(0, start_t_c - ready_c)
                else:
                    q_delay_c = ""

                # Total queueing delay
                if q_delay_src != "" and q_delay_c != "":
                    total_q = q_delay_src + q_delay_c
                else:
                    total_q = ""

                # End-to-end delay
                if q_src is not None and r_d is not None:
                    e2e_delay = r_d - q_src
                else:
                    e2e_delay = ""

                writer.writerow([
                    pkt_id,
                    data.get("source", ""),
                    q_src if q_src is not None else "",
                    t_src if t_src is not None else "",
                    r_c if r_c is not None else "",
                    t_c if t_c is not None else "",
                    r_d if r_d is not None else "",
                    q_delay_src,
                    q_delay_c,
                    total_q,
                    e2e_delay
                ])

    def visualize_burst(self, output_file: str = "burst_delay_analysis.png"):
        """
        Create visualization showing delay breakdown for a packet burst.
        
        Generates a stacked bar chart showing how end-to-end delay is composed
        of fixed components (transmission, propagation, processing) and variable
        queueing delays. Focuses on the first burst from Host A to clearly
        demonstrate queueing behavior.
        
        Args:
            output_file: Path for output image file
            
        Visualization shows:
            - Transmission delay (constant)
            - Propagation delay (constant)
            - Router processing delay (constant)
            - Queueing delay at source (increases within burst)
            - Queueing delay at router (variable based on congestion)
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            print("Warning: matplotlib not installed. Skipping visualization.")
            print("Install with: pip install matplotlib")
            return

        # Extract data for first burst (first 5 packets from Host A)
        burst_packets = []
        for pkt_id in range(5):
            if pkt_id in self.packet_log:
                data = self.packet_log[pkt_id]
                
                # Calculate queue delays
                q_src = data.get("queue_src")
                t_src = data.get("transmit_src")
                r_c = data.get("receive_c")
                t_c = data.get("transmit_c")
                
                if q_src is not None and t_src is not None:
                    start_t_src = t_src - self.transmission_delay
                    queue_src_delay = max(0, start_t_src - q_src)
                else:
                    queue_src_delay = 0
                
                if r_c is not None and t_c is not None:
                    # Get router processing delay
                    router = None
                    for n in self.nodes.values():
                        if isinstance(n, Router):
                            router = n
                            break
                    proc_delay = router.processing_delay if router else 0
                    
                    start_t_c = t_c - self.transmission_delay
                    ready_c = r_c + proc_delay
                    queue_router_delay = max(0, start_t_c - ready_c)
                else:
                    queue_router_delay = 0
                
                burst_packets.append({
                    'id': pkt_id,
                    'transmission': self.transmission_delay,
                    'propagation': self.propagation_delay,
                    'processing': proc_delay if 'proc_delay' in locals() else 1,
                    'queue_src': queue_src_delay,
                    'queue_router': queue_router_delay
                })

        if len(burst_packets) < 5:
            print("Warning: Not enough packets for burst visualization")
            return

        # Extract data for plotting
        packet_nums = [p['id'] for p in burst_packets]
        transmission = [p['transmission'] for p in burst_packets]
        propagation = [p['propagation'] for p in burst_packets]
        processing = [p['processing'] for p in burst_packets]
        queue_src = [p['queue_src'] for p in burst_packets]
        queue_router = [p['queue_router'] for p in burst_packets]

        # Create stacked bar chart
        fig, ax = plt.subplots(figsize=(10, 6))

        # Stack the bars
        ax.bar(packet_nums, transmission, label='Transmission Delay', 
               color='#3498db', edgecolor='black', linewidth=0.5)
        ax.bar(packet_nums, propagation, bottom=transmission, 
               label='Propagation Delay', color='#2ecc71', 
               edgecolor='black', linewidth=0.5)
        
        bottom_so_far = [t + p for t, p in zip(transmission, propagation)]
        ax.bar(packet_nums, processing, bottom=bottom_so_far, 
               label='Router Processing', color='#f39c12', 
               edgecolor='black', linewidth=0.5)
        
        bottom_so_far = [b + pr for b, pr in zip(bottom_so_far, processing)]
        ax.bar(packet_nums, queue_src, bottom=bottom_so_far, 
               label='Queue @ Source', color='#e74c3c', 
               edgecolor='black', linewidth=0.5)
        
        bottom_so_far = [b + q for b, q in zip(bottom_so_far, queue_src)]
        ax.bar(packet_nums, queue_router, bottom=bottom_so_far, 
               label='Queue @ Router', color='#c0392b', 
               edgecolor='black', linewidth=0.5)

        # Add total delay labels on top of each bar
        totals = [t + p + pr + qs + qr for t, p, pr, qs, qr in 
                  zip(transmission, propagation, processing, queue_src, queue_router)]
        for i, total in enumerate(totals):
            ax.text(i, total + 2, f'{total}', ha='center', 
                   fontweight='bold', fontsize=11)

        # Formatting
        ax.set_xlabel('Packet Number (within burst)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Delay (ticks)', fontweight='bold', fontsize=12)
        ax.set_title('End-to-End Delay Breakdown: Burst of 5 Packets from Host A', 
                    fontweight='bold', fontsize=14)
        ax.set_xticks(packet_nums)
        ax.set_xticklabels(['1st', '2nd', '3rd', '4th', '5th'])
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, max(totals) + 10)

        plt.tight_layout()
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")


def run_router_experiment(output_dir: str = "Outputs (Delay Info)", verbose: bool = False):
    """
    Run router queueing experiment.
    
    Network topology:
        Host A ─┐
                ├─→ Router C ─→ Host D
        Host B ─┘
    
    Workload:
        - Host A: Sends 5 packets every 1000 ticks (bursty traffic)
        - Host B: Sends 2 packets every 500 ticks (steady traffic)
        
    This experiment demonstrates:
        - Queueing behavior at both hosts and routers
        - Competition for router bandwidth
        - Impact of different traffic patterns on delays
        
    Args:
        output_dir: Directory for output files (CSV and visualization)
        verbose: If True, print detailed event log
    """
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    sim = Simulator(transmission_delay=10, propagation_delay=1)
    
    # Create network topology
    A = sim.new_host('A')
    B = sim.new_host('B')
    C = sim.new_router('C', processing_delay=1)
    D = sim.new_host('D')
    
    # Configure routing
    C.add_route(D, D)

    # Schedule workload: Host A sends 5 packets every 1000 ticks
    for t in range(0, 10000, 1000):
        for _ in range(5):
            pkt = Packet(A, D, next_hop=C)
            sim.schedule_event_after(Event(Event.ENQUEUE, A, pkt), t)

    # Schedule workload: Host B sends 2 packets every 500 ticks
    for t in range(0, 10000, 500):
        for _ in range(2):
            pkt = Packet(B, D, next_hop=C)
            sim.schedule_event_after(Event(Event.ENQUEUE, B, pkt), t)

    # Run simulation
    sim.run(verbose=verbose)
    
    # Define output file paths
    csv_path = os.path.join(output_dir, "router_queueing.csv")
    viz_path = os.path.join(output_dir, "burst_delay_analysis.png")
    
    # Export results
    sim.export_results(csv_path)
    print(f"Simulation complete. Results exported to {csv_path}")
    
    # Generate visualization
    sim.visualize_burst(viz_path)


if __name__ == '__main__':
    run_router_experiment()