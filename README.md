# Network Packet Simulator

A discrete-event network simulator demonstrating packet transmission, router queueing, and delay analysis. Built to explore and help myself understand fundamental networking concepts, including congestion, queueing theory, and end-to-end latency in computer networks.

![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

---

## Overview

This simulator models how packets travel through a network, accounting for realistic delays and queueing behavior at both hosts and routers. It uses **discrete-event simulation** to track packets as they move through the network, capturing timing (delay) information at every stage.

**Key Features:**
- Discrete-event simulation engine with priority queue
- Realistic delay modeling (transmission, propagation, processing, queueing)
- Router processing and forwarding logic
- FIFO queue management at hosts and routers
- Detailed performance metrics and CSV export
- Automated visualization of queueing behavior

---

## Quick Start

### Installation

```bash
git clone https://github.com/[your-username]/network-packet-simulator.git
cd network-packet-simulator
pip install -r requirements.txt
```

### Run Simulation

```python
python network_simulator.py
```

**Output:**
- `Outputs (Delay Info)/router_queueing.csv` - Detailed timing data for all packets
- `Outputs (Delay Info)/burst_delay_analysis.png` - Visual breakdown of delay components

---

## Network Topology

The simulator models a simple but realistic network scenario:

```
    Host A ────┐
               ├──→ Router C ───→ Host D
    Host B ────┘
```

**Traffic Pattern:**
- **Host A**: Sends 5 packets every 1000 time units (bursty traffic)
- **Host B**: Sends 2 packets every 500 time units (steady traffic)

This setup demonstrates how different traffic patterns compete for router bandwidth and create queueing delays.

---

## How It Works

### Discrete-Event Simulation

Rather than advancing time in fixed increments, the simulator jumps from event to event:

**Event Types:**
1. **ENQUEUE** - Packet arrives in a queue
2. **TRANSMIT** - Packet finishes transmission
3. **PROPAGATE** - Packet travels through physical medium
4. **RECEIVE** - Packet arrives at destination
5. **PROCESSING** - Router determines next hop

Events are processed in chronological order using a priority queue, ensuring accurate timing even with complex interactions.

### Delay Components

Each packet experiences four types of delay:

**1. Transmission Delay (10 ticks)**
- Time to push all bits onto the wire
- Depends on packet size and link bandwidth
- *Constant in this simulation*

**2. Propagation Delay (1 tick)**
- Time for signal to travel through medium
- Depends on distance and speed of light
- *Constant in this simulation*

**3. Processing Delay (1 tick at router)**
- Time for router to examine packet and determine routing
- Depends on router hardware
- *Constant in this simulation*

**4. Queueing Delay (VARIABLE)**
- Time spent waiting in queues when resources are busy
- Depends on traffic patterns and congestion
- *This is what the simulation explores*

### Example: Packet Journey

**Packet traveling from Host A to Host D:**
```
Time 0:   Enqueue at A (arrives in output queue)
Time 0:   Transmit starts (if queue was empty)
Time 10:  Transmit completes, propagation begins
Time 11:  Receive at Router C
Time 12:  Processing complete, determine next hop = D
Time 12:  Transmit starts from C (if queue was empty)
Time 22:  Transmit completes, propagation begins
Time 23:  Receive at D (DELIVERED)

Total: 23 ticks with no queueing
```

If other packets are transmitting, queueing delays are added.

---

## Visualization

The simulator generates a visualization showing how delays build up within a packet burst:

![Delay Breakdown](Outputs%20(Delay%20Info)/burst_delay_analysis.png)

**What This Shows:**

The graph breaks down end-to-end delay for 5 packets sent in a burst from Host A:

- **Blue (Transmission)**: Constant ~10 ticks per packet
- **Green (Propagation)**: Constant ~1 tick per packet
- **Orange (Processing)**: Constant ~1 tick at router
- **Red (Queueing)**: Variable - increases as burst progresses

**Key Insight:** The first packet experiences minimal queueing (12 ticks total), but the fifth packet experiences significant queueing (72 ticks total) because it must wait for the previous four packets to transmit.

This demonstrates a fundamental networking principle: **bursty traffic creates congestion in network queues**.

---

## Sample Output

### Console

```
Simulation complete. Results exported to Outputs (Delay Info)/router_queueing.csv
Visualization saved to Outputs (Delay Info)/burst_delay_analysis.png
```

### CSV Data (Sample Rows)

| Seq num. | Source | Queue @ source | Transmit @ source | Receive @ C | Transmit @ C | Receive @ D | Queue Delay @ source | Queue Delay @ C | Total Queueing Delay | End-to-End Delay |
|----------|--------|----------------|-------------------|-------------|--------------|-------------|---------------------|----------------|---------------------|-----------------|
| 0        | A      | 0              | 10                | 11          | 22           | 23          | 0                   | 0              | 0                   | 23              |
| 1        | A      | 0              | 20                | 21          | 42           | 43          | 10                  | 10             | 20                  | 43              |
| 2        | A      | 0              | 30                | 31          | 62           | 63          | 20                  | 20             | 40                  | 63              |

**Observations:**
- Packet 0: No queueing (23 ticks end-to-end)
- Packet 1: 20 ticks of queueing (10 at source + 10 at router)
- Packet 2: 40 ticks of queueing (20 at source + 20 at router)

The pattern shows queueing delays building up as the burst progresses.

---

## Code Architecture

### Core Classes

**`Node`** - Base class for network devices
- Manages output queues
- Maintains routing tables

**`Host(Node)`** - End devices (computers, servers)
- Send and receive packets
- Simple FIFO queue

**`Router(Node)`** - Network forwarding devices
- Process packets and determine next hops
- Add processing delays
- Forward based on routing tables

**`Packet`** - Data being transmitted
- Tracks source, destination, and next hop
- Auto-incrementing ID for tracking

**`Event`** - Simulation events
- Five types: ENQUEUE, TRANSMIT, PROPAGATE, RECEIVE, PROCESSING
- Scheduled with timestamps
- Processed in chronological order

**`Simulator`** - Main simulation engine
- Priority queue-based event scheduler
- Tracks timing for all packets
- Exports results and visualizations

### Design Patterns

- **Event-Driven Architecture**: Uses priority queue for efficient event scheduling
- **Object-Oriented Design**: Clean separation of concerns (hosts, routers, packets)
- **Observer Pattern**: Events trigger state changes in network devices
- **Strategy Pattern**: Different node types handle events differently

---

## Networking Concepts Demonstrated

### 1. Queueing Theory
Packets wait in FIFO queues when network resources (transmission links) are busy. The simulation shows how queueing delays grow with traffic intensity.

### 2. Store-and-Forward
Routers must receive an entire packet before forwarding it. This is modeled by the TRANSMIT → PROPAGATE → RECEIVE → PROCESSING → TRANSMIT sequence.

### 3. Congestion
When multiple traffic sources compete for the same router, packets experience increased queueing delays. The simulation demonstrates this with Host A and Host B both sending to Router C.

### 4. Traffic Patterns
- **Bursty traffic** (Host A): Creates temporary congestion
- **Steady traffic** (Host B): Lower average delays but can be affected by bursts

### 5. End-to-End Performance
Total delay = Transmission + Propagation + Processing + Queueing

The first three are predictable; queueing is the variable component affected by network conditions.

---

## Technical Highlights

- **Type hints throughout** - Modern Python best practices
- **Comprehensive docstrings** - PEP 257 compliant documentation
- **Clean architecture** - Separated event handlers for maintainability
- **Error handling** - Graceful degradation if visualization libraries unavailable
- **Professional visualization** - Publication-quality graphs with matplotlib

---

## Project Structure

```
network-packet-simulator/
├── network_simulator.py          # Main simulator code
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── LICENSE                       # MIT License
└── Outputs (Delay Info)/         # Generated outputs
    ├── router_queueing.csv       # Detailed timing data
    └── burst_delay_analysis.png  # Delay visualization
```

---

## Requirements

- Python 3.7+
- matplotlib >= 3.5.0
- numpy >= 1.21.0

Install with:
```bash
pip install -r requirements.txt
```

---

## Usage Examples

### Basic Run
```python
python network_simulator.py
```

### With Verbose Output
```python
from network_simulator import run_router_experiment

run_router_experiment(verbose=True)
```

This prints each event as it's processed:
```
   0 ENQUEUE      A pkt=P[0,A->D] next=C
   0 TRANSMIT     A pkt=P[0,A->D] next=C
  10 PROPAGATE    C pkt=P[0,A->D] next=C
  11 RECEIVE      C pkt=P[0,A->D] next=C
  ...
```

### Custom Output Directory
```python
run_router_experiment(output_dir="my_results")
```

---

## What I Learned

This project deepened my understanding of:

- **Discrete-event simulation** - How to model complex systems by processing events chronologically
- **Computer networking** - Practical understanding of delay components and queueing behavior
- **Data structures** - Priority queues for efficient event scheduling
- **Performance analysis** - Identifying bottlenecks and understanding congestion patterns
- **Data visualization** - Presenting technical data in intuitive visual formats

---

## Potential Extensions

While the current simulator is feature-complete for demonstrating core concepts, potential enhancements could include:

- **Variable packet sizes** - Model realistic traffic with different packet lengths
- **Multiple routers** - Create more complex network topologies
- **Packet loss** - Simulate congestion-induced drops and retransmissions
- **Priority queues** - Model Quality of Service (QoS) mechanisms
- **Dynamic routing** - Implement routing protocols that adapt to congestion
- **Network protocols** - Add TCP-like flow control and congestion avoidance

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Author

**Connor** (pospsl)  
Computer Science Student

*Developed as a course project exploring discrete-event simulation and computer networking fundamentals.*

---

## Acknowledgments

- Original framework developed for computer networking coursework
- Discrete-event simulation concepts from standard networking textbooks
- Visualization inspired by network performance analysis tools

---
