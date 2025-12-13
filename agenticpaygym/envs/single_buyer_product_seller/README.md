# Single Buyer + Product + Seller Environments

This category includes negotiation environments with:
- Single buyer agent
- Single product
- Single seller agent

## Tasks

### Task1: Basic Price Negotiation

The fundamental price negotiation environment that manages multi-round negotiation processes between a buyer and a seller. This task serves as the base implementation for all other tasks in this category. It handles standard price negotiation scenarios where both parties have their confidential price limits and negotiate to reach an agreement.

### Task2: Close Price Negotiation

This environment tests edge cases where the buyer's maximum acceptable price is very close to the seller's minimum acceptable price. It is designed to evaluate whether agents can successfully reach a deal when the negotiation space is extremely narrow. This scenario challenges the agents' ability to find common ground in tight price ranges.

### Task3: Close to Market Price Negotiation

This environment tests scenarios where the seller's minimum acceptable price is close to the product's market price (listed price). It examines whether a deal can be reached when the seller's bottom line is near the market value, which represents a realistic scenario where sellers have limited flexibility in pricing.

