# Only Multi-Products Environments

This category includes negotiation environments with:
- Single buyer agent
- Multiple products
- Single seller agent

## Tasks

### Task1: Multi-Product Negotiation

The fundamental multi-product negotiation environment that manages continuous negotiation processes for multiple products between a buyer and a seller. This task supports preserving conversation context across different products, allowing agents to negotiate multiple items in sequence while maintaining the negotiation history and context.

### Task2: Two-Product Negotiation

This environment manages negotiation for two products with total price negotiation. The buyer_max_price and seller_min_price represent the total expected cost for both products combined. This task tests agents' ability to negotiate a bundle deal where the total price matters rather than individual product prices.

### Task3: Five-Product Negotiation

This environment manages negotiation for five products with total price negotiation. The buyer_max_price and seller_min_price represent the total expected cost for all five products combined. This task challenges agents to handle more complex multi-product scenarios with a larger number of items.

### Task4: Select Three from Five Products Negotiation

This environment manages negotiation where the user needs 3 products, and the buyer agent automatically selects 3 from 5 available products for total price negotiation. The buyer_max_price and seller_min_price represent the total expected cost for the selected 3 products. This task tests agents' ability to make product selection decisions while negotiating prices.
