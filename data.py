import random
from typing import List, Set, Dict
import json
from collections import defaultdict

class TransactionGenerator:
    def __init__(self):
        # Define product categories and items within each category
        self.categories = {
            'dairy': ['milk', 'cheese', 'yogurt', 'butter', 'cream', 'cream cheese'],
            'bread': ['white bread', 'wheat bread', 'bagels', 'rolls', 'croissants'],
            'produce': ['apples', 'bananas', 'carrots', 'lettuce', 'tomatoes'],
            'meat': ['chicken', 'beef', 'pork', 'fish', 'ham'],
            'snacks': ['chips', 'cookies', 'crackers', 'candy', 'popcorn'],
            'beverages': ['soda', 'juice', 'coffee', 'tea', 'water'],
            'household': ['paper towels', 'toilet paper', 'detergent', 'soap', 'trash bags']
        }
        
        # Define common item associations (items frequently bought together)
        self.associations = [
            (['white bread', 'butter'], 0.7),    # 70% chance of buying butter if buying white bread
            (['coffee', 'cream'], 0.6),          # 60% chance of buying cream with coffee
            (['chips', 'soda'], 0.5),            # 50% chance of buying soda with chips
            (['bagels', 'cream cheese'], 0.8),   # 80% chance of buying cream cheese with bagels
            (['chicken', 'lettuce'], 0.4),       # 40% chance of buying lettuce with chicken
        ]
        
        # Base probabilities for each category (likelihood of appearing in any transaction)
        self.category_probabilities = {
            'dairy': 0.6,
            'bread': 0.5,
            'produce': 0.7,
            'meat': 0.4,
            'snacks': 0.3,
            'beverages': 0.5,
            'household': 0.2
        }

    def generate_transaction(self) -> Set[str]:
        """
        Generate a single transaction with realistic shopping patterns.
        Takes into account category probabilities and item associations.
        """
        transaction = set()
        
        # First, decide which categories will be represented in this transaction
        selected_categories = [
            category for category in self.categories
            if random.random() < self.category_probabilities[category]
        ]
        
        # Add 1-2 random items from each selected category
        for category in selected_categories:
            num_items = random.randint(1, 2)
            items = random.sample(self.categories[category], num_items)
            transaction.update(items)
        
        # Apply associations
        for base_items, probability in self.associations:
            # If all base items are in transaction, maybe add associated items
            if all(item in transaction for item in base_items[:-1]):
                if random.random() < probability:
                    transaction.add(base_items[-1])
        
        return transaction

    def generate_dataset(self, num_transactions: int) -> List[Set[str]]:
        """
        Generate a dataset with the specified number of transactions.
        
        Args:
            num_transactions: Number of transactions to generate
            
        Returns:
            List of sets, where each set contains items in a transaction
        """
        transactions = []
        for _ in range(num_transactions):
            transaction = self.generate_transaction()
            transactions.append(transaction)
        
        return transactions

    def analyze_dataset(self, transactions: List[Set[str]]) -> Dict:
        """
        Analyze the generated dataset to show transaction statistics.
        """
        stats = {
            'total_transactions': len(transactions),
            'avg_items_per_transaction': sum(len(t) for t in transactions) / len(transactions),
            'item_frequencies': defaultdict(int),
            'category_frequencies': defaultdict(int),
            'largest_transaction': max(transactions, key=len)
        }
        
        # Calculate item and category frequencies
        for transaction in transactions:
            for item in transaction:
                stats['item_frequencies'][item] += 1
                # Find category for this item
                for category, items in self.categories.items():
                    if item in items:
                        stats['category_frequencies'][category] += 1
                        break
        
        # Convert frequencies to percentages
        stats['item_frequencies'] = {
            item: count/len(transactions) 
            for item, count in stats['item_frequencies'].items()
        }
        stats['category_frequencies'] = {
            category: count/len(transactions) 
            for category, count in stats['category_frequencies'].items()
        }

        stats['largest_transaction'] = list(stats['largest_transaction'])
        
        return stats

# Example usage
if __name__ == "__main__":
    # Create generator and generate dataset
    generator = TransactionGenerator()
    transactions = generator.generate_dataset(100000)
    
    # Analyze the generated dataset
    stats = generator.analyze_dataset(transactions)
    
    # Print some statistics
    print(f"\nDataset Statistics:")
    print(f"Total Transactions: {stats['total_transactions']}")
    print(f"Average Items per Transaction: {stats['avg_items_per_transaction']:.2f}")
    print(f"Largest Transaction: {stats['largest_transaction']}")
    
    print("\nCategory Frequencies:")
    for category, freq in sorted(stats['category_frequencies'].items(), 
                               key=lambda x: x[1], reverse=True):
        print(f"{category}: {freq:.1%}")
    
    print("\nTop 10 Most Common Items:")
    top_items = sorted(stats['item_frequencies'].items(), 
                      key=lambda x: x[1], reverse=True)[:10]
    for item, freq in top_items:
        print(f"{item}: {freq:.1%}")
    
    # Save transactions to file
    with open('transactions.json', 'w') as f:
        # Convert sets to lists for JSON serialization
        json_transactions = [list(t) for t in transactions]
        json.dump(json_transactions, f)