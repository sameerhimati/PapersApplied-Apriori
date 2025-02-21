from collections import defaultdict
import math
import random
from typing import List, Set, Dict, Tuple

class AprioriMiner:
    def __init__(self, transactions: List[Set[str]], min_support: float, sample_size: int = 1000, confidence_level: float = 0.95):
        """
        Initialize the Apriori algorithm with optimization techniques from the 1993 paper.
        
        Args:
            transactions: List of sets, where each set contains items in a transaction
            min_support: Minimum support threshold (between 0 and 1)
            sample_size: Size of sample for support estimation
            confidence_level: Confidence level for support estimation (default 0.95)
        """
        self.transactions = transactions
        self.min_support = min_support
        self.n_transactions = len(transactions)
        self.sample_size = min(sample_size, self.n_transactions)
        self.z_score = 1.96  # z-score for 95% confidence level
        
        # Storage for frequent itemsets and boundary sets
        self.frequent_itemsets = {}  # key: size of itemset, value: set of frequent itemsets
        self.boundary_sets = set()   # stores minimal infrequent itemsets
        
        # Memory management: track which transactions can be safely ignored
        self.active_transactions = set(range(self.n_transactions))
    
    def estimate_support(self, itemset: frozenset) -> bool:
        """
        Estimate if an itemset is likely to be frequent using sampling.
        Implementation of the "carefully tuned estimation procedure" from the paper.
        
        Returns:
            bool: True if itemset is expected to be frequent
        """
        # Take a random sample of transactions
        sample_indices = random.sample(list(self.active_transactions), self.sample_size)
        
        # Count occurrences in sample
        count = sum(1 for idx in sample_indices if itemset.issubset(self.transactions[idx]))
        p_hat = count / self.sample_size
        
        # Calculate upper bound of confidence interval
        margin = self.z_score * math.sqrt((p_hat * (1 - p_hat)) / self.sample_size)
        upper_bound = p_hat + margin
        
        return upper_bound >= self.min_support
    
    def find_frequent_1_itemsets(self) -> Set[frozenset]:
        """
        Find all frequent 1-itemsets by scanning the database once.
        """
        item_counts = defaultdict(int)
        
        # Count occurrences of each item
        for idx in self.active_transactions:
            for item in self.transactions[idx]:
                item_counts[item] += 1
        
        # Find items that meet minimum support
        frequent_1_itemsets = {frozenset([item]) for item, count in item_counts.items() 
                             if count / self.n_transactions >= self.min_support}
        
        # Update boundary sets with minimal infrequent 1-itemsets
        self.boundary_sets.update(
            frozenset([item]) for item, count in item_counts.items()
            if count / self.n_transactions < self.min_support
        )
        
        return frequent_1_itemsets
    
    def generate_candidates(self, prev_frequent: Set[frozenset]) -> Set[frozenset]:
        """
        Generate candidate itemsets by combining previous frequent itemsets.
        Uses pruning based on boundary sets.
        """
        candidates = set()
        prev_list = list(prev_frequent)
        k = len(next(iter(prev_frequent))) + 1  # size of new candidates
        
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                items1 = prev_list[i]
                items2 = prev_list[j]
                
                # Check if first k-2 items are the same
                if list(items1)[:-1] == list(items2)[:-1]:
                    # Create new candidate
                    candidate = items1.union(items2)
                    
                    # Pruning: check if any subset is in boundary sets
                    if not any(boundary.issubset(candidate) for boundary in self.boundary_sets):
                        # Estimation-based pruning
                        if self.estimate_support(candidate):
                            candidates.add(candidate)
        
        return candidates
    
    def calculate_support(self, candidates: Set[frozenset]) -> Dict[frozenset, float]:
        """
        Calculate actual support for candidate itemsets.
        Uses memory management optimization.
        """
        supports = defaultdict(int)
        removable_transactions = set()
        
        # Count support for each candidate
        for idx in self.active_transactions:
            transaction = self.transactions[idx]
            found_frequent = False
            
            for candidate in candidates:
                if candidate.issubset(transaction):
                    supports[candidate] += 1
                    found_frequent = True
            
            # If transaction doesn't contain any candidates, mark it for removal
            if not found_frequent:
                removable_transactions.add(idx)
        
        # Memory management: remove unnecessary transactions
        self.active_transactions -= removable_transactions
        
        # Convert counts to support values
        return {itemset: count / self.n_transactions for itemset, count in supports.items()}
    
    def mine_frequent_itemsets(self) -> Dict[int, Set[frozenset]]:
        """
        Main method to mine frequent itemsets using all optimization techniques.
        """
        # Find frequent 1-itemsets
        frequent_1_itemsets = self.find_frequent_1_itemsets()
        self.frequent_itemsets[1] = frequent_1_itemsets
        
        k = 2
        while self.frequent_itemsets[k-1]:
            # Generate candidates using previous frequent itemsets
            candidates = self.generate_candidates(self.frequent_itemsets[k-1])
            
            if not candidates:
                break
                
            # Calculate support for candidates
            supports = self.calculate_support(candidates)
            
            # Find frequent k-itemsets and update boundary sets
            frequent_k = {itemset for itemset, sup in supports.items() 
                         if sup >= self.min_support}
            boundary_k = {itemset for itemset, sup in supports.items() 
                         if sup < self.min_support}
            
            self.frequent_itemsets[k] = frequent_k
            self.boundary_sets.update(boundary_k)
            
            k += 1
        
        return self.frequent_itemsets

# Example usage:
if __name__ == "__main__":
    # Sample transaction database
    transactions = [
        {"bread", "milk"},
        {"bread", "diapers", "beer", "eggs"},
        {"milk", "diapers", "beer", "cola"},
        {"bread", "milk", "diapers", "beer"},
        {"bread", "milk", "diapers", "cola"}
    ]
    
    # Initialize and run algorithm
    miner = AprioriMiner(transactions, min_support=0.3)
    frequent_itemsets = miner.mine_frequent_itemsets()
    
    # Print results
    for k, itemsets in frequent_itemsets.items():
        print(f"\nFrequent {k}-itemsets:")
        for itemset in itemsets:
            print(itemset)