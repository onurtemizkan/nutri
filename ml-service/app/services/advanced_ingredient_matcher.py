"""
Advanced Ingredient Matcher with High-Performance Data Structures

Optimizations:
- Trie-based exact matching: O(m) where m = query length
- BK-Tree for fuzzy matching: O(log n) average case
- LRU caching with TTL for repeated queries
- N-gram indexing for partial matches
- Phonetic matching (Soundex/Metaphone) for spelling variations
- Semantic embeddings for conceptual similarity (optional)

Performance targets:
- Exact match: <1ms
- Fuzzy match: <10ms for 10k ingredients
- Batch processing: 1000 queries/second
"""

import logging
import hashlib
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass
from collections import defaultdict
import time

# Try to import optional advanced dependencies
try:
    from rapidfuzz.distance import Levenshtein

    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import jellyfish

    PHONETIC_AVAILABLE = True
except ImportError:
    PHONETIC_AVAILABLE = False

from app.data.allergen_database import (
    INGREDIENT_DATABASE,
    IngredientData,
)

logger = logging.getLogger(__name__)


# =============================================================================
# TRIE DATA STRUCTURE FOR EXACT/PREFIX MATCHING
# =============================================================================


class TrieNode:
    """Node in a Trie (prefix tree)."""

    __slots__ = ["children", "is_end", "ingredient_key", "weight"]

    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.is_end: bool = False
        self.ingredient_key: Optional[str] = None
        self.weight: float = 1.0  # For ranking


class Trie:
    """
    Trie (prefix tree) for O(m) exact and prefix matching.

    Supports:
    - Exact match lookup
    - Prefix search (autocomplete)
    - Weighted results for ranking
    """

    def __init__(self):
        self.root = TrieNode()
        self.size = 0

    def insert(self, word: str, ingredient_key: str, weight: float = 1.0):
        """Insert a word with associated ingredient key."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.ingredient_key = ingredient_key
        node.weight = weight
        self.size += 1

    def search(self, word: str) -> Optional[str]:
        """Exact match search. Returns ingredient_key or None."""
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return None
            node = node.children[char]
        return node.ingredient_key if node.is_end else None

    def starts_with(self, prefix: str, limit: int = 10) -> List[Tuple[str, str, float]]:
        """
        Find all words starting with prefix.

        Returns:
            List of (word, ingredient_key, weight) tuples
        """
        node = self.root
        for char in prefix.lower():
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        self._collect_words(node, prefix.lower(), results, limit)
        results.sort(key=lambda x: -x[2])  # Sort by weight descending
        return results[:limit]

    def _collect_words(
        self,
        node: TrieNode,
        current_word: str,
        results: List[Tuple[str, str, float]],
        limit: int,
    ):
        """Recursively collect words from a node."""
        if len(results) >= limit * 2:  # Collect extra for sorting
            return

        if node.is_end:
            results.append((current_word, node.ingredient_key, node.weight))

        for char, child_node in node.children.items():
            self._collect_words(child_node, current_word + char, results, limit)


# =============================================================================
# BK-TREE FOR FUZZY MATCHING
# =============================================================================


class BKTreeNode:
    """Node in a BK-Tree."""

    __slots__ = ["word", "ingredient_key", "children"]

    def __init__(self, word: str, ingredient_key: str):
        self.word = word
        self.ingredient_key = ingredient_key
        self.children: Dict[int, "BKTreeNode"] = {}


class BKTree:
    """
    Burkhard-Keller Tree for efficient fuzzy string matching.

    Uses edit distance as the metric. Average case O(log n) for queries.
    """

    def __init__(self):
        self.root: Optional[BKTreeNode] = None
        self.size = 0

    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance."""
        if RAPIDFUZZ_AVAILABLE:
            return Levenshtein.distance(s1, s2)
        else:
            # Simple DP implementation
            if len(s1) < len(s2):
                s1, s2 = s2, s1
            if len(s2) == 0:
                return len(s1)

            prev_row = range(len(s2) + 1)
            for i, c1 in enumerate(s1):
                curr_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = prev_row[j + 1] + 1
                    deletions = curr_row[j] + 1
                    substitutions = prev_row[j] + (c1 != c2)
                    curr_row.append(min(insertions, deletions, substitutions))
                prev_row = curr_row
            return prev_row[-1]

    def insert(self, word: str, ingredient_key: str):
        """Insert a word into the BK-Tree."""
        word_lower = word.lower()

        if self.root is None:
            self.root = BKTreeNode(word_lower, ingredient_key)
            self.size += 1
            return

        node = self.root
        while True:
            dist = self._edit_distance(word_lower, node.word)
            if dist == 0:
                return  # Duplicate word

            if dist in node.children:
                node = node.children[dist]
            else:
                node.children[dist] = BKTreeNode(word_lower, ingredient_key)
                self.size += 1
                return

    def search(
        self, query: str, max_distance: int = 2, limit: int = 10
    ) -> List[Tuple[str, str, int]]:
        """
        Find all words within max_distance of query.

        Returns:
            List of (word, ingredient_key, distance) tuples, sorted by distance
        """
        if self.root is None:
            return []

        results = []
        query_lower = query.lower()
        self._search_recursive(self.root, query_lower, max_distance, results)

        # Sort by distance and limit
        results.sort(key=lambda x: x[2])
        return results[:limit]

    def _search_recursive(
        self,
        node: BKTreeNode,
        query: str,
        max_distance: int,
        results: List[Tuple[str, str, int]],
    ):
        """Recursively search the BK-Tree."""
        dist = self._edit_distance(query, node.word)

        if dist <= max_distance:
            results.append((node.word, node.ingredient_key, dist))

        # Only visit children in the valid range
        for d in range(max(0, dist - max_distance), dist + max_distance + 1):
            if d in node.children:
                self._search_recursive(node.children[d], query, max_distance, results)


# =============================================================================
# N-GRAM INDEX FOR PARTIAL MATCHING
# =============================================================================


class NGramIndex:
    """
    N-gram index for fast partial/substring matching.

    Creates n-grams (character sequences) and inverts the index
    for quick lookup of words containing specific sequences.
    """

    def __init__(self, n: int = 3):
        self.n = n
        self.index: Dict[str, Set[str]] = defaultdict(set)
        self.words: Dict[str, str] = {}  # word -> ingredient_key

    def _get_ngrams(self, word: str) -> Set[str]:
        """Generate n-grams for a word."""
        word = f"${word}$"  # Add boundary markers
        ngrams = set()
        for i in range(len(word) - self.n + 1):
            ngrams.add(word[i : i + self.n])
        return ngrams

    def insert(self, word: str, ingredient_key: str):
        """Add a word to the index."""
        word_lower = word.lower()
        self.words[word_lower] = ingredient_key

        for ngram in self._get_ngrams(word_lower):
            self.index[ngram].add(word_lower)

    def search(
        self, query: str, min_overlap: float = 0.5, limit: int = 10
    ) -> List[Tuple[str, str, float]]:
        """
        Find words with overlapping n-grams.

        Args:
            query: Search query
            min_overlap: Minimum fraction of query n-grams that must match
            limit: Maximum results

        Returns:
            List of (word, ingredient_key, overlap_score) tuples
        """
        query_ngrams = self._get_ngrams(query.lower())
        if not query_ngrams:
            return []

        # Count matches for each candidate word
        candidates: Dict[str, int] = defaultdict(int)
        for ngram in query_ngrams:
            if ngram in self.index:
                for word in self.index[ngram]:
                    candidates[word] += 1

        # Calculate overlap score and filter
        min_matches = int(len(query_ngrams) * min_overlap)
        results = []

        for word, match_count in candidates.items():
            if match_count >= min_matches:
                word_ngrams = self._get_ngrams(word)
                # Jaccard similarity
                overlap = len(query_ngrams & word_ngrams) / len(
                    query_ngrams | word_ngrams
                )
                results.append((word, self.words[word], overlap))

        results.sort(key=lambda x: -x[2])
        return results[:limit]


# =============================================================================
# PHONETIC MATCHER
# =============================================================================


class PhoneticMatcher:
    """
    Phonetic matching using Soundex and Metaphone.

    Matches words that sound similar but are spelled differently.
    E.g., "parmesan" and "parmesean" would match.
    """

    def __init__(self):
        self.soundex_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.metaphone_index: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    def _soundex(self, word: str) -> str:
        """Calculate Soundex code."""
        if PHONETIC_AVAILABLE:
            return jellyfish.soundex(word)
        else:
            # Simple Soundex implementation
            word = word.upper()
            if not word:
                return "0000"

            # First letter
            code = word[0]

            # Character mappings
            mapping = {
                "B": "1",
                "F": "1",
                "P": "1",
                "V": "1",
                "C": "2",
                "G": "2",
                "J": "2",
                "K": "2",
                "Q": "2",
                "S": "2",
                "X": "2",
                "Z": "2",
                "D": "3",
                "T": "3",
                "L": "4",
                "M": "5",
                "N": "5",
                "R": "6",
            }

            prev_digit = mapping.get(word[0], "")
            for char in word[1:]:
                digit = mapping.get(char, "")
                if digit and digit != prev_digit:
                    code += digit
                    if len(code) == 4:
                        break
                prev_digit = digit if digit else prev_digit

            return (code + "0000")[:4]

    def _metaphone(self, word: str) -> str:
        """Calculate Metaphone code."""
        if PHONETIC_AVAILABLE:
            return jellyfish.metaphone(word)
        else:
            # Fall back to Soundex
            return self._soundex(word)

    def insert(self, word: str, ingredient_key: str):
        """Add a word to the phonetic indices."""
        soundex = self._soundex(word)
        metaphone = self._metaphone(word)

        self.soundex_index[soundex].append((word.lower(), ingredient_key))
        self.metaphone_index[metaphone].append((word.lower(), ingredient_key))

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, str, str]]:
        """
        Find phonetically similar words.

        Returns:
            List of (word, ingredient_key, match_type) tuples
        """
        results = []
        seen = set()

        # Search by Soundex
        soundex = self._soundex(query)
        for word, key in self.soundex_index.get(soundex, []):
            if word not in seen:
                results.append((word, key, "soundex"))
                seen.add(word)

        # Search by Metaphone
        metaphone = self._metaphone(query)
        for word, key in self.metaphone_index.get(metaphone, []):
            if word not in seen:
                results.append((word, key, "metaphone"))
                seen.add(word)

        return results[:limit]


# =============================================================================
# LRU CACHE WITH TTL
# =============================================================================


@dataclass
class CacheEntry:
    """Cache entry with timestamp."""

    value: Any
    timestamp: float
    hits: int = 0


class TTLCache:
    """
    LRU Cache with Time-To-Live expiration.

    Features:
    - Automatic expiration of old entries
    - Hit counting for analytics
    - Memory-bounded with max_size
    """

    def __init__(self, max_size: int = 10000, ttl_seconds: float = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.hits = 0
        self.misses = 0

    def _make_key(self, *args, **kwargs) -> str:
        """Create cache key from arguments."""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key not in self.cache:
            self.misses += 1
            return None

        entry = self.cache[key]

        # Check TTL
        if time.time() - entry.timestamp > self.ttl:
            del self.cache[key]
            self.access_order.remove(key)
            self.misses += 1
            return None

        # Update access order
        self.access_order.remove(key)
        self.access_order.append(key)
        entry.hits += 1
        self.hits += 1

        return entry.value

    def set(self, key: str, value: Any):
        """Set value in cache."""
        # Evict if at max size
        while len(self.cache) >= self.max_size:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = CacheEntry(value=value, timestamp=time.time())
        self.access_order.append(key)

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0,
            "ttl_seconds": self.ttl,
        }


# =============================================================================
# ADVANCED INGREDIENT MATCHER
# =============================================================================


@dataclass
class MatchResult:
    """Result from ingredient matching."""

    query: str
    ingredient_key: str
    ingredient_data: Optional[IngredientData]
    score: float  # 0.0 - 1.0
    match_type: str  # exact, prefix, fuzzy, ngram, phonetic
    edit_distance: Optional[int] = None

    # Property aliases for API consistency
    @property
    def ingredient(self) -> str:
        """Alias for ingredient_key."""
        return self.ingredient_key

    @property
    def similarity(self) -> float:
        """Alias for score."""
        return self.score


class AdvancedIngredientMatcher:
    """
    High-performance ingredient matcher using multiple data structures.

    Search order:
    1. Exact match (Trie) - O(m)
    2. Prefix match (Trie) - O(m + k)
    3. Phonetic match - O(1) lookup
    4. N-gram match - O(q * avg_matches)
    5. Fuzzy match (BK-Tree) - O(log n)

    Features:
    - Sub-millisecond exact matches
    - <10ms fuzzy matches for large databases
    - Automatic caching of repeated queries
    - Batch processing support
    """

    def __init__(self, cache_size: int = 10000, cache_ttl: float = 3600):
        """Initialize all data structures."""
        self.trie = Trie()
        self.bk_tree = BKTree()
        self.ngram_index = NGramIndex(n=3)
        self.phonetic_matcher = PhoneticMatcher()
        self.cache = TTLCache(max_size=cache_size, ttl_seconds=cache_ttl)

        # Build indices from database
        self._build_indices()

        logger.info(
            f"Initialized AdvancedIngredientMatcher: "
            f"Trie={self.trie.size}, BK-Tree={self.bk_tree.size}, "
            f"Phonetic={'enabled' if PHONETIC_AVAILABLE else 'disabled'}"
        )

    def _build_indices(self):
        """Build all search indices from ingredient database."""
        for key, data in INGREDIENT_DATABASE.items():
            # Insert main name
            self._insert_all(key, key, weight=1.0)
            self._insert_all(data.display_name, key, weight=0.95)

            # Insert name variants
            if data.name_variants:
                for i, variant in enumerate(data.name_variants):
                    weight = 0.9 - (i * 0.05)  # Decreasing weight for variants
                    self._insert_all(variant, key, weight=max(0.5, weight))

    def _insert_all(self, word: str, ingredient_key: str, weight: float = 1.0):
        """Insert word into all indices."""
        self.trie.insert(word, ingredient_key, weight)
        self.bk_tree.insert(word, ingredient_key)
        self.ngram_index.insert(word, ingredient_key)
        self.phonetic_matcher.insert(word, ingredient_key)

    def add_ingredient(
        self,
        name: str,
        metadata: Optional[Dict[str, Any]] = None,
        variants: Optional[List[str]] = None,
        weight: float = 1.0,
    ) -> None:
        """
        Dynamically add an ingredient to all indices.

        Args:
            name: Ingredient name (used as key)
            metadata: Optional metadata (not stored, for compatibility)
            variants: Optional list of name variants
            weight: Base weight for matching (default 1.0)
        """
        # Insert main name
        self._insert_all(name, name, weight)

        # Insert variants
        if variants:
            for i, variant in enumerate(variants):
                variant_weight = weight * (0.9 - (i * 0.05))
                self._insert_all(variant, name, max(0.5, variant_weight))

    def match(
        self,
        query: str,
        threshold: float = 0.75,
        max_results: int = 5,
        use_cache: bool = True,
    ) -> List[MatchResult]:
        """
        Find best matches for a query.

        Args:
            query: Search query
            threshold: Minimum score (0-1)
            max_results: Maximum results to return
            use_cache: Whether to use cache

        Returns:
            List of MatchResult sorted by score
        """
        if not query or len(query) < 2:
            return []

        # Check cache
        cache_key = f"match:{query}:{threshold}:{max_results}"
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return cached

        results: List[MatchResult] = []
        seen_keys: Set[str] = set()

        # 1. Exact match (highest priority)
        exact_key = self.trie.search(query)
        if exact_key and exact_key not in seen_keys:
            results.append(
                MatchResult(
                    query=query,
                    ingredient_key=exact_key,
                    ingredient_data=INGREDIENT_DATABASE.get(exact_key),
                    score=1.0,
                    match_type="exact",
                )
            )
            seen_keys.add(exact_key)

        # 2. Prefix matches
        if len(results) < max_results:
            prefix_matches = self.trie.starts_with(query, limit=max_results)
            for word, key, weight in prefix_matches:
                if key not in seen_keys:
                    # Score based on how much of the word the prefix covers
                    coverage = len(query) / len(word)
                    score = weight * coverage * 0.95  # Slightly lower than exact
                    if score >= threshold:
                        results.append(
                            MatchResult(
                                query=query,
                                ingredient_key=key,
                                ingredient_data=INGREDIENT_DATABASE.get(key),
                                score=score,
                                match_type="prefix",
                            )
                        )
                        seen_keys.add(key)

        # 3. Phonetic matches
        if len(results) < max_results:
            phonetic_matches = self.phonetic_matcher.search(query, limit=max_results)
            for word, key, match_type in phonetic_matches:
                if key not in seen_keys:
                    # Phonetic matches get moderate score
                    score = 0.85
                    if score >= threshold:
                        results.append(
                            MatchResult(
                                query=query,
                                ingredient_key=key,
                                ingredient_data=INGREDIENT_DATABASE.get(key),
                                score=score,
                                match_type=f"phonetic_{match_type}",
                            )
                        )
                        seen_keys.add(key)

        # 4. N-gram matches
        if len(results) < max_results:
            ngram_matches = self.ngram_index.search(
                query, min_overlap=0.4, limit=max_results
            )
            for word, key, overlap in ngram_matches:
                if key not in seen_keys:
                    score = overlap * 0.9  # Scale overlap to score
                    if score >= threshold:
                        results.append(
                            MatchResult(
                                query=query,
                                ingredient_key=key,
                                ingredient_data=INGREDIENT_DATABASE.get(key),
                                score=score,
                                match_type="ngram",
                            )
                        )
                        seen_keys.add(key)

        # 5. Fuzzy matches (BK-Tree) - most expensive, do last
        if len(results) < max_results:
            max_edit_dist = max(1, len(query) // 4)  # Scale with query length
            fuzzy_matches = self.bk_tree.search(
                query, max_distance=max_edit_dist, limit=max_results
            )
            for word, key, distance in fuzzy_matches:
                if key not in seen_keys:
                    # Convert edit distance to score
                    max_len = max(len(query), len(word))
                    score = 1.0 - (distance / max_len)
                    if score >= threshold:
                        results.append(
                            MatchResult(
                                query=query,
                                ingredient_key=key,
                                ingredient_data=INGREDIENT_DATABASE.get(key),
                                score=score,
                                match_type="fuzzy",
                                edit_distance=distance,
                            )
                        )
                        seen_keys.add(key)

        # Sort by score and limit
        results.sort(key=lambda r: -r.score)
        results = results[:max_results]

        # Cache results
        if use_cache:
            self.cache.set(cache_key, results)

        return results

    def match_batch(
        self,
        queries: List[str],
        threshold: float = 0.75,
        max_results_per_query: int = 3,
    ) -> Dict[str, List[MatchResult]]:
        """
        Match multiple queries efficiently.

        Args:
            queries: List of search queries
            threshold: Minimum score
            max_results_per_query: Max results per query

        Returns:
            Dict mapping query -> list of results
        """
        results = {}
        for query in queries:
            results[query] = self.match(
                query,
                threshold=threshold,
                max_results=max_results_per_query,
            )
        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self.cache.stats()

    def clear_cache(self):
        """Clear the cache."""
        self.cache = TTLCache(max_size=self.cache.max_size, ttl_seconds=self.cache.ttl)


# Singleton instance
advanced_ingredient_matcher = AdvancedIngredientMatcher()
