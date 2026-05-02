"""
Rule-based query classifier.
Decides whether to search the people_collection, places_collection, or both.

Strategy combines two signals:
  1. Direct entity-name match against the known PEOPLE / PLACES lists.
  2. Keyword scoring (person-signal vs place-signal vocabulary).

Entity-name matches dominate: if the query mentions a known person and no
known place, we are confident this is a person query (and vice-versa).
"""

import re
import logging

from config import PEOPLE, PLACES

logger = logging.getLogger(__name__)

# Signals that suggest the query is about a person
_PERSON_KEYWORDS = {
    "who", "whose", "whom", "born", "died", "raised", "lived", "alive",
    "invented", "discovered", "wrote", "painted", "composed", "sang",
    "scientist", "physicist", "chemist", "biologist", "mathematician",
    "artist", "musician", "singer", "composer", "actor", "actress",
    "author", "writer", "poet", "philosopher", "politician", "president",
    "king", "queen", "emperor", "general", "leader", "athlete", "player",
    "footballer", "engineer", "inventor", "celebrity", "famous",
    "he", "she", "his", "her", "him", "himself", "herself",
    "man", "woman", "person", "people", "life", "career",
    "biography", "childhood", "nationality", "wife", "husband", "child",
    "discover",
}

# Signals that suggest the query is about a place / landmark
_PLACE_KEYWORDS = {
    "where", "located", "location", "country", "city", "town", "village",
    "region", "continent", "built", "construction", "architecture",
    "monument", "landmark", "tower", "wall", "palace", "temple", "pyramid",
    "statue", "mountain", "canyon", "waterfall", "island", "lake", "river",
    "ocean", "park", "national", "heritage", "site", "place", "visit",
    "travel", "tourism", "tourist", "destination", "building", "structure",
    "ruin", "ancient", "historical", "height", "elevation", "meters",
}


def _tokenise(text: str) -> set[str]:
    return set(re.findall(r"\b[a-z]+\b", text.lower()))


def _count_entity_matches(query: str, entities: list[str]) -> int:
    """Return how many entity titles appear (case-insensitive) in the query."""
    q_lower = query.lower()
    return sum(1 for e in entities if e.lower() in q_lower)


def classify_query(query: str) -> str:
    """
    Return 'person', 'place', or 'both'.

    1. Detect direct entity-name mentions (highest signal).
    2. Fall back to keyword scoring if no entity is found.
    3. If both sides tie, return 'both' to query both collections.
    """
    person_hits = _count_entity_matches(query, PEOPLE)
    place_hits = _count_entity_matches(query, PLACES)

    if person_hits and not place_hits:
        logger.debug("Classifier: name match → person (q=%r)", query)
        return "person"
    if place_hits and not person_hits:
        logger.debug("Classifier: name match → place (q=%r)", query)
        return "place"
    if person_hits and place_hits:
        logger.debug("Classifier: both names matched → both (q=%r)", query)
        return "both"

    # No direct name match — fall back to keyword scoring
    words = _tokenise(query)
    person_score = len(words & _PERSON_KEYWORDS)
    place_score = len(words & _PLACE_KEYWORDS)

    logger.debug(
        "Classifier keyword scores — person=%d place=%d (q=%r)",
        person_score, place_score, query,
    )

    if person_score == 0 and place_score == 0:
        return "both"
    if person_score > 0 and place_score == 0:
        return "person"
    if place_score > 0 and person_score == 0:
        return "place"

    # Both sides have signal — winner needs at least 2:1 ratio
    ratio = max(person_score, place_score) / min(person_score, place_score)
    if ratio >= 2.0:
        return "person" if person_score > place_score else "place"
    return "both"
