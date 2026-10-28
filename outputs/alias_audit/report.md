# Alias audit: false failures a Wikidata alias table would fix (report only)

Rows audited: failure labels from `unmatched_gold`, gold not a date, hop text not a hedge. Gold entities looked up: 1405 (successful lookups 1405, errors 0); entities with at least one alias: 890.

## full10

* audited failure rows: 1167
* **proposed flips (alias found in hop text): 0**

Proposed flips (review by hand):


## tfidf

* audited failure rows: 2053
* **proposed flips (alias found in hop text): 2**

Proposed flips (review by hand):

- hop 2 | gold `John Philip II, Wild- and Rhinegrave of Salm-Dhaun` ~ alias `John Philip II of Salm-Dhaun` | From the context, I find that John Philip II of Salm-Dhaun was the son of Count John Louis of Salm-Dhaun and his wife, Elisabeth o
- hop 1 | gold `William II of Villehardouin` ~ alias `William of Villehardouin` | From the context, I find that Isabella of Villehardouin's father is William of Villehardouin.
