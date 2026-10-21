# Real portfolio allocations (evaluation only)

`cabinets.csv` lists, for 195 majority coalition cabinets in 31 countries
(1988-2014), every party of the parliament with its seats and, for cabinet
parties, its share of the cabinet's partisan ministers. It is the processed
table of our team's companion analysis (same authors), built from:

* **Seki-Williams Government and Ministers Data** v2.0 (Harvard Dataverse,
  doi:10.7910/DVN/0UNUAM, CC0 1.0): ministers per party, PM party, government
  type, ParlGov cabinet id. Seki & Williams (2014), *Electoral Studies* 34.
* **ParlGov** (Döring & Manow, parlgov.org; releases CC0 1.0): the full
  parliament behind each cabinet.

Parties were linked by seats, names and ParlGov's own cabinet coding; weakly
or ambiguously linked cabinets were dropped (`cabinets_dropped.csv`, with the
reason). `clean_core` = 1 marks the 149 cabinets without flagged cases
(presidential or collegial executives, PM-majority cabinets, coding
disagreements). These data are never used for training.
