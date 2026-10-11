"""Real datasets for research question 1 (ETTh1 + an ECL subset).

``REALDATA_VERSION`` versions the processed real-data artefacts: any change
that alters a processed series, the subset or a split boundary must bump it;
the golden tests enforce this. Runs pin the real data through the split
manifest's ``realdata_version`` field plus ``run_start.git_commit``; nothing
in this task writes ledger records or changes ``PROTOCOL_KEYS``.
"""

REALDATA_VERSION = "1"
