"""Search-behaviour metrics computed on deep research agent trajectories (tool-call logs).

Modules (each also runs as ``python -m trajectory_analysis.<module>``):
    query_drift          1 - embedding similarity of the first and last search query
    transition_category  Specialization / Generalization / Exploration / Repetition per consecutive search pair
    query_provenance     where query tokens come from: user question, prior queries, prior retrieved content, untraced
    document_novelty     share of returned documents not seen earlier in the session
    context_adoption     share of a query's new words taken from previously retrieved content
"""
