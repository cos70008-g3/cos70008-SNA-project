"""HTTP API layer for the conceptual network.

The :mod:`src.api.server` module exposes the read-only operations of
:class:`src.extensions.neo4j_reader.Neo4jGraphReader` over HTTP, so the
network can be browsed from a browser, notebook, or any HTTP client
without writing Cypher or Python.

Run locally with::

    uvicorn src.api.server:app --reload

Then open http://localhost:8000/docs for the interactive Swagger UI.
"""
