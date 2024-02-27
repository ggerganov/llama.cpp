# List of ongoing issues
@bug
Feature: Issues llama.cpp server
  # No confirmed issue at the moment
      Background: Server startup
          Given a server listening on localhost:8080
          And   a model with n_embed=4096
          And   n_ctx=32768
          And   8 slots
          And   embeddings extraction
          Then  the server is starting
          Then  the server is healthy
          
      Scenario: Embedding
          When 8 identical inputs (1000 tokens) are computed simultaneously.
          Then embeddings are generated, but they are different
