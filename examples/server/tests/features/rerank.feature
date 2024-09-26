@llama.cpp
@rerank
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/models/resolve/main/jina-reranker-v1-tiny-en/ggml-model-f16.gguf
    And   a model file jina-reranker-v1-tiny-en.gguf
    And   a model alias jina-reranker-v1-tiny-en
    And   42 as server seed
    And   2 slots
    And   512 as batch size
    And   512 as ubatch size
    And   512 KV cache size
    And   embeddings extraction
    Then  the server is starting
    Then  the server is healthy

  Scenario: Rerank
    Given a rerank query:
      """
      Organic skincare products for sensitive skin
      """
    And   a rerank document:
      """
      Organic skincare for sensitive skin with aloe vera and chamomile: Imagine the soothing embrace of nature with our organic skincare range, crafted specifically for sensitive skin. Infused with the calming properties of aloe vera and chamomile, each product provides gentle nourishment and protection. Say goodbye to irritation and hello to a glowing, healthy complexion.
      """
    And   a rerank document:
      """
      New makeup trends focus on bold colors and innovative techniques: Step into the world of cutting-edge beauty with this seasons makeup trends. Bold, vibrant colors and groundbreaking techniques are redefining the art of makeup. From neon eyeliners to holographic highlighters, unleash your creativity and make a statement with every look.
      """
    And   a rerank document:
      """
      Las nuevas tendencias de maquillaje se centran en colores vivos y técnicas innovadoras: Entra en el fascinante mundo del maquillaje con las tendencias más actuales. Colores vivos y técnicas innovadoras están revolucionando el arte del maquillaje. Desde delineadores neón hasta iluminadores holográficos, desata tu creatividad y destaca en cada look.
      """
    And   a rerank document:
      """
      新的化妆趋势注重鲜艳的颜色和创新的技巧：进入化妆艺术的新纪元，本季的化妆趋势以大胆的颜色和创新的技巧为主。无论是霓虹眼线还是全息高光，每一款妆容都能让您脱颖而出，展现独特魅力。
      """
    When  reranking request
    Then  reranking results are returned
    Then  reranking highest score is index 2
