@llama.cpp
@embeddings
Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model url https://huggingface.co/ggml-org/models/resolve/main/bert-bge-small/ggml-model-f16.gguf
    And   a model file bert-bge-small.gguf
    And   a model alias bert-bge-small
    And   42 as server seed
    And   2 slots
    And   1024 as batch size
    And   1024 as ubatch size
    And   2048 KV cache size
    And   embeddings extraction
    Then  the server is starting
    Then  the server is healthy

  Scenario: Embedding
    When embeddings are computed for:
    """
    What is the capital of Bulgaria ?
    """
    Then embeddings are generated

  Scenario: Tokenize / Detokenize complex
    When tokenizing:
    """
    北京的清晨，空氣清新而寧靜，一个年轻的旅行者在长城上漫步，他从自己的故乡—서울에서 출발하여 아시아의 다양한 문화를 탐험하고자 하는 꿈을 품고 떠났다。彼は日本の古都、京都を訪れ、そこで美しい桜の花が満開の下で古典音楽のコンサートに参加しました。祭りの夜、彼は色とりどりの灯籠が空に浮かぶのを見て、その美しさに感動しました。その後、彼は印度のバラナシに到着し、गंगा की घाटों पर आध्यात्मिक शांति की खोज में जुट गया। वहाँ उसने दिवाली के उत्सव में हिस्सा लिया, जहां लाखों दीये जलाकर समृद्धि और खुशहाली की कामना की गई थी।この旅は彼にとって非常に啓発的であり、多くの異なる文化から新しいことを学び、新しい友達を作る機会を与えました。彼はこの経験を通じて、 異なる文化の間の共通点と相違点を理解するようになりました。España is your's mine's l'heure èspciâl café über naïve résumé cañón élite cañas Barça 例子 東京 こんにちは 你好 中国
    """
    Then tokens can be detokenize and is equivalent False

  Scenario: OAI Embeddings compatibility
    Given a model bert-bge-small
    When an OAI compatible embeddings computation request for:
    """
    What is the capital of Spain ?
    """
    Then embeddings are generated

  Scenario: OAI Embeddings compatibility with multiple inputs
    Given a model bert-bge-small
    Given a prompt:
      """
      In which country Paris is located ?
      """
    And a prompt:
      """
      Is Madrid the capital of Spain ?
      """
    When an OAI compatible embeddings computation request for multiple inputs
    Then embeddings are generated

  Scenario: Multi users embeddings
    Given a prompt:
      """
      Write a very long story about AI.
      """
    And a prompt:
      """
      Write another very long music lyrics.
      """
    And a prompt:
      """
      Write a very long poem.
      """
    And a prompt:
      """
      Write a very long joke.
      """
    Given concurrent embedding requests
    Then the server is busy
    Then the server is idle
    Then all embeddings are generated

  Scenario: Multi users OAI compatibility embeddings
    Given a prompt:
      """
      In which country Paris is located ?
      """
    And a prompt:
      """
      Is Madrid the capital of Spain ?
      """
    And a prompt:
      """
      What is the biggest US city ?
      """
    And a prompt:
      """
      What is the capital of Bulgaria ?
      """
    And   a model bert-bge-small
    Given concurrent OAI embedding requests
    Then the server is busy
    Then the server is idle
    Then all embeddings are generated

  Scenario: All embeddings should be the same
    Given 10 fixed prompts
    And   a model bert-bge-small
    Given concurrent OAI embedding requests
    Then all embeddings are the same
