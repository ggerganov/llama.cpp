# run with: ./tests.sh --no-skipped --tags truncation
@truncation
@slow
Feature: Chat truncation

  Background: Server startup
    Given a server listening on localhost:8080
    And   a model file mistral-7b-v0.2-iq3_s-imat.gguf from HF repo ggml-org/models
    And   prompt caching is enabled
    And   a list of stop strings ["\n"]
    And   82 tokens to keep
    And   256 KV cache size
    And   32 server max tokens to predict
    Then  the server is starting
    Then  the server is healthy

  Scenario: Correctly truncate the prompt when the prompt exceeds the context size
    Given a prompt:
    """
    Continue the chat below.
    Me: Hey there, how's it going?
    You: I'm doing well, thanks for asking! How are you?
    Me: I'm doing good, just trying to get some work done. How's your day?
    You: My day has been pretty productive so far. I've been working on a new project.
    Me: That's great to hear! What's the new project you're working on?
    You: It's a web application that's designed to help people manage their personal finances. I'm really excited about it.
    Me: That sounds really useful, I'd be interested to hear more about it. Do you have a timeframe for when you expect to have it ready to launch?
    You: I'm aiming to have the initial version ready within the next few months. I want to ensure it's robust before launching it.
    Me: That's really nice, are you happy with the progress so far?

    """
    And   an ongoing completion request
    Then  -1 tokens are predicted matching You:
    Given an ongoing prompt:
    """

    Me: I have one more question for you my friend. What's the most value thing you learned during your development journey?

    """
    And   52 tokens to truncate
    And   a completion request with no api error
    Then  -1 tokens are predicted matching You:
    # 28 because '\n' stop string is not pushed to the context
    And   28 prompt tokens are processed
