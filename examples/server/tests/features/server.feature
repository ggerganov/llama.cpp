Feature: llama.cpp server

  Background: Server startup
    Given a server listening on localhost:8080 with 2 slots
    Then  the server is starting
    Then  the server is healthy

  Scenario: Health
    When the server is healthy
    Then the server is ready

  Scenario Outline: Completion
    Given a <prompt> completion request with maximum <n_predict> tokens
    Then  <predicted_n> tokens are predicted

    Examples: Prompts
      | prompt                           | n_predict | predicted_n |
      | I believe the meaning of life is | 128       | 128         |
      | Write a joke about AI            | 512       | 512         |

  Scenario Outline: OAI Compatibility
    Given a system prompt <system_prompt>
    And   a user prompt <user_prompt>
    And   a model <model>
    And   <max_tokens> max tokens to predict
    And   streaming is <enable_streaming>
    Given an OAI compatible chat completions request
    Then  <predicted_n> tokens are predicted

    Examples: Prompts
      | model        | system_prompt               | user_prompt                          | max_tokens | enable_streaming | predicted_n |
      | llama-2      | You are ChatGPT.            | Say hello.                           | 64         | false            | 64          |
      | codellama70b | You are a coding assistant. | Write the fibonacci function in c++. | 512        | true             | 512         |

  Scenario: Multi users
    Given a prompt:
      """
      Write a formal complaint email to Air France about my delayed
      baggage from my flight on Tuesday, January 17th, from Paris to Toulouse. Be verbose.
      """
    And a prompt:
      """
      Translate the following War & Peace chapter into Russian: WELL, PRINCE,
      Genoa and Lucca are now no more than private estates of the Bonaparte
      family. No, I warn you, that if you do not tell me we are at war,
      if you again allow yourself to palliate all the infamies and atrocities
      of this Antichrist (upon my word, I believe he is), I don’t know you
      in future, you are no longer my friend, no longer my faithful slave,
      as you say. There, how do you do, how do you do? I see I’m scaring you,
      sit down and talk to me.” These words were uttered in July 1805 by
      Anna Pavlovna Scherer, a distinguished lady of the court,
      and confidential maid-of-honour to the Empress Marya Fyodorovna.
      It was her greeting to Prince Vassily, a man high in rank
      and office, who was the first to arrive at her soirée.
      """
    Given concurrent completion requests
    Then the server is busy
    Then the server is idle
    Then all prompts are predicted


  Scenario: Multi users OAI Compatibility
    Given a system prompt "You are an AI assistant."
    And a model tinyllama-2
    And 1024 max tokens to predict
    And streaming is enabled
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
      Write yet another very long music lyrics.
      """
    Given concurrent OAI completions requests
    Then the server is busy
    Then the server is idle
    Then all prompts are predicted