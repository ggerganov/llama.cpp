Feature: llama.cpp server

  Scenario Outline: run a completion request
    Given a prompt <prompt>
    When we request a completion
    Then tokens are predicted

    Examples: Prompts
      | prompt       |
      | I believe    |
      | Write a joke |

  Scenario Outline: run a completion on the OAI endpoint
    Given a system prompt <system_prompt>
    And a user prompt <user_prompt>
    And a model <model>
    When we request the oai completions endpoint
    Then the oai response contains completion tokens

    Examples: Prompts
      | model          | system_prompt                | user_prompt                            |
      | tinyllama-2    | You are ChatGPT.             | Say hello                              |
      | tinyllama-2    | You are a coding assistant.  | Write the fibonacci function in c++    |