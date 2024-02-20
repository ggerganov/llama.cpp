Feature: llama.cpp server

  Background: The server is started and ready to accept prompts
    When wait for the server to be started
    Then wait for the server to be healthy

  Scenario: Health endpoint
    Given an health liveness probe
    Then the server must be healthy

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
      | model       | system_prompt               | user_prompt                         |
      | tinyllama-2 | You are ChatGPT.            | Say hello                           |
      | tinyllama-2 | You are a coding assistant. | Write the fibonacci function in c++ |


  Scenario: Health endpoint during processing with concurrent requests
    Given 2 slow concurrent prompts
    Then wait for all slots processing
    Then the server is overloaded
    When wait for all slots idle
    Then all prompts must be predicted