Feature: Completion request

  Scenario Outline: run a completion request
      Given a prompt <prompt>
      When we request a completion
      Then tokens are predicted

    Examples: Prompts
      | prompt                                                         |
      | I believe the meaning of life is                               |
      | Write a detailed analogy between mathematics and a lighthouse. |