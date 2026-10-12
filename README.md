# README

## [Question Q1](Q1.md)

## [Question Q2](Q2.md)

## Question Q3

**The definition of PASS is when the agent recognises the intent correctly (probabilistic checked with an LLM) and goes to the next step (question) correctly (determinsitic)**

The script that evaluates the test runs: check_conversation.py. To run:  

1. Set a key an Anthropic key in a .env file in the same dir. 

1. pip install -r requirements.txt
2. python check_conversation.py

Example:


python check_conversation.py logs/gym_agent_conversations.json --threads thread_02

thread_02  PASS
  ✓ Q1 Intent: caller "Yes, please, I definitely need help with that." -> yes
      expected: ask Q2 Active Medi-Cal | agent: ask Q2 Active Medi-Cal | ok
  ✓ Q2 Active Medi-Cal: caller "No, I don't think it's active right now, which is why I’m calling." -> inactive
      expected: ask Q3 Other Coverage | agent: ask Q3 Other Coverage | ok
  ✓ Q3 Other Coverage: caller "No, I don't have any other insurance at all." -> false
      expected: ask Q4 Residency | agent: ask Q4 Residency | ok
  ✓ Q4 Residency: caller "Yeah, I still live here in San Diego. I really need help getting my..." -> true
      expected: ask Q5 Yellow Packet | agent: ask Q5 Yellow Packet | ok
  ✓ Q5 Yellow Packet: caller "No, I never actually got it in the mail, so I don't have it. Can we..." -> no_or_lost
      expected: _complete (start renewal) | agent: _complete (start renewal) | ok
  note: 1 caller turn(s) after the flow ended were not checked

 