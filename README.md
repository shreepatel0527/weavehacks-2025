# weavehacks-2025
The public repo for WeaveHacks 2025, focused on agentic workflows for research scientists

## Vision 

Scientists are tackling some of the world’s most pressing problems; however, their industry remains highly manual, slowing down these brilliant minds in achieving the breakthroughs we need. We are introducing agentic workflows to the research space, with a focus on wet lab scientists, to reduce the burden on these brilliant minds.

## Use Case
Nanoparticles are a hot topic right now; we are exploring applications in energy, cosmetics, robotic technologies, and cancer treatments. Nanoparticles are unique because they allow us a gateway to understand the connection between classical and quantum states of being. As a nanoparticle researcher, I am working on building a nanoparticle that I believe will be a great carrier for targeted cancer therapy. 

This involves up to 16 hours of experiments, the first 3 of which I have to do by hand. All of my work occurs in the fume hood, and I have to wear special gloves and wrist covers to conduct my work because of the nature of the chemicals I am working with. 

Thus, my hands are tied and it is a huge disruption when I need to perform calculations, turn on and off instruments, and monitor for safety conditions. I need to record the mass of my gold compound (I’ll find a name for this I forget), the mass of my sulfur compound (also will find name), and the volume of my solvent (dichloromethane) to do my computations later. 

When I perform an experiment, I want to be able to scan my parameters effectively, record one experiment on one page, and have the agent record both my numbers and my qualitative results. I want to be able to visualize my parameters (volume, heat, etc.) and compare how these parameters relate to my UV-Vis peak (extracted from a chromatogram, determines whether or not I was successful in making my nanoparticle, or how close I am to the right structure)

I would want the agent to: turn on / off lab instruments (centrifuge, UV-Vis, etc.)-- let the agent figure it out -- also implement safety checks and controls on those instruments for shut off, inventory management (notie I'm running out of chemical, tell agent to find it and order it for me), agentic video feed monitoring to automate data collection for overinght experiments, agent communicates to scientist

## List of Tasks
1. Grab HAuCl4 * H2O, TOAB, Nanopure water, toluene
2. Weigh HAuCl₄·3H₂O (0.1576g) -- record mass 
3. Measure water -- record vol 
4. Dissolve 0.1576 g HAuCl₄·3H₂O in 5 mL Nanopure water -- qual obs
5. Weigh TOAB (~0.25g) -- record mass 
6. Measure toluene -- record vol 
7. Dissolve \~0.25 g TOAB in 10 mL toluene -- qual obs
8. Combine both in a 25 mL tri-neck round-bottom flask. -- qual obs
9. Move to fume hood
10. Plce on stir plate with stir bar
11. Stir vigorously (\~1100 rpm) for \~15 min. -- qual obv at the end of 15 min -- safety monitoring
12. Remove the aqueous layer with a 10 mL syringe.
13. Place gasket and gas needle on round-bottom flask
14. Purge with N₂
15. Cool to 0°C in ice bath over 30 min with stirring.
16. Grab PhCH₂CH₂SH
17. Calculate amount of PhCH₂CH₂SH -- calculation
18. Weight PhCH₂CH₂SH -- record mass 
19. Add 3 eq. PhCH₂CH₂SH (relative to gold). 
20. Stir slowly; observe color change: deep red → faint yellow (\~5 min) → clear (\~1 hr). -- qual obs -- safety monitoring
21. Grab NaBH₄
22. Increase stir speed
23. Calculate amount of NaBH₄ -- calculation
24. Weight NaBH₄ 
25. Add 10 eq. NaBH₄ in 7 mL ice-cold Nanopure water. - qual obs
26. Stir overnight under N₂ atmosphere. -- safety monitoring
27. Remove aqueous layer next day with syringe. -- qual obs
28. Dry toluene layer.
29. Add 20 mL ethanol to separate Au₂₅ clusters. -- record vol
30. Remove supernatant and collect Au₂₅ clusters.


| **Category / Prize**                       | **Relevant Tasks**                                                            | **Suggested Tools / APIs**                                         | **Why It Helps**                                                                                   |
| ------------------------------------------ | ----------------------------------------------------------------------------- | ------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------- |
| 🧠 **Agent Protocols (MCP, A2A)**          | All agent transitions (grab reagents → weigh → record → move → observe, etc.) | MCP, A2A                                                           | Required for eligibility. Use to show explicit agent handoffs and action chains.                   |
| 🟠 **Best Use of Weave**                   | All **Data Collection Agents**, **Safety Agent**, **Video Monitor Agent**     | `@trace`, `weave.init()`, media logging (audio, video, params)     | Enables observability, rich logs, and debugging. Show full trace of agent workflows.               |
| ☁️ **Best Use of Google Tools**            | Vision agent, projection agent, orchestration logic                           | A2A, ADK, Vertex AI (e.g. Gemini 1.5 Pro for multimodal reasoning) | Showcase stateful agents and deploy on Google infra. Use Gemini for image/text interpretation.     |
| 🤖 **Best Use of CrewAI**                  | Data Collection Agent, Lab Control Agent, Safety Agent, Projection Agent      | CrewAI (agent roles + task orchestration)                          | Define clear agent personas and schedule task handoffs with dependencies.                          |
| 🌐 **Best Use of BrowserBase / Stagehand** | Remote lab control, data scraping, reagent lookup                             | Stagehand (code gen), BrowserBase (automation platform)            | Show real-world automation of browser interfaces (e.g., ELNs or reagent supplier sites).           |
| 🚀 **Best Use of Fly.io**                  | Long-running agents (e.g., video monitor), modular agent deployment           | Fly.io container deployment, scale-to-zero microservices           | Host distributed agents; separate background jobs like monitoring from active control agents.      |
| 🧠 **Best Use of Exa**                     | Experiment Projection Agent: "should I continue?" based on observed changes   | Exa Search API (for protocols, research papers, observed outcomes) | Let agent search research context and reason about next steps using real-world synthesis examples. |


## Agents 

### Data Collection Agent (Rohit)

Problem Statement: Lab scientists spend a majority of their time working with their hands. When occupied by other tasks, it becomes difficult to record data in a timely manner. If recording data after an experiment ends, a scientist runs the risk of forgetting their observations or not recording them in their entirety. Not to mention, leaving the experiment to record data involves taking off layers of PPE and could introduce safety hazards if experiments are not monitored adequately in that time. 

Solution: Our solution enables scientists to automate data collection through voice into their electronic lab notebook, ensuring that they are able to focus on their experiments.

Build: (include the tools we used, approach, etc.)

Prompt: We are building a suite that supports wet lab scientists in their day to day work by automationg data collection, calculation, and safety monitoring tasks. You are an AI agent that specializes in recording chemical data, supporting wet lab scientists at the bench. You are given a protocol and a data table 

When the user specifies a compound, find the relevant reagent in the table. Determine if the user description allows you to pick one reagent. If it does, then replace the value attached to that reagent with the one that the user specified. 

If the user description is unclear, prompt the user with a follow-up question in the following format: "I am unsure which reagent you mean. Do you mean {option 1} or {option 2}?". Based on their response, determine which attribute to modify. 

Record the user format. 

Examples:
"The mass of the gold compound is 0.1598g" --> find the gold compound in the table, clarify if needed, and modify the value. 

Inputs:
Protocol: {protocol}
Data table: {table}

### Lab Instrument Control Agent

Problem Statement: My hands are often tied with the pipettes, etc. that I am holding. 

Solution: Being able to automate turning an instrument on or off by voice or with another agent 

### Safety Monitoring Agent (Andy)

Problem Statement: When experiments run for a long amount of time, scientists leave the lab and let the experiments go on their own. If they become dangerous, they become a safety concern. 

Solution: Notify the scientist. If we cannot reach the scientist, shut down the system. 

Features: monitor (random) input stream of four parameters: temperature, pressure, nitrogen level, butane level

Prompt: We are building a suite that supports wet lab scientists in their day to day work by automationg data collection, calculation, and safety monitoring tasks. You are an AI agent that specializes in monitoring instrument I/O streams tracking experiment parameters like temperature, pressure, gas concentration, etcl. You are responsible for determining if the experiment is at risk of causing an incident. You are given a protocol as input

Based on the protocol, determine what suitable ranges are for the temperature and pressure of the instrument. 

If the I/O stream for a parameter indicates that the instrument is nearing a safety concern for at least 1 minute, contact the scientist. 

If the scientist does not reply in three minutes, and the parameter has returned to non-concerning levels, record a warning, but do not take further action. 

If the scientist does not reply in three minutes, and the parameter has stayed at the same level, or increased in potential safety severity, shut off the instrument. 

Input:
Protocol: {protocol}

### Data Visualization Agent / Computation Agent (Michael)
Prompt: We are building a suite that supports wet lab scientists in their day to day work by automationg data collection, calculation, and safety monitoring tasks. You are a multi-purpose AI agent that performs computations on lab data, performs literature search, and completes other administrative tasks. You are given a protocol as context and a library of python scripts to connect to external services via API.

When asked to perform a computation, read the data table in streamlit. Extract the necessary values to perform the computation

Examples:
"Calculate the percent yield of my experiment based on my limiting reagent" --> determine the amount of reagents used from the data table, search the protocol for molar equivalent information to help perform a limiting reagent computation, perform the computation and identify the limiting reagent, determine the theoretical yield from the limiting reagent, compare with the actual yield, determine the final percent yield and provide your reasoning. 
"Determine how much sulfur compound I need based on the amount of gold I have" --> determine the amount of gold used from the data table, search the protocol for molar equivalent information to help perform that conversion, perform the computation, determine the final value and provide your reasoning. 
"I forgot what it is called when a nanoparticle is used to direct radiation in oncology. Can you help me?" --> perform a search on the user's behalf to identify the vocabulary term, report back to the user 

Context:
Protocol: {protocol}


# weavehacks-2025
The public repo for WeaveHacks 2025, focused on agentic workflows for research scientists



Once Shree picks a use case / API, I will define the API call(s) for you to make:
https://benchling.com/api/reference
https://docs.latch.bio/api/latch.html
https://www.uncountable.com/cloud-platform
https://www.scinote.net/product/integrations-and-api
https://www.collaborativedrug.com/cdd-blog/vault-snack-17-all-things-api
https://www.dropbox.com/scl/fi/vb2qgs8o2bu6d3lzp63mg/ELN-analytics-template.xlsx
https://scinote-eln.github.io/scinote-api-docs/#introduction


High level AI scientist and computational biology notes:
https://www.futurehouse.org/about
https://www.linkedin.com/posts/james-zou-2123a4133_agents4science-activity-7349456876696199170-BSKR
https://agents4science.stanford.edu

