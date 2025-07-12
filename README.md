# weavehacks-2025
The public repo for WeaveHacks 2025, focused on agentic workflows for research scientists

## Vision 

Scientists are tackling some of the world’s most pressing problems; however, their industry remains highly manual, slowing down these brilliant minds in achieving the breakthroughs we need. We are introducing agentic workflows to the research space, with a focus on wet lab scientists, to reduce the burden on these brilliant minds.

## Use Case
Nanoparticles are a hot topic right now; we are exploring applications in energy, cosmetics, robotic technologies, and cancer treatments. Nanoparticles are unique because they allow us a gateway to understand the connection between classical and quantum states of being. As a nanoparticle researcher, I am working on building a nanoparticle that I believe will be a great carrier for targeted cancer therapy. 

This involves up to 16 hours of experiments, the first 3 of which I have to do by hand. All of my work occurs in the fume hood, and I have to wear special gloves and wrist covers to conduct my work because of the nature of the chemicals I am working with. 

I need to record the mass of my gold compound (I’ll find a name for this I forget), the mass of my sulfur compound (also will find name), and the volume of my solvent (dichloromethane) to do my computations later. 

When I mix together my ingredients, I am looking for the formation of black particles at the bottom of my orange solution. I am concerned with how long it takes for these particles to form (currently, I am manually timing the process with a timer). I am also on the lookout for steam formation (heat, could be a concern)

I extract the black particles from the solution and mix them into a different solvent, which I need to record the volume of. I also am on the lookout for qualitative changes. 

I place this into an oil bath, which I need the temperature, compounds, and pressure of. 

When I perform an experiment, I want to be able to scan my parameters effectively, record one experiment on one page, and have the agent record both my numbers and my qualitative results. I want to be able to visualize my parameters (volume, heat, etc.) and compare how these parameters relate to my UV-Vis peak (extracted from a chromatogram, determines whether or not I was successful in making my nanoparticle, or how close I am to the right structure)

I would want the agent to: turn on / off lab instruments (centrifuge, UV-Vis, etc.)-- let the agent figure it out -- also implement safety checks and controls on those instruments for shut off, inventory management (notie I'm running out of chemical, tell agent to find it and order it for me), agentic video feed monitoring to automate data collection for overinght experiments, agent communicates to scientist

## List of Tasks
1. Grab reagents (gold, sulfur, solvents)
2. Weigh reagents (gold, sulfur)
3. Record volume (solvents) (Data Collection Agent: Voice to Text)
4. Record mass (Data Collection Agent: Voice to Text)
5. Move to fume hood 
6. Mix ingredients
7. Observe changes 
8. Record changes (Data Collection Agent: Voice to Text)
9. Determine, based on changes, whether to continue with experiment (OPTIONAL: Experiment Projection Agent -- Visualization / Computation)
10. Extract black particles
11. Mix with solvent
12. Observe changes 
13. Record changes (Data Collection Agent: Voice to Text)
14. Pre-heat oil bath (Lab Instrument Control Agent)
15. Move to oil bath
16. Record parameters (Data Collection Agent: Voice to Text)
17. Set up video feed to monitor experiment
18. Go home bc the oil bath will take 16 hours
19. Monitor video feed (AI TBD WHAT THIS LOOKS LIKE)
20. Record data from video feed -- qualitative changes (Data Collection Agent? TBD)
21. In the event of a safety issue, alert scientist (Safety Monitoring Agent)
22. In the event of a safety issue, turn off lab instruments (Lab Instrument Control Agent)

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

### Data Collection Agent

Problem Statement: Lab scientists spend a majority of their time working with their hands. When occupied by other tasks, it becomes difficult to record data in a timely manner. If recording data after an experiment ends, a scientist runs the risk of forgetting their observations or not recording them in their entirety. Not to mention, leaving the experiment to record data involves taking off layers of PPE and could introduce safety hazards if experiments are not monitored adequately in that time. 

Solution: Our solution enables scientists to automate data collection through voice into their electronic lab notebook, ensuring that they are able to focus on their experiments.

Build: (include the tools we used, approach, etc.)

### Lab Instrument Control Agent

Problem Statement: My hands are often tied with the pipettes, etc. that I am holding. 

Solution: Being able to automate turning an instrument on or off by voice or with another agent 

### Safety Monitoring Agent

Problem Statement: When experiments run for a long amount of time, scientists leave the lab and let the experiments go on their own. If they become dangerous, they become a safety concern. 

Solution: Notify the scientist. If we cannot reach the scientist, shut down the system. 

### Data Visualization Agent


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

