# Tool calling implementation with a decision model

Demo code for LBKE's article [« Implémenter le tool calling avec le modèle de décision Jev par TypeSafe
 »](https://www.lbke.fr/formations/ressources/ia/jev-sytem-one-tool-calling)

Tool selection has to be uncoupled from tool arguments generation, which is still done via a small LLM.

The "shovel_demo" shows a tiny example and comparison with a direct LLM use.
The "jev_tool_calling" demo generalizes it with parallel tool calling.

Next step:
- running evaluations
- using Jev to select bounded arguments (boolean tools arguments for instance)
- implementing a full-fledged agent loop (via LangChain)