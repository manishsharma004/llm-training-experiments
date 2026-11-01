# Agent notes

## Subagents and model quota

Do not launch a subagent on an external model such as Claude or OpenAI. That includes model slugs from those families (Claude, GPT, and the same vendors under other names). Those calls spend the other models' quota.

Do the work in this session. If a subagent is actually required, leave its model unset so it inherits this session's model. Do not switch it to Claude or OpenAI to get a second opinion, a review, or a browser pass.
