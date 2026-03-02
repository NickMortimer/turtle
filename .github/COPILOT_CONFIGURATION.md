# GitHub Copilot Configuration Guide

This document explains the GitHub Copilot configuration in this repository and how to use it effectively.

## Overview

Albatri uses GitHub Copilot's custom instructions, agents, and prompts to ensure AI-generated code follows project-specific guidelines, safety requirements, and architectural decisions.

## Configuration Structure

```
.github/
├── copilot-instructions.md       # Repository-wide instructions
├── instructions/                  # Path-specific instructions
│   ├── python.instructions.md
│   ├── agents.instructions.md
│   ├── prompt.instructions.md
│   └── instructions.instructions.md
├── agents/                        # Custom agent definitions
│   ├── Thinking-Beast-Mode.agent.md
│   └── blueprint-mode.agent.md
└── prompts/                       # Reusable prompt files
    ├── structured-autonomy-plan.prompt.md
    ├── structured-autonomy-generate.prompt.md
    └── structured-autonomy-implement.prompt.md
```

## Core Configuration Files

### copilot-instructions.md

This is the **primary configuration file** for the repository. It contains:
- Safety-critical guidelines for UAV mission planning
- Architecture decisions and constraints
- Module responsibilities and structure
- Coding standards and testing requirements
- Common pitfalls to avoid
- Quick reference for common operations
- Development workflow

**When to update**: When architectural decisions change, new modules are added, or project-wide standards are updated.

### Instruction Files

Located in `.github/instructions/`, these provide **scoped guidance** for specific file types or areas:

- **python.instructions.md** (`**/*.py`): Python coding conventions, PEP 8 compliance, type hints, docstrings
- **agents.instructions.md** (`**/*.agent.md`): Guidelines for creating custom agent files
- **prompt.instructions.md** (`**/*.prompt.md`): Guidelines for creating prompt files
- **instructions.instructions.md** (`**/*.instructions.md`): Meta-guidelines for instruction files

**When to use**: Add new instruction files when you need specific guidance for:
- New programming languages or frameworks
- Specific tool configurations
- Domain-specific code areas (e.g., testing, documentation)

### Agent Files

Located in `.github/agents/`, these define **specialized AI personas** for specific tasks:

- **Thinking-Beast-Mode.agent.md**: For complex problems requiring deep research, multi-perspective analysis, and thorough validation
- **blueprint-mode.agent.md**: For structured workflows with strict validation (Debug, Express, Main, Loop modes)

**When to use**: Select specific agents in Copilot Chat for tasks that match their expertise.

### Prompt Files

Located in `.github/prompts/`, these provide **reusable workflows** for common tasks:

- **structured-autonomy-plan.prompt.md**: Strategic planning and requirement analysis
- **structured-autonomy-generate.prompt.md**: Code generation workflows
- **structured-autonomy-implement.prompt.md**: Implementation and validation workflows

**When to use**: Run prompts via `/` commands in Copilot Chat for standardized workflows.

## Using Copilot with Albatri

### For General Development

When working in Python files, Copilot automatically applies:
1. Repository-wide instructions from `copilot-instructions.md`
2. Python-specific instructions from `python.instructions.md`

This ensures generated code follows:
- Safety-first mindset (no default assumptions for safety parameters)
- PEP 8 style guide (79 char lines, type hints, PEP 257 docstrings)
- Project architecture (dataclasses, explicit validation, pyproj for geodetics)
- Testing requirements (pytest, ISTQB techniques, ≥90% coverage)

### For Complex Tasks

Use **@Thinking-Beast-Mode** when you need:
- Deep investigation of unfamiliar code
- Multi-step problem solving with research
- Adversarial validation of solutions
- Cross-domain synthesis and pattern recognition

Example: `@Thinking-Beast-Mode Investigate why geodetic calculations fail near the poles`

### For Structured Workflows

Use **@blueprint-mode** when you need:
- Strict validation and correctness checking
- Structured debugging workflows
- Repetitive tasks across multiple files
- Express changes with minimal overhead

Example: `@blueprint-mode Debug the mission validation failure for coastal_survey.yaml`

### For Standardized Tasks

Use prompt files via `/` commands:
- `/structured-autonomy-plan` - Analyze requirements and create implementation plan
- `/structured-autonomy-generate` - Generate code following project standards
- `/structured-autonomy-implement` - Implement and validate with testing

## Best Practices

### When Adding Features

1. **Check ADRs first**: Consult `docs/ARCHITECTURE_DECISIONS.md` before choosing approaches
2. **Follow safety guidelines**: Never infer defaults for safety-critical parameters
3. **Maintain test coverage**: Write tests before or alongside implementation (TDD preferred)
4. **Use explicit validation**: Validate at YAML boundary, use explicit `is None` checks

### When Fixing Bugs

1. **Use Debug workflow**: Let `@blueprint-mode` guide you through structured debugging
2. **Reproduce first**: Create a test that demonstrates the bug
3. **Fix root cause**: Don't just address symptoms
4. **Add edge case tests**: Ensure the bug can't regress

### When Refactoring

1. **Maintain safety invariants**: Refactoring must not weaken validation
2. **Keep tests passing**: Run tests frequently during refactoring
3. **Update documentation**: Keep ADRs, docstrings, and comments current
4. **Use tools**: Rely on linters, type checkers, and formatters

## Maintaining Configuration

### Reviewing Instructions

Periodically review configuration files to ensure they remain:
- **Accurate**: Reflect current project state and decisions
- **Complete**: Cover all important patterns and requirements
- **Consistent**: No conflicts between different instruction files
- **Concise**: No unnecessary duplication or verbosity

### Adding New Agents

When creating custom agents:
1. Use template from `agents.instructions.md`
2. Define clear scope and boundaries
3. Specify required tools and permissions
4. Document in `AGENTS.md`
5. Test with representative tasks

### Adding New Prompts

When creating prompt files:
1. Use template from `prompt.instructions.md`
2. Define clear inputs, workflow, and outputs
3. Include validation steps
4. Document usage in this guide

## Troubleshooting

### Copilot Not Following Guidelines

- **Check scope**: Ensure instruction file `applyTo` pattern matches the files you're editing
- **Check order**: More specific instructions take precedence over general ones
- **Review conflicts**: Look for contradictory guidance in different files
- **Be explicit**: Make requirements clear and unambiguous in natural language

### Agent Not Available

- **Verify frontmatter**: Ensure agent file has valid YAML frontmatter
- **Check filename**: Use kebab-case with `.agent.md` extension
- **Restart IDE**: Reload VS Code or refresh GitHub.com after changes
- **Check syntax**: Validate YAML syntax in frontmatter

### Instructions Not Applied

- **Check path**: Verify file location (`.github/instructions/` or `.github/`)
- **Verify extension**: Use `.instructions.md` for instruction files
- **Validate frontmatter**: Ensure `applyTo` glob pattern is correct
- **Clear cache**: Restart IDE or refresh browser

## Recommended Workflows

### Structured Autonomy Workflow

The repository includes structured autonomy prompts for systematic development:

1. **Prepare context**
	- Attach relevant files using #file or #codebase references
	- Include logs or terminal output with #terminalSelection if relevant

2. **Plan**
	- Run `/sa-plan` with a clear feature description and constraints
	- Review `plans/{feature-name}/plan.md` and edit if needed

3. **(Optional) Generate step-by-step implementation**
	- Run `/sa-generate` to create `plans/{feature-name}/implementation.md`

4. **Implement**
	- Run `/sa-implement` and provide the plan file in context
	- Follow STOP & COMMIT checkpoints exactly

5. **Validate**
	- Run tests or checks specified in the plan
	- Capture results in the plan file or notes as required

### How to Use Agents

**Best practices for agent invocation:**
- Pick the agent from the VS Code agent selector, or start a chat message with `@agent-name`
- Provide concrete context (files, error logs, expected behavior) to reduce back-and-forth
- Example: `@blueprint-mode Debug the mission validation failure for coastal_survey.yaml`

### How to Run Prompts

**Best practices for prompt execution:**
- Use `/prompt-name` in chat (e.g., `/sa-plan`)
- Or run "Chat: Run Prompt" from the Command Palette
- Or open the prompt file and click Run

## Resources

### Internal Documentation
- [Safety Invariants](../docs/SAFETY_INVARIANTS.md)
- [Architecture Decisions](../docs/ARCHITECTURE_DECISIONS.md)
- [Agent Index](../AGENTS.md)

### GitHub Copilot Documentation
- [Adding Repository Instructions](https://docs.github.com/en/copilot/how-tos/configure-custom-instructions/add-repository-instructions)
- [Creating Custom Agents](https://docs.github.com/en/copilot/how-tos/use-copilot-agents/coding-agent/create-custom-agents)
- [Using Prompt Files](https://code.visualstudio.com/docs/copilot/customization/prompt-files)

### Best Practices
- [How to Write Great Instructions](https://github.blog/ai-and-ml/github-copilot/how-to-write-a-great-agents-md-lessons-from-over-2500-repositories/)
- [Setting Up Copilot Coding Agent](https://github.blog/ai-and-ml/github-copilot/onboarding-your-ai-peer-programmer-setting-up-github-copilot-coding-agent-for-success/)

### Reference Documentation
- [Awesome Copilot: Prompts](https://github.com/github/awesome-copilot/blob/main/docs/README.prompts.md)
- [Awesome Copilot: Agents](https://github.com/github/awesome-copilot/blob/main/docs/README.agents.md)
- [VS Code Copilot Chat](https://code.visualstudio.com/docs/copilot/chat/copilot-chat)

## Contributing

When contributing to Copilot configuration:
1. Test changes with representative code
2. Document updates in commit messages
3. Review impact on existing prompts/agents
4. Update this guide if structure changes
5. Keep [AGENTS.md](../AGENTS.md) synchronized with actual agent files

---

**Last Updated**: 2026-01-30
**Maintainers**: Albatri Development Team
**Source**: [github/awesome-copilot](https://github.com/github/awesome-copilot)
