---
description: 'Guidelines for creating and maintaining instruction files'
applyTo: '**/*.instructions.md'
---

# Instruction File Guidelines

This file provides guidelines for creating and maintaining instruction files that guide GitHub Copilot's behavior in this repository.

## Purpose

Instruction files allow you to scope coding standards, conventions, and practices to specific file patterns. They help Copilot understand project-specific requirements and maintain consistency across the codebase.

## File Structure

### Frontmatter (Required)

Every instruction file must include YAML frontmatter with at least these fields:

```yaml
---
description: 'Brief description of what this instruction file covers'
applyTo: 'glob pattern to match files (e.g., **/*.py, src/**/*.ts)'
---
```

### Fields

- **description**: A concise description of the instruction file's purpose
- **applyTo**: Glob pattern(s) determining which files these instructions apply to
  - Use `**/*.ext` for all files with a specific extension
  - Use `path/**/*.ext` for files in a specific directory
  - Multiple patterns can be specified as a comma-separated list

### Content Structure

After the frontmatter, organize content with clear sections:

1. **Purpose/Overview**: Briefly explain what these instructions cover
2. **Conventions**: Specific coding standards or patterns to follow
3. **Examples**: Good and bad examples where helpful
4. **References**: Links to relevant documentation or standards

## Best Practices

### Keep Instructions Focused

- Each instruction file should cover a specific aspect (language, tool, framework)
- Avoid duplicating general project instructions that belong in `copilot-instructions.md`
- Be specific about what Copilot should and shouldn't do

### Use Clear Language

- Write in imperative mood ("Use", "Follow", "Avoid")
- Be explicit about requirements vs. recommendations
- Provide rationale for non-obvious rules

### Maintain Consistency

- Follow the same structure across all instruction files
- Use consistent terminology with other project documentation
- Cross-reference related instruction files when appropriate

### Keep Updated

- Review instruction files when project conventions change
- Update examples to reflect current best practices
- Remove outdated or superseded guidelines

## File Naming

Use the pattern `<topic>.instructions.md`:
- `python.instructions.md` for Python-specific guidelines
- `testing.instructions.md` for test-related conventions
- `agents.instructions.md` for agent file guidelines
- `prompt.instructions.md` for prompt file guidelines

## Integration with copilot-instructions.md

The root `copilot-instructions.md` file provides repository-wide guidance. Instruction files provide more specific, scoped guidance:

- **copilot-instructions.md**: Project-specific architecture, safety requirements, general patterns
- **Instruction files**: Language-specific, tool-specific, or path-specific conventions

When there's overlap, instruction files take precedence for their matched file patterns.

## Example Template

```markdown
---
description: 'Description of what this covers'
applyTo: 'file/pattern/**/*.ext'
---

# Title

## Purpose

Brief explanation of what these instructions cover.

## Conventions

### Convention Name

- Rule 1
- Rule 2

### Another Convention

- Rule A
- Rule B

## Examples

### Good Example

\`\`\`language
// Good code example
\`\`\`

### Bad Example

\`\`\`language
// Bad code example
\`\`\`

## References

- [Link to relevant documentation](url)
```

## Common Patterns

### Language-Specific Instructions

Focus on:
- Code style and formatting
- Type annotations and documentation
- Error handling patterns
- Testing conventions
- Language-specific anti-patterns

### Framework-Specific Instructions

Focus on:
- Framework conventions and patterns
- Configuration requirements
- Common pitfalls
- Integration patterns

### Domain-Specific Instructions

Focus on:
- Domain terminology
- Business rule enforcement
- Data validation requirements
- Security considerations

## Validation

Before committing instruction files:

- [ ] Frontmatter is valid YAML
- [ ] `applyTo` pattern is correct and not too broad
- [ ] Description is clear and concise
- [ ] Content is organized with clear sections
- [ ] Examples are accurate and helpful
- [ ] No conflicts with copilot-instructions.md
- [ ] No duplicate content with other instruction files
