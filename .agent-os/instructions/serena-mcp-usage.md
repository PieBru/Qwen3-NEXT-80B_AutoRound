---
description: Serena MCP Usage Guidelines for Agent OS Projects
globs:
alwaysApply: true
version: 1.0
encoding: UTF-8
---

# Serena MCP Usage Guidelines

<ai_meta>
  <rules>Use Serena MCP for semantic code analysis and editing when available</rules>
  <priority>high</priority>
</ai_meta>

## Overview

Serena MCP (Model Context Protocol) provides semantic code retrieval and editing capabilities based on symbolic understanding of code. When available in a project, use Serena's capabilities to enhance code analysis and modification tasks.

<mcp_detection>
  <check_availability>
    AT START OF CODING TASKS:
    CHECK if MCP tools starting with "mcp__serena" are available
    IF available:
      PREFER Serena tools over basic file operations
      USE semantic understanding for better code navigation
  </check_availability>
</mcp_detection>

## When to Use Serena MCP

<use_cases>
  <semantic_search>
    - Finding function definitions across codebase
    - Locating all usages of a variable or function
    - Understanding code relationships and dependencies
    - Finding similar code patterns
  </semantic_search>
  
  <code_navigation>
    - Jumping to definitions
    - Finding references
    - Understanding code structure
    - Exploring inheritance hierarchies
  </code_navigation>
  
  <intelligent_editing>
    - Refactoring with semantic awareness
    - Renaming symbols across entire codebase
    - Making structural code changes
    - Ensuring consistency in modifications
  </intelligent_editing>
</use_cases>

## Integration with Agent OS Workflows

<workflow_integration>
  <spec_implementation>
    WHEN implementing specs:
    1. USE Serena to understand existing code structure
    2. FIND similar patterns before implementing new features
    3. ENSURE consistency with existing codebase
  </spec_implementation>
  
  <code_analysis>
    WHEN analyzing code:
    1. USE semantic search to understand relationships
    2. FIND all usages before making changes
    3. VERIFY impact of modifications
  </code_analysis>
  
  <refactoring>
    WHEN refactoring:
    1. USE Serena's rename capabilities
    2. ENSURE all references are updated
    3. MAINTAIN semantic consistency
  </refactoring>
</workflow_integration>

## Best Practices

<best_practices>
  <tool_selection>
    - PREFER Serena MCP tools when available
    - FALLBACK to standard tools if MCP unavailable
    - COMBINE both for comprehensive coverage
  </tool_selection>
  
  <semantic_awareness>
    - UNDERSTAND code relationships before changes
    - USE language-specific understanding
    - MAINTAIN code semantics during edits
  </semantic_awareness>
  
  <efficiency>
    - BATCH related searches together
    - USE semantic patterns to find similar code
    - LEVERAGE symbolic understanding for accuracy
  </efficiency>
</best_practices>

## Supported Languages

<language_support>
  - Python
  - TypeScript/JavaScript
  - PHP
  - Go
  - Rust
  - C/C++
  - Java
  - And more...
</language_support>

## Fallback Strategy

<fallback>
  IF Serena MCP not available:
    USE standard Agent OS tools (Grep, Glob, Read, Edit)
    MAINTAIN same semantic approach manually
    DOCUMENT when MCP would have helped
</fallback>

## Example Usage Patterns

<examples>
  <find_all_usages>
    # Instead of grep for function name
    USE: mcp__serena_find_references("function_name")
    BENEFIT: Finds actual usages, not just text matches
  </find_all_usages>
  
  <rename_symbol>
    # Instead of search-replace
    USE: mcp__serena_rename_symbol("old_name", "new_name")
    BENEFIT: Updates all references semantically
  </rename_symbol>
  
  <understand_structure>
    # Instead of reading multiple files
    USE: mcp__serena_get_code_structure()
    BENEFIT: Gets semantic understanding of relationships
  </understand_structure>
</examples>

## Important Notes

- Serena MCP enhances but doesn't replace standard tools
- Always verify MCP availability before relying on it
- Combine semantic and text-based approaches for best results
- Document when semantic understanding would improve efficiency

---

*Serena MCP integration for enhanced semantic code understanding in Agent OS projects*