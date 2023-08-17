# Pull Request Description

## Type of Change
<!-- Mark the appropriate option with an [x] -->
- [ ] New Model Support
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Other (please describe):


## Changes Overview
<!-- Provide a brief summary of the changes in this PR -->

## Motivation and Context
<!-- Explain why this change is necessary and what problem it solves -->

## Checklist
<!-- Mark completed items with an [x] -->
- [ ] I have performed a self-review of my own code
- [ ] I have added tests that prove my fix is effective or that my feature works (If needed)

## Additional Information
<!-- Any additional information, configuration, or data that might be necessary to reproduce the issue or use the new feature -->

## Related Issues
<!-- Link any related issues here using the syntax: Closes #123, Fixes #456 -->



----
# Conventional commit
```
type(optional scope): description
```
----
# Type candidate
  - Model Updates
    - `model`: Adding New models or Bugfix for existing models
      - ex) Add LlavaNext 
      - ex) Bugfix Whisper
  - Enhancements
    - `performance`: Optimizing some models or this library itself
      - ex) Loading RBLNModel faster
      - ex) Optimizing Memory Usage of DecoderOnlyModel
  - Code Refactor
    - `refactor`: Re-arrange class architecture, or more.
      - ex) Refactor Seq2Seq
  - Documentation
    - `doc`: Update docstring only     
  - Library Dependencies
    - `dependency`: Update requirements, something like that.
  - Other
    - `other`: None of above.
      - ex) ci update
      - ex) pdm update