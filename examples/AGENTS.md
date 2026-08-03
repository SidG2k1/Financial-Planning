# Example instructions

These rules supplement the repository instructions for `examples/`.

- Examples are user-facing compatibility fixtures. Use only documented public APIs and fields.
- Keep examples minimal enough to understand without reading package internals, while remaining
  complete and runnable with the README's installation instructions.
- JSON examples must be syntactically valid, use finite values, and satisfy the current serializer
  schema. Update the README when an example demonstrates a new supported workflow.
- Use fictional, non-sensitive values. Do not copy personalized task inputs into tracked examples.
- Smoke-test changed examples through the CLI or documented loading API.
