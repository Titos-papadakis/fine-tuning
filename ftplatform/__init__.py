"""
`ftplatform` — multi-customer orchestration around the `ftspec` engine.

`ftspec` proves the technique for one workload, run by hand, for one customer
at a time. This package adds the layer that runs it for many customers,
isolated from each other, without changing how `ftspec` itself works.

Every module here reaches `ftspec` only through a `CustomerContext`
(`ftplatform.customers.context`), which is the sole place allowed to build a
customer-scoped `Config`/`Profile` pair. See that module's docstring for why.
"""
