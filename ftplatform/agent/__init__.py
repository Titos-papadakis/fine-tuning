"""
The customer's model as a support agent: it converses AND acts.

Each turn the model reads the conversation so far and emits one step as
JSON -- reply to the customer, call one of the customer's tools, or wait for
the customer -- validated against a schema built from the customer's own
tool catalog (tools.step_schema). That makes an agent just another custom
workload: the same import, training, benchmark and deploy path as
extraction, with a runtime loop (session.respond) on top.

What stands between the model and the customer's systems is executor.py:
every tool has a permission (auto / approval / forbidden), a customer starts
in dry-run until they switch to live, and every call is recorded.
"""
