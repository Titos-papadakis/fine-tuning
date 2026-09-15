"""
Multi-tenant serving: discovering which of a workload's customers have a
production adapter ready, so one shared vLLM engine can serve all of them
(see ftspec.serving.serve's multi-adapter mode and
shared_server.py::discover_production_adapters()).
"""
