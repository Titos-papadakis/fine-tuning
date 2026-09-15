"""
Per-customer usage metering -- aggregate monthly counters, not a
request-level log (that's memory/production_log.jsonl per customer, see
ftplatform/monitoring/). What a $-per-month customer relationship actually
needs to bill against: requests, tokens, cache hits, by calendar month.
"""
