"""
API-key authentication for the serving layer. A key's plaintext is shown
exactly once, at creation -- only its sha256 hash is ever stored, so this
database being read cannot leak a usable credential.
"""
