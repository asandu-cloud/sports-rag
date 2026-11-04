"""Owner dashboard (tracker D67): how Most likely and Value picks did, for the owner only, on 127.0.0.1:8300.

It keeps its own file (``Index/owner_dashboard.db``) and only reads ``Index/platform.db``, inside ``data_access()``,
in short statements. Nothing here changes the engine or a running job. Start: ``python -m Scripts.ops.owner_dashboard``.
"""
