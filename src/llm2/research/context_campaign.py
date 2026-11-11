"""Explicit recipe entry; shared bounded language orchestration."""
import time
STARTED=time.monotonic()
from . import context_protocol
from .stories_campaign import main

if __name__=='__main__':main(recipe=context_protocol,started=STARTED)
