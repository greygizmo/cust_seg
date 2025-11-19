"""
Playbook definitions and registry.
"""
from dataclasses import dataclass

@dataclass
class PlaybookDefinition:
    name: str
    description: str
