from pathlib import Path
from typing import Dict, List, Union

from deeppavlov.core.common.file import read_yaml


class DomainKnowledge:
    """the DTO-like class to store the domain knowledge from the domain yaml config."""

    def __init__(self, domain_knowledge_di: Dict):
        self.known_entities: List = domain_knowledge_di.get("entities", [])
        self.known_intents: List = domain_knowledge_di.get("intents", [])
        self.known_actions: List = domain_knowledge_di.get("actions", [])
        self.known_slots: Dict = domain_knowledge_di.get("slots", {})
        self.response_templates: Dict = domain_knowledge_di.get("responses", {})
        self.session_config: Dict = domain_knowledge_di.get("session_config", {})
        self.forms: Dict = domain_knowledge_di.get("forms", {})

    @classmethod
    def from_yaml(cls, domain_yml_fpath: Union[str, Path] = "domain.yml"):
        """
        Parses domain.yml domain config file into the DomainKnowledge object
        Args:
            domain_yml_fpath: path to the domain config file, defaults to domain.yml
        Returns:
            the loaded DomainKnowledge obect
        """
        return cls(read_yaml(domain_yml_fpath))