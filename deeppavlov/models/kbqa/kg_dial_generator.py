# Copyright 2017 Neural Networks and Deep Learning lab, MIPT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from logging import getLogger
from typing import List

from transformers import AutoTokenizer, AutoModelWithLMHead
from deeppavlov.core.common.registry import register
from deeppavlov.core.models.component import Component
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.file import load_pickle
from deeppavlov.models.kbqa.rel_ranking_infer import RelRankerInfer
from deeppavlov.models.kbqa.wiki_parser import WikiParser

log = getLogger(__name__)


@register('kg_dial_generator')
class KGDialGenerator(Component):
    def __init__(self, transformer_model: str,
                       path_to_model: str, *args, **kwargs) -> None:
                       
        self.tokenizer = AutoTokenizer.from_pretrained(transformer_model)
        special_tokens_dict = {"sep_token": "<SEP>"}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model = AutoModelWithLMHead.from_pretrained(str(expand_path(path_to_model)))
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    def __call__(self, prev_utterances_batch: List[str], triplets_batch: List[List[str]]) -> List[str]:
        generated_utterances_batch = []
        for prev_utterance, triplets in zip(prev_utterances_batch, triplets_batch):
            triplets = ' '.join([' '.join(triplet) for triplet in triplets])
            context_plus_gk = triplets + " <SEP> " + prev_utterance + self.tokenizer.eos_token
            input_ids = self.tokenizer.encode(context_plus_gk, return_tensors="pt")
            generated_ids = self.model.generate(input_ids, max_length=200,
                                                pad_token_id=tokenizer.eos_token_id,
                                                no_repeat_ngram_size=3, do_sample=True,
                                                top_k=100, top_p=0.7, temperature=0.8)
            generated_utterance = self.tokenizer.decode(generated_ids[:, input_ids.shape[-1]:][0],
                                                        skip_special_tokens=True)
            generated_utterances_batch.append(generated_utterance)
        
        return generated_utterance_batch
        

@register('dial_path_ranker')    
class DialPathRanker(Component):
    def __init__(self, wiki_parser: WikiParser,
                       path_ranker: RelRankerInfer,
                       type_paths_file: str,
                       type_groups_file: str = None,
                       *args, **kwargs) -> None:
        
        self.type_paths = load_pickle(expand_path(type_paths_file))
        if type_groups_file:
            self.type_groups = load_pickle(expand_path(type_groups_file))
        self.wiki_parser = wiki_parser
        self.path_ranker = path_ranker
    
    def __call__(self, utterances_batch: List[str], entities_batch: List[List[str]]) -> List[List[str]]:
        paths_batch = []
        for utterance, entities_list in zip(utterances_batch, entities_batch):
            entity = entities_list[0]
            log.debug(f"seed entity {entity}")
            entity_types = self.wiki_parser(["find_types"], [entity])[0]
            log.debug(f"entity types {entity_types}")
            candidate_paths = set()
            for entity_type in entity_types:
                candidate_paths = candidate_paths.union(self.type_paths[entity_type])
            
            paths_with_scores = self.path_ranker.rank_paths(utterance, candidate_paths)
            top_paths = [path for path, score in paths_with_scores]
            retrieved_paths = self.wiki_parser(["retrieve_paths"], [[entity, top_paths]])[0]
            log.debug(f"retrieved paths {retrieved_paths}")
            if retrieved_paths:
                paths_batch.append(retrieved_paths[0])
            else:
                paths_batch.append([])
                
        return paths_batch
