import re
from logging import getLogger
from typing import List

import numpy as np

# from deeppavlov.models.go_bot.network import log
import deeppavlov.models.go_bot.templates as templ
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.models.go_bot.tracker import DialogueStateTracker

log = getLogger(__name__)


class DataHandler:

    def __init__(self, debug, template_path, template_type, word_vocab, bow_embedder, api_call_action, embedder):
        self.debug = debug

        template_path = expand_path(template_path)
        template_type = getattr(templ, template_type)
        log.info(f"[loading templates from {template_path}]")
        self.templates = templ.Templates(template_type).load(template_path)  # upper-level model logic
        log.info(f"{len(self.templates)} templates loaded.")

        self.api_call_id = -1  # api call should have smth like action index
        if api_call_action is not None:
            self.api_call_id = self.templates.actions.index(api_call_action)  # upper-level model logic

        self.word_vocab = word_vocab
        self.bow_embedder = bow_embedder
        self.embedder = embedder

    def use_bow_embedder(self):
        return callable(self.bow_embedder)

    def word_vocab_size(self):
        return len(self.word_vocab) if self.word_vocab else None


    def _encode_response(self, act: str) -> int:
        # conversion
        return self.templates.actions.index(act)

    def _decode_response(self, action_id: int, tracker: DialogueStateTracker) -> str:
        """
        Convert action template id and entities from tracker
        to final response.
        """
        # conversion
        template = self.templates.templates[int(action_id)]

        slots = tracker.get_state()
        if tracker.db_result is not None:
            for k, v in tracker.db_result.items():
                slots[k] = str(v)

        resp = template.generate_text(slots)
        # in api calls replace unknown slots to "dontcare"
        if action_id == self.api_call_id:
            # todo: move api_call_id here
            resp = re.sub("#([A-Za-z]+)", "dontcare", resp).lower()
        return resp

    def embed_tokens(self, tokens, use_attn, attn_window_size, attn_token_size):
        if use_attn:
            padding_length = attn_window_size - len(tokens)
            padding = np.zeros(shape=(padding_length, attn_token_size), dtype=np.float32)
            if tokens:
                tokens_embedded = np.array(self.embedder([tokens])[0])
                emb_context = np.concatenate((padding, tokens_embedded))
            else:
                emb_context = padding
            return emb_context
        else:
            emb_features = self.embedder([tokens], mean=True)[0]
            # random embedding instead of zeros
            if np.all(emb_features < 1e-20):
                emb_dim = self.embedder.dim
                emb_features = np.fabs(np.random.normal(0, 1 / emb_dim, emb_dim))
            return emb_features

    def _encode_context(self, gobot_obj,
                        use_attn,
                        attn_window_size,
                        attn_token_size,
                        attn_action_as_key,
                        attn_intent_as_key,
                        tokens: List[str],
                        tracker: DialogueStateTracker) -> List[np.ndarray]:

        # Bag of words features
        bow_features = []
        if self.use_bow_embedder():
            tokens_idx = self.word_vocab(tokens)
            bow_features = self.bow_embedder([tokens_idx])[0]
            bow_features = bow_features.astype(np.float32)


        # Intent features
        intent_features = []
        if callable(gobot_obj.intent_classifier):
            intent_features = gobot_obj.intent_classifier([' '.join(tokens)])[1][0]

            if self.debug:
                intent = gobot_obj.intents[np.argmax(intent_features[0])]
                # log.debug(f"Predicted intent = `{intent}`")

        attn_key = np.array([], dtype=np.float32)
        tracker_prev_action = tracker.prev_action
        if use_attn:
            if attn_action_as_key:
                attn_key = np.hstack((attn_key, tracker_prev_action))
            if attn_intent_as_key:
                attn_key = np.hstack((attn_key, intent_features))
            if len(attn_key) == 0:
                attn_key = np.array([1], dtype=np.float32)

        state_features = tracker.get_features()

        # Other features
        # todo (?) rename tracker -> dlg_stt_tracker
        result_matches_state = 0.
        if tracker.db_result is not None:
            matching_items = tracker.get_state().items()
            result_matches_state = all(v == tracker.db_result.get(s)
                                       for s, v in matching_items
                                       if v != 'dontcare') * 1.
        context_features = np.array([
            bool(tracker.current_db_result) * 1.,
            (tracker.current_db_result == {}) * 1.,
            (tracker.db_result is None) * 1.,
            bool(tracker.db_result) * 1.,
            (tracker.db_result == {}) * 1.,
            result_matches_state
        ], dtype=np.float32)

        # Embeddings
        emb_features = []
        emb_context = np.array([], dtype=np.float32)

        if callable(self.embedder):
            if use_attn:
                padding_length = attn_window_size - len(tokens)
                padding = np.zeros(shape=(padding_length, attn_token_size), dtype=np.float32)

                if tokens:
                    tokens_embedded = np.array(self.embedder([tokens])[0])
                    emb_context = np.concatenate((padding, tokens_embedded))
                else:
                    emb_context = padding

            else:
                emb_features = self.embedder([tokens], mean=True)[0]
                # random embedding instead of zeros
                if np.all(emb_features < 1e-20):
                    emb_dim = self.embedder.dim
                    emb_features = np.fabs(np.random.normal(0, 1 / emb_dim, emb_dim))

        if self.debug:
            # log.debug(f"Context features = {context_features}")
            debug_msg = f"num bow features = {bow_features}" + \
                        f", num emb features = {emb_features}" + \
                        f", num intent features = {intent_features}" + \
                        f", num state features = {len(state_features)}" + \
                        f", num context features = {len(context_features)}" + \
                        f", prev_action shape = {len(tracker_prev_action)}"
            # log.debug(debug_msg)
        # todo move this out of here
        # todo move attention logic out of here.
        concat_feats = np.hstack((bow_features, emb_features, intent_features,
                                  state_features, context_features,
                                  tracker_prev_action))
        return concat_feats, emb_context, attn_key

    def _prepare_data(self, gobot_obj, dialogue_state_tracker, x: List[dict], y: List[dict]) -> List[np.ndarray]:
        b_features, b_u_masks, b_a_masks, b_actions = [], [], [], []
        b_emb_context, b_keys = [], []  # for attention
        max_num_utter = max(len(d_contexts) for d_contexts in x)
        for d_contexts, d_responses in zip(x, y):
            dialogue_state_tracker.reset_state()
            d_features, d_a_masks, d_actions = [], [], []
            d_emb_context, d_key = [], []  # for attention

            for context, response in zip(d_contexts, d_responses):
                tokens = gobot_obj.tokenizer([context['text'].lower().strip()])[0]

                # update state
                dialogue_state_tracker.get_ground_truth_db_result_from(context)

                if callable(gobot_obj.slot_filler):
                    context_slots = gobot_obj.slot_filler([tokens])[0]
                    dialogue_state_tracker.update_state(context_slots)

                use_attn = bool(gobot_obj.nn_stuff_handler.attn)
                attn_window_size = gobot_obj.nn_stuff_handler.attn.max_num_tokens if use_attn else None
                attn_token_size = gobot_obj.nn_stuff_handler.attn.token_size if use_attn else None
                attn_action_as_key = gobot_obj.nn_stuff_handler.attn.action_as_key if use_attn else None
                attn_intent_as_key = gobot_obj.nn_stuff_handler.attn.intent_as_key if use_attn else None

                features, emb_context, key = self._encode_context(gobot_obj,
                                                                  use_attn,
                                                                  attn_window_size,
                                                                  attn_token_size,
                                                                  attn_action_as_key,
                                                                  attn_intent_as_key,
                                                                  tokens,
                                                                  tracker=dialogue_state_tracker)
                d_features.append(features)
                d_emb_context.append(emb_context)
                d_key.append(key)
                d_a_masks.append(dialogue_state_tracker.calc_action_mask(self.api_call_id))

                action_id = self._encode_response(response['act'])
                d_actions.append(action_id)
                # update state
                # - previous action is teacher-forced here
                dialogue_state_tracker.update_previous_action(action_id)

                if self.debug:
                    # log.debug(f"True response = '{response['text']}'.")
                    if d_a_masks[-1][action_id] != 1.:
                        pass
                        # log.warning("True action forbidden by action mask.")

            # padding to max_num_utter
            num_padds = max_num_utter - len(d_contexts)
            d_features.extend([np.zeros_like(d_features[0])] * num_padds)
            d_emb_context.extend([np.zeros_like(d_emb_context[0])] * num_padds)
            d_key.extend([np.zeros_like(d_key[0])] * num_padds)
            d_u_mask = [1] * len(d_contexts) + [0] * num_padds
            d_a_masks.extend([np.zeros_like(d_a_masks[0])] * num_padds)
            d_actions.extend([0] * num_padds)

            b_features.append(d_features)
            b_emb_context.append(d_emb_context)
            b_keys.append(d_key)
            b_u_masks.append(d_u_mask)
            b_a_masks.append(d_a_masks)
            b_actions.append(d_actions)
        return b_features, b_emb_context, b_keys, b_u_masks, b_a_masks, b_actions
