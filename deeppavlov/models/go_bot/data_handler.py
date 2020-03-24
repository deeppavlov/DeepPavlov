import re
from typing import List

import numpy as np

# from deeppavlov.models.go_bot.network import log
from deeppavlov.models.go_bot.tracker import DialogueStateTracker


class DataHandler:
    def _encode_response(self, gobot_obj, act: str) -> int:
        return gobot_obj.templates.actions.index(act)

    def _decode_response(self, gobot_obj, action_id: int, tracker: DialogueStateTracker) -> str:
        """
        Convert action template id and entities from tracker
        to final response.
        """
        template = gobot_obj.templates.templates[int(action_id)]

        slots = tracker.get_state()
        if tracker.db_result is not None:
            for k, v in tracker.db_result.items():
                slots[k] = str(v)

        resp = template.generate_text(slots)
        # in api calls replace unknown slots to "dontcare"
        if action_id == gobot_obj.api_call_id:
            resp = re.sub("#([A-Za-z]+)", "dontcare", resp).lower()
        return resp

    def _encode_context(self, gobot_obj,
                        tokens: List[str],
                        tracker: DialogueStateTracker) -> List[np.ndarray]:
        # Bag of words features
        bow_features = []
        if callable(gobot_obj.bow_embedder):
            tokens_idx = gobot_obj.word_vocab(tokens)
            bow_features = gobot_obj.bow_embedder([tokens_idx])[0]
            bow_features = bow_features.astype(np.float32)

        # Embeddings
        emb_features = []
        emb_context = np.array([], dtype=np.float32)
        if callable(gobot_obj.embedder):
            if gobot_obj.attn:
                if tokens:
                    pad = np.zeros((gobot_obj.attn.max_num_tokens,
                                    gobot_obj.attn.token_size),
                                   dtype=np.float32)
                    sen = np.array(gobot_obj.embedder([tokens])[0])
                    # TODO : Unsupport of batch_size more than 1
                    emb_context = np.concatenate((pad, sen))
                    emb_context = emb_context[-gobot_obj.attn.max_num_tokens:]
                else:
                    emb_context = np.zeros((gobot_obj.attn.max_num_tokens,
                                            gobot_obj.attn.token_size),
                                           dtype=np.float32)
            else:
                emb_features = gobot_obj.embedder([tokens], mean=True)[0]
                # random embedding instead of zeros
                if np.all(emb_features < 1e-20):
                    emb_dim = gobot_obj.embedder.dim
                    emb_features = np.fabs(np.random.normal(0, 1 / emb_dim, emb_dim))

        # Intent features
        intent_features = []
        if callable(gobot_obj.intent_classifier):
            intent_features = gobot_obj.intent_classifier([' '.join(tokens)])[1][0]

            if gobot_obj.debug:
                intent = gobot_obj.intents[np.argmax(intent_features[0])]
                # log.debug(f"Predicted intent = `{intent}`")

        attn_key = np.array([], dtype=np.float32)
        if gobot_obj.attn:
            if gobot_obj.attn.action_as_key:
                attn_key = np.hstack((attn_key, tracker.prev_action))
            if gobot_obj.attn.intent_as_key:
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

        if gobot_obj.debug:
            # log.debug(f"Context features = {context_features}")
            debug_msg = f"num bow features = {bow_features}" + \
                        f", num emb features = {emb_features}" + \
                        f", num intent features = {intent_features}" + \
                        f", num state features = {len(state_features)}" + \
                        f", num context features = {len(context_features)}" + \
                        f", prev_action shape = {len(tracker.prev_action)}"
            # log.debug(debug_msg)

        concat_feats = np.hstack((bow_features, emb_features, intent_features,
                                  state_features, context_features,
                                  tracker.prev_action))
        return concat_feats, emb_context, attn_key