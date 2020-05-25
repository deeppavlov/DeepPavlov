class DigitizedPolicyFeatures:
    def __init__(self, attn_key, concat_feats, action_mask):
        self.attn_key = attn_key
        self.concat_feats = concat_feats
        self.action_mask = action_mask
