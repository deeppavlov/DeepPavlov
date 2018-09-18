from keras.layers import Input, LSTM, Embedding, GlobalMaxPooling1D, Lambda, subtract, Conv2D, Dropout
from keras.layers import Dense, Activation, Reshape, CuDNNGRU, CuDNNLSTM, GRU, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import Dot, Subtract, Add, Multiply, concatenate
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers.wrappers import Bidirectional
from keras.optimizers import Adam
from keras.initializers import glorot_uniform, Orthogonal
from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np
from deeppavlov.core.models.tf_backend import TfModelMeta
from deeppavlov.core.common.log import get_logger
from deeppavlov.core.layers import keras_layers
from pathlib import Path
from deeppavlov.core.commands.utils import expand_path
from deeppavlov.core.common.registry import register
from deeppavlov.models.ranking.siamese_network import SiameseEmbeddingsNetwork
from keras.layers.advanced_activations import LeakyReLU


log = get_logger(__name__)

@register('qa_memnet')
class QAMemnet(SiameseEmbeddingsNetwork):

    def __init__(self,
                 bdr: bool = True,
                 pool: bool = True,
                 cudnn: bool = True,
                 dsn: bool = False,
                 dsl: int = 2,
                 dsd: int = 300,
                 dsp: float = 0.3,
                 dp: float = 0.0,
                 rdp: float = 0.1,
                 drp: float = 0.2,
                 lstm_dim: int = 300,
                 seq_len: int = 20,
                 layers: int = 2,
                 dlrs = 1,
                 **kwargs):

        self.bdr = bdr
        self.pool = pool
        self.cudnn = cudnn
        self.dsn = dsn
        self.dsl = dsl
        self.dsd = dsd
        self.dsp = dsp
        self.dp = dp
        self.rdp = rdp
        self.drp = drp
        self.lstm_dim = lstm_dim
        self.seq_len = seq_len
        self.layers = layers
        self.dlrs = 1

    def pairwise_mul(self, vests):
        x, y = vests
        return x * y

    def last_timestep(vests):
        return vests[:, -1, :]

    def last_timestep_rev(vests):
        rdim = int(K.int_shape(vests)[-1] / 2)
        return concatenate([vests[:, -1, :rdim], vests[:, 0, rdim:]], axis=-1)

    def cosine_similarity(self, vests):
        x, y = vests
        x = K.l2_normalize(x, axis=-1)
        y = K.l2_normalize(y, axis=-1)
        return K.sum((x * y), axis=-1, keepdims=True)

    def build_embedder(self, seqlen, voc_siz=None,
                       weighted=False, transformed=False, emb_dim=300, prefix='word'):

        inp = Input(shape=(seqlen,))

        if voc_siz is not None:
            trainable = False
            enc = Embedding(voc_siz, emb_dim, input_length=seqlen,
                            trainable=trainable, name="embedding")(inp)
        else:
            enc = Embedding(voc_siz, emb_dim, input_length=seqlen, name="embedding")(inp)

        if transformed:
            trf = Dense(emb_dim)(enc)
            act = LeakyReLU()(trf)
        else:
            act = enc

        if weighted:
            wwt = Embedding(voc_siz, 1, input_length=seqlen,
                            weights=[np.ones(shape=(voc_siz, 1))])(inp)
            wac = Reshape((-1, 1))(Activation("softmax")(Reshape((-1,))(wwt)))
            out = Lambda(self.pairwise_mul, name='MulLayer')([act, wac])
        else:
            out = act

        return Model(inputs=[inp], outputs=out, name=prefix + '_embedding_model')

    def build_rnn_encoder(self, input_shape, return_sequences=False, bidirectional=False, lstm_dim=300,
                          prefix="word", rdp=0.1, dp=0.0, layers=1, lstm=True, cudnn=True, use_pool=True):

        inp = Input(shape=tuple(input_shape[-2:]))
        rnn = inp

        if lstm:
            if cudnn:
                Cell = CuDNNLSTM
            else:
                Cell = LSTM
        else:
            if cudnn:
                Cell = CuDNNGRU
            else:
                Cell = GRU

        for li in range(layers):
            if bidirectional:
                rnn = Bidirectional(Cell(lstm_dim, return_sequences=True))(rnn)
                get_last_step = self.last_timestep_rev
            else:
                if cudnn:
                    rnn = Cell(lstm_dim, return_sequences=True)(rnn)
                else:
                    rnn = Cell(lstm_dim, return_sequences=True, dropout=dp, recurrent_dropout=rdp)(rnn)
                get_last_step = self.last_timestep

        if not return_sequences and use_pool:
            last = Lambda(get_last_step)(rnn)
            mean = GlobalAveragePooling1D()(rnn)
            maxp = GlobalMaxPooling1D()(rnn)
            sms = [last, mean, maxp]

            out = [concatenate(sms)] + sms

        elif not return_sequences:
            out = Lambda(get_last_step)(rnn)
        else:
            out = rnn

        mod = Model(inputs=inp, outputs=out, name=prefix + '_lstm_encoder')
        return mod


    def embeddings_model(self):
        dense_dim = self.lstm_dim

        if self.pool:
            dense_dim *= 3
        if self.bdr:
            dense_dim *= 2

        # inputs for contexts and reply
        inp_ctx = Input(shape=(self.seq_len,), name='inp_ctx')
        inp_rpl = Input(shape=(self.seq_len,), name='inp_reply')

        # word embedding model
        embedder = self.build_embedder(self.seq_len)

        # shared sentence-level encoder
        encoder_ctx = self.build_rnn_encoder(embedder.output_shape, return_sequences=False,
                                        lstm_dim=self.lstm_dim, bidirectional=self.bdr, cudnn=self.cudnn,
                                        dp=self.edp, rdp=self.rdp,
                                        prefix="sentence", layers=self.layers, use_pool=self.pool)

        def dense_comb(layers):
            # shared dense layer to combine context and reply vectors
            inp1 = Input(shape=encoder_ctx.output_shape[0])
            inp2 = Input(shape=encoder_ctx.output_shape[0])
            cnc1 = concatenate([inp1, inp2])

            bnr1 = BatchNormalization()(cnc1)
            drp1 = Dropout(self.dropout)(bnr1)
            dns1 = drp1

            for _ in range(layers):
                dns1 = Dense(self.dense_dim, activation='relu')(dns1)

            return Model(inputs=[inp1, inp2], outputs=dns1, name='merge_model')

        dm = dense_comb(self.dlrs)

        emb_ctx = embedder(inp_ctx)
        emb_rpl = embedder(inp_rpl)

        # encode contexts and reply
        enc_ctx, c1, c2, c3 = encoder_ctx(emb_ctx)
        enc_rpl, r1, r2, r3 = encoder_ctx(emb_rpl)

        sims = [Lambda(self.cosine_similarity)([enc_ctx, enc_rpl]),
                Lambda(self.cosine_similarity)([c1, r1]),
                Lambda(self.cosine_similarity)([c1, r2]),
                Lambda(self.cosine_similarity)([c1, r3]),
                Lambda(self.cosine_similarity)([c2, r1]),
                Lambda(self.cosine_similarity)([c2, r2]),
                Lambda(self.cosine_similarity)([c2, r3]),
                Lambda(self.cosine_similarity)([c3, r1]),
                Lambda(self.cosine_similarity)([c3, r2]),
                Lambda(self.cosine_similarity)([c3, r3])]

        # condition context encoding on reply encoding
        enc_ctx = dm([enc_ctx, enc_rpl])

        if self.dsn:
            dsn = self.build_deep_sim_net(encoder_ctx.output_shape, inr_dim=self.dsd, layers=self.dsl, DROPOUT=self.dsp)
            css = dsn([enc_ctx, enc_rpl])
        else:
            sims.append(Lambda(self.cosine_similarity)([enc_ctx, enc_rpl]))

        sims = concatenate(sims)
        fc1 = Dense(16, activation='relu')(sims)
        # output neuron (during pretraining we do binary classification)
        fc2 = Dense(1, activation='sigmoid', name='relevance')(fc1)

        model = Model(inputs=[inp_ctx, inp_rpl], outputs=fc2)
        att_model = None  # Model(inputs=[inp_ctx, inp_rpl], outputs=[att_ctx, att_rpl])

        model.compile(optimizer='adam',
                      loss=contrastive_loss,
                      metrics=['accuracy'])

        return model
