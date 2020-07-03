import torch.nn as nn

from examples.speech_recognition.models.vggtransformer import VGGTransformerModel, vggtransformer_2, TransformerDecoder
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import LinearizedConvolution, TransformerDecoderLayer


class DirichletTransformerDecoder(TransformerDecoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dirichlet_projection = nn.Linear(self.output_embed_dim, 1)

    def forward(self, prev_output_tokens, encoder_out=None, incremental_state=None):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for input feeding/teacher forcing
            encoder_out (Tensor, optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
        Returns:
            tuple:
                - the last decoder layer's output of shape `(batch, tgt_len,
                  vocab)`
                - the last decoder layer's attention weights of shape `(batch,
                  tgt_len, src_len)`
        """
        target_padding_mask = (
            (prev_output_tokens == self.padding_idx).to(prev_output_tokens.device)
            if incremental_state is None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        # embed tokens
        x = self.embed_tokens(prev_output_tokens)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        for layer in self.conv_layers:
            if isinstance(layer, LinearizedConvolution):
                x = layer(x, incremental_state)
            else:
                x = layer(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_inference(x, incremental_state)

        # decoder layers
        for layer in self.layers:
            if isinstance(layer, TransformerDecoderLayer):
                x, _ = layer(
                    x,
                    (encoder_out["encoder_out"] if encoder_out is not None else None),
                    (
                        encoder_out["encoder_padding_mask"].t()
                        if encoder_out["encoder_padding_mask"] is not None
                        else None
                    ),
                    incremental_state,
                    self_attn_mask=(
                        self.buffered_future_mask(x)
                        if incremental_state is None
                        else None
                    ),
                    self_attn_padding_mask=(
                        target_padding_mask if incremental_state is None else None
                    ),
                )
            else:
                x = layer(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        dirichlet_params = self.dirichlet_projection(x)
        x = self.fc_out(x)

        return x, {'dirichlet_params': dirichlet_params}


@register_model('dirichlet_vggtransformer')
class DirichletVGGTransformerModel(VGGTransformerModel):
    @classmethod
    def build_decoder(cls, args, task):
        return DirichletTransformerDecoder(
            dictionary=task.target_dictionary,
            embed_dim=args.tgt_embed_dim,
            transformer_config=eval(args.transformer_dec_config),
            conv_config=eval(args.conv_dec_config),
            encoder_output_dim=args.enc_output_dim,
        )


@register_model_architecture('dirichlet_vggtransformer', 'dirichlet_vggtransformer_2')
def dirichlet_vggtransformer_2(args):
    vggtransformer_2(args)
