import torch.nn as nn

from fairseq.models import register_model, register_model_architecture
from fairseq.models.transformer import TransformerDecoder, TransformerModel, transformer_wmt_en_de_big


@register_model('dirichlet_transformer')
class DirichletTransformerModel(TransformerModel):
    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return DirichletTransformerDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )


class DirichletTransformerDecoder(TransformerDecoder):
    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn)
        self.dirichlet_projection = nn.Linear(self.output_embed_dim, 1)

    def extract_features(
            self,
            prev_output_tokens,
            encoder_out=None,
            incremental_state=None,
            full_context_alignment=False,
            alignment_layer=None,
            alignment_heads=None,
            **unused,
    ):
        x, extra = super().extract_features(
            prev_output_tokens,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            full_context_alignment=full_context_alignment,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            **unused
        )
        extra['dirichlet_params'] = self.dirichlet_projection(x)
        return x, extra


@register_model_architecture('dirichlet_transformer', 'dirichlet_transformer_wmt_en_de_big')
def dirichlet_transformer_wmt_en_de_big(args):
    transformer_wmt_en_de_big(args)
