"""
Define 2 MF inference models - basic (eg vanilla bert) and domain adapt model.
"""

import torch
import transformers
from transformers import AutoModel, AutoConfig

from .grad_rev_fn import ReverseLayerF
from .modules import FFClassifier, Reconstruction, Transformation

transformers.logging.set_verbosity_error()


class MFBasic(torch.nn.Module):
    """
    Model for moral foundation prediction by simply finetuning a pretrained model.

    Attr:
        feature: the feature encoder, ie the pretrained model
        embedding_dim: usually 768 for bert or xlm-t
        mf_classifier: the prediction layer in the end
    """

    def __init__(self, pretrained_dir: str, n_mf_classes: int,
                 dropout_rate: float):
        """
        Args:
            pretrained_dir: where to load pretrained model file
            n_mf_classes: number of moral foundation classes in the data
            dropout_rate: how much nodes to drop to mitigate overfitting. from 0 to 1.
        """
        super(MFBasic, self).__init__()

        self.feature = AutoModel.from_pretrained(pretrained_dir)

        self.embedding_dim = self.feature.embeddings.word_embeddings.embedding_dim

        self.mf_classifier = FFClassifier(self.embedding_dim, 100,
                                          n_mf_classes,
                                          dropout_rate)  # hidden_dim = 100

    def forward(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids, attention_mask=att_mask)

        class_output = self.mf_classifier(feature.pooler_output)

        return {'class_output': class_output}

    def gen_feature_embeddings(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids, attention_mask=att_mask)

        return feature.last_hidden_state, feature.pooler_output


class MFDomainAdapt(torch.nn.Module):
    """
    Model for moral foundation prediction with domain adversarial training.

    Attr:
        n_mf_classes: how many mf classes are there
        n_domain_classes: how many domains are there
        feature: the feature encoder, ie the pretrained model
        embedding_dim: usually 768 for bert or xlm-t
        has_rec: boolean indicator for whether to use rec_module
        rec_module: a module to prevent the feature encoder being corrupted by the adversarial training.
        has_trans: boolean whether to use trans module
        trans_module: a module to facilitate domain-invariant transformation
        mf_classifier: the prediction layer in the end
        domain_classifier: the adversary to distinguish which domain a data example comes from
    """

    def __init__(self,
                 pretrained_dir: str,
                 n_mf_classes: int,
                 n_domain_classes: int,
                 dropout_rate: float,
                 device: str,
                 has_trans=False,
                 has_rec=False):

        super(MFDomainAdapt, self).__init__()

        self.n_mf_classes = n_mf_classes
        self.n_domain_classes = n_domain_classes
        self.has_rec = has_rec
        self.has_trans = has_trans
        self.is_adv = False
        self.lambda_domain = 0.0

        self.feature = AutoModel.from_pretrained(pretrained_dir)
        self.embedding_dim = self.feature.embeddings.word_embeddings.embedding_dim

        self.rec_module = Reconstruction(self.embedding_dim,
                                         device) if self.has_rec else None

        self.trans_module = Transformation(self.embedding_dim,
                                           device) if self.has_trans else None

        self.mf_classifier = FFClassifier(
            self.embedding_dim + self.n_domain_classes, 100, self.n_mf_classes,
            dropout_rate)  # hidden_dim = 100

        self.domain_classifier = FFClassifier(self.embedding_dim, 100,
                                              self.n_domain_classes,
                                              dropout_rate)

    def gen_feature_embeddings(self, input_ids, att_mask):

        feature = self.feature(input_ids=input_ids, attention_mask=att_mask)

        return feature.last_hidden_state, feature.pooler_output

    def update_is_adv(self, adv):
        self.is_adv = adv

    def update_lambda_domain(self,lambda_domain):
        self.lambda_domain = lambda_domain

    def forward(self,
                input_ids,
                att_mask,
                domain_labels):
        """
        Args:
            input_ids: generated from tokenizer
            att_mask: attention mask generated from tokenizer
            domain_labels: list of which domain each example belongs to
            lambda_domain: regularization hyperparameter for domain adversarial training
            adv: bool whether we are doing adversarial training
        """
        # feature encoder
        last_hidden_state, pooler_output = self.gen_feature_embeddings(
            input_ids, att_mask)

        # reconstruction
        rec_embeddings = None
        if self.has_rec and self.is_adv:
            rec_embeddings = self.rec_module(last_hidden_state)

        # domain-invariant transformation
        if self.has_trans and self.is_adv:
            pooler_output = self.trans_module(pooler_output)

        # concat domain features onto pretrained LM embeddings before sending to MF classifier
        domain_feature = torch.nn.functional.one_hot(
            domain_labels, num_classes=self.n_domain_classes).squeeze(1)
        class_output = torch.cat((pooler_output, domain_feature), dim=1)

        # connect to mf classifier
        class_output = self.mf_classifier(class_output)

        # connect to domain classifier with gradient reversal
        domain_output = None
        if self.is_adv:
            domain_output = self.domain_classifier(
                ReverseLayerF.apply(pooler_output, self.lambda_domain))

        return {
            'class_output': class_output,
            'domain_output': domain_output,
            'rec_embed': rec_embeddings
        }
