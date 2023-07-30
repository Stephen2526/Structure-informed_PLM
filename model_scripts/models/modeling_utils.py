# coding=utf-8
"""utilities for PyTorch models. Modified based on TAPE

Author: Yuanfei Sun
"""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
from ctypes import Union
import typing,copy,json,logging,os,math
from io import open
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.weight_norm import weight_norm

from .file_utils import cached_path
from tokenizers import ab_HL_subclass

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model"

logger = logging.getLogger(__name__)


class BaseConfig(object):
    """ Base class for all configuration classes.
        Handles a few parameters common to all models' configurations as well as methods
        for loading/downloading/saving configurations.

        Class attributes (overridden by derived classes):
            - ``pretrained_config_archive_map``: a python ``dict`` of with `short-cut-names`
                (string) as keys and `url` (string) of associated pretrained model
                configurations as values.

        Parameters:
            ``finetuning_task``: string, default `None`. Name of the task used to fine-tune
                the model.
            ``num_labels``: integer, default `2`. Number of classes to use when the model is
                a classification model (sequences/tokens)
            ``output_attentions``: boolean, default `False`. Should the model returns
                attentions weights.
            ``output_hidden_states``: string, default `False`. Should the model returns all
                hidden-states.
            ``torchscript``: string, default `False`. Is the model used with Torchscript.
    """
    pretrained_config_archive_map: typing.Dict[str, str] = {}

    def __init__(self, **kwargs):
        # pop(key[,default]) if key in dict, remove it and return value, else
        # return default. if default is not given and key not in dict, raise KeyError
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', True)
        self.output_hidden_states = kwargs.pop('output_hidden_states', True)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to the directory `save_directory`, so that it
            can be re-loaded using the :func:`~BaseConfig.from_pretrained`
            class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the " \
                                              "model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a :class:`~BaseConfig`
             (or a derived class) from a pre-trained model configuration.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model configuration to
                  load from cache or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing a configuration file saved using the
                  :func:`~BaseConfig.save_pretrained` method,
                  e.g.: ``./my_model_directory/``.
                - a path or url to a saved configuration JSON `file`,
                  e.g.: ``./my_model_directory/configuration.json``.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            kwargs: (`optional`) dict:
                key/value pairs with which to update the configuration object after loading.

                - The values in kwargs of any keys which are configuration attributes will
                  be used to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration
                  attributes is controlled by the `return_unused_kwargs` keyword parameter.

            return_unused_kwargs: (`optional`) bool:

                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)`
                  where `unused_kwargs` is a dictionary consisting of the key/value pairs
                  whose keys are not configuration attributes: ie the part of kwargs which
                  has not been used to update `config` and is otherwise ignored.

        Examples::

            # We can't instantiate directly the base class `BaseConfig` so let's
              show the examples on a derived class: BertConfig
            # Download configuration from S3 and cache.
            config = BertConfig.from_pretrained('bert-base-uncased')
            # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            config = BertConfig.from_pretrained('./test/saved_model/')
            config = BertConfig.from_pretrained(
                './test/saved_model/my_configuration.json')
            config = BertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True, foo=False)
            assert config.output_attention == True
            config, unused_kwargs = BertConfig.from_pretrained(
                'bert-base-uncased', output_attention=True,
                foo=False, return_unused_kwargs=True)
            assert config.output_attention == True
            assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error("Couldn't reach server at '{}' to download pretrained model "
                             "configuration file.".format(config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file))
            return None
        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class BaseModel(nn.Module):
    r""" Base class for all models.

        :class:`~BaseModel` takes care of storing the configuration of
        the models and handles methods for loading/downloading/saving models as well as a
        few methods commons to all models to (i) resize the input embeddings and (ii) prune
        heads in the self-attention heads.

        Class attributes (overridden by derived classes):
            - ``config_class``: a class derived from :class:`~BaseConfig`
              to use as configuration class for this model architecture.
            - ``pretrained_model_archive_map``: a python ``dict`` of with `short-cut-names`
              (string) as keys and `url` (string) of associated pretrained weights as values.

            - ``base_model_prefix``: a string indicating the attribute associated to the
              base model in derived classes of the same architecture adding modules on top
              of the base model.
    """
    config_class: typing.Type[BaseConfig] = BaseConfig
    pretrained_model_archive_map: typing.Dict[str, str] = {}
    base_model_prefix = ""

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, BaseConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class "
                "`BaseConfig`. To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        # Save config in model
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if
            new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self._init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = \
            old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if
            new_num_tokens != config.vocab_size. Take care of tying weights embeddings
            afterwards if the model class has a `tie_weights()` method.

        Arguments:

            new_num_tokens: (`optional`) int:
                New number of tokens in the embedding matrix. Increasing the size will add
                newly initialized vectors at the end. Reducing the size will remove vectors
                from the end. If not provided or None: does nothing and just returns a
                pointer to the input tokens ``torch.nn.Embeddings`` Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embeddings Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.config.vocab_size = new_num_tokens

        # Tie weights again if needed
        if hasattr(self, 'tie_weights'):
            self.tie_weights()
        
        # resize bias layer
        if hasattr(self, 'resize_output_bias'):
            self.resize_output_bias(new_num_tokens)

        return model_embeds

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

        # Prune heads if needed
        if getattr(self.config, 'pruned_heads', False):
            self.prune_heads(self.config.pruned_heads)

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.

            Arguments:

                heads_to_prune: dict with keys being selected layer indices (`int`) and
                    associated values being the list of heads to prune in said layer
                    (list of `int`).
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory, epoch_id, save_freq, num_train_epochs, num_evals_no_improvement):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~BaseModel.from_pretrained`
            ` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where "\
                                              "the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        if isinstance(save_freq, int):
            if (((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs)) and num_evals_no_improvement == 0:
                torch.save(model_to_save.state_dict(), os.path.join(save_directory, f'{WEIGHTS_NAME}_{epoch_id}.bin'))
                torch.save(model_to_save.state_dict(), os.path.join(save_directory, f'{WEIGHTS_NAME}.bin'))
            elif ((epoch_id + 1) % save_freq == 0) or ((epoch_id + 1) == num_train_epochs) or (epoch_id == 0):
                torch.save(model_to_save.state_dict(), os.path.join(save_directory, f'{WEIGHTS_NAME}_{epoch_id}.bin'))
            elif num_evals_no_improvement == 0:
                torch.save(model_to_save.state_dict(), os.path.join(save_directory, f'{WEIGHTS_NAME}.bin'))
        else: # 'improvement' only
            if epoch_id == 0:
                torch.save(model_to_save.state_dict(), os.path.join(save_directory, f'{WEIGHTS_NAME}_{epoch_id}.bin'))
            else:
                torch.save(model_to_save.state_dict(), os.path.join(save_directory, f'{WEIGHTS_NAME}.bin'))



    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

        The model is set in evaluation mode by default using ``model.eval()``
        (Dropout modules are deactivated)
        To train the model, you should first set it back in training mode with ``model.train()``

        The warning ``Weights from XXX not initialized from pretrained model`` means that
        the weights of XXX do not come pre-trained with the rest of the model.
        It is up to you to train those weights with a downstream fine-tuning task.

        The warning ``Weights from XXX not used in YYY`` means that the layer XXX is not used
        by YYY, therefore those weights are discarded.

        Parameters:
            pretrained_model_name_or_path: either:

                - a string with the `shortcut name` of a pre-trained model to load from cache
                  or download, e.g.: ``bert-base-uncased``.
                - a path to a `directory` containing model weights saved using
                  :func:`~BaseModel.save_pretrained`,
                  e.g.: ``./my_model_directory/``.

            model_args: (`optional`) Sequence of positional arguments:
                All remaning positional arguments will be passed to the underlying model's
                ``__init__`` method

            config: (`optional`) instance of a class derived from
                :class:`~BaseConfig`: Configuration for the model to
                use instead of an automatically loaded configuation. Configuration can be
                automatically loaded when:

                - the model is a model provided by the library (loaded with the
                  ``shortcut-name`` string of a pretrained model), or
                - the model was saved using
                  :func:`~BaseModel.save_pretrained` and is reloaded
                  by suppling the save directory.
                - the model is loaded by suppling a local directory as
                  ``pretrained_model_name_or_path`` and a configuration JSON file named
                  `config.json` is found in the directory.

            state_dict: (`optional`) dict:
                an optional state dictionnary for the model to use instead of a state
                dictionary loaded from saved weights file. This option can be used if you
                want to create a model from a pretrained configuration but load your own
                weights. In this case though, you should check if using
                :func:`~BaseModel.save_pretrained` and
                :func:`~BaseModel.from_pretrained` is not a
                simpler option.

            cache_dir: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.

            force_download: (`optional`) boolean, default False:
                Force to (re-)download the model weights and configuration files and override
                the cached versions if they exists.

            resume_download: (`optional`) boolean, default False:
                Do not delete incompletely recieved file. Attempt to resume the download if
                such a file exists.

            output_loading_info: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys,
                unexpected keys and error messages.

            kwargs: (`optional`) Remaining dictionary of keyword arguments:
                Can be used to update the configuration object (after it being loaded) and
                initiate the model. (e.g. ``output_attention=True``). Behave differently
                depending on whether a `config` is provided or automatically loaded:

                - If a configuration is provided with ``config``, ``**kwarg
                  directly passed to the underlying model's ``__init__`` method (we assume
                  all relevant updates to the configuration have already been done)
                - If a configuration is not provided, ``kwargs`` will be first passed to the
                  configuration class initialization function
                  (:func:`~BaseConfig.from_pretrained`). Each key of
                  ``kwargs`` that corresponds to a configuration attribute will be used to
                  override said attribute with the supplied ``kwargs`` value. Remaining keys
                  that do not correspond to any configuration attribute will be passed to the
                  underlying model's ``__init__`` function.

        Examples::

            # Download model and configuration from S3 and cache.
            model = BertModel.from_pretrained('bert-base-uncased')
            # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            model = BertModel.from_pretrained('./test/saved_model/')
            # Update configuration during loading
            model = BertModel.from_pretrained('bert-base-uncased', output_attention=True)
            assert model.config.output_attention == True

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        output_loading_info = kwargs.pop('output_loading_info', False) 
        force_download = kwargs.pop("force_download", False)
        kwargs.pop("resume_download", False)
        pretrained_epoch = kwargs.pop('pretrained_epoch', None)
        # key replacement in state_dict: Dict[str,str], {old_key: new_key}
        state_dict_key_replace = kwargs.pop('state_dict_key_replace', None)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                # force_download=force_download,
                # resume_download=resume_download,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if pretrained_epoch is None:
                archive_file = os.path.join(pretrained_model_name_or_path, '{}.bin'.format(WEIGHTS_NAME))
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, '{}_{}.bin'.format(WEIGHTS_NAME,pretrained_epoch))
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir,
                                                force_download=force_download)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_model_archive_map.keys()),
                        archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        # Convert old format to new format if needed from a PyTorch state_dict
        # Module name replacement from old model to new model
        old_keys = []
        new_keys = []
        state_dict_key_replace = json.loads(state_dict_key_replace) if state_dict_key_replace is not None else {}
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
            # replace old key name with new one
            for k_rep in state_dict_key_replace.keys():
                if k_rep in key:
                    new_key = key.replace(k_rep,state_dict_key_replace[k_rep])
                    if new_key:
                        old_keys.append(key)
                        new_keys.append(new_key)
            
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys,
                unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        if cls.base_model_prefix not in (None, ''):
            if not hasattr(model, cls.base_model_prefix) and \
                    any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                start_prefix = cls.base_model_prefix + '.'
            if hasattr(model, cls.base_model_prefix) and \
                    not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
                model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}..., total {}".format(
                model.__class__.__name__, missing_keys[:10], len(missing_keys)))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}..., total {}".format(
                model.__class__.__name__, unexpected_keys[:10], len(unexpected_keys)))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "error_msgs": error_msgs}
            return model, loading_info

        return model


def prune_linear_layer(layer: torch.nn.Linear, index: torch.LongTensor, dim: int = 0) -> torch.nn.Linear:
    """
    Prune a linear layer to keep only entries in index.
    Used to remove heads.
    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.
    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def find_pruneable_heads_and_indices(
    heads: typing.List[int], n_heads: int, head_size: int, already_pruned_heads: typing.Set[int]
) -> typing.Tuple[typing.Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.
    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.
    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def accuracy(logits, labels, ignore_index: int = -100):
    with torch.no_grad():
        valid_mask = (labels != ignore_index)
        predictions = logits.float().argmax(-1)
        correct = (predictions == labels) * valid_mask
        return correct.sum().float() / valid_mask.sum().float()

def glorot_orthogonal(tensor, scale):
    """Initialize a tensor's values according to an orthogonal Glorot initialization scheme."""
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
            (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


def get_activation_fn(name: str, nnModule: bool = False):
    if name == 'gelu':
        if nnModule:
            return nn.GELU()
        else:
            return gelu
    elif name == 'relu':
        if nnModule:
            return nn.ReLU()
        else:
            return torch.nn.functional.relu
    elif name == 'swish':
        if nnModule:
            return nn.SiLU()
        else:
            return swish
    else:
        raise ValueError(f"Unrecognized activation fn: {name}")


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm  # type: ignore
except (ImportError, AttributeError):
    logger.info("Better speed can be achieved with apex installed from "
                "https://www.github.com/nvidia/apex .")

class LayerNorm(nn.Module):  # type: ignore
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class SimpleMLP(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout),
            weight_norm(nn.Linear(hid_dim, out_dim), dim=None))

    def forward(self, x):
        return self.main(x)


class SimpleConv(nn.Module):

    def __init__(self,
                 in_dim: int,
                 hid_dim: int,
                 out_dim: int,
                 dropout: float = 0.):
        super().__init__()
        self.main = nn.Sequential(
            nn.BatchNorm1d(in_dim),  # Added this
            weight_norm(nn.Conv1d(in_dim, hid_dim, 5, padding=2), dim=None),
            nn.ReLU(),
            nn.Dropout(dropout, inplace=True),
            weight_norm(nn.Conv1d(hid_dim, out_dim, 3, padding=1), dim=None))

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.main(x)
        x = x.transpose(1, 2).contiguous()
        return x


class Accuracy(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, target):
        return accuracy(inputs, target, self.ignore_index)


class PredictionHeadTransform(nn.Module):

    def __init__(self,
                 hidden_size_in: int,
                 hidden_size_out: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12):
        super().__init__()
        self.dense = nn.Linear(hidden_size_in, hidden_size_out)
        if isinstance(hidden_act, str):
            self.transform_act_fn = get_activation_fn(hidden_act)
        else:
            self.transform_act_fn = hidden_act
        self.LayerNorm = LayerNorm(hidden_size_out, eps=layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class MLMHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 use_class_weights: bool = False,
                 ignore_index: int = -1):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.use_class_weights = use_class_weights
        self._ignore_index = ignore_index

    def get_class_weight(self, target_inputs: torch.Tensor, class_num: int):
        total_num = target_inputs.reshape(-1).ge(0).sum().to('cpu').to(torch.float)
        bin_count = torch.bincount(torch.masked_select(target_inputs,target_inputs.ge(0))).to('cpu').to(torch.float)
        bin_miss = class_num - bin_count.size(0)
        if bin_miss > 0:
            bin_count = torch.cat((bin_count,torch.zeros(bin_miss).to('cpu')))
        bin_count = torch.where(bin_count == 0.,total_num,bin_count)
        class_weights = torch.reciprocal(bin_count)
        return class_weights

    def forward(self, hidden_states, targets=None):
        hidden_states_interm = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states_interm) + self.bias
        outputs = (hidden_states,hidden_states_interm,)
        vocab_size = self.bias.data.size()[0]
        if targets is not None:
            if self.use_class_weights:
                aa_class_weight = self.get_class_weight(targets,vocab_size).to(hidden_states)
            else:
                aa_class_weight = None
            loss_fct = nn.CrossEntropyLoss(weight=aa_class_weight,ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            loss_and_metrics = (masked_lm_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_logits_token

class Class1DHead(nn.Module):
    """
    Prediction head for 1D site-wise classification
    """
    def __init__(self,
                 hidden_size: int,
                 class_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 dropout_rate: float = 0.1,
                 ignore_index: int = -1):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, class_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(class_size))  # type: ignore
        self._ignore_index = ignore_index

    def get_class_weight(self, target_inputs: torch.Tensor, class_num: int):
        total_num = target_inputs.reshape(-1).ge(0).sum().to('cpu').to(torch.float)
        bin_count = torch.bincount(torch.masked_select(target_inputs,target_inputs.ge(0))).to('cpu').to(torch.float)
        bin_miss = class_num - bin_count.size(0)
        if bin_miss > 0:
            bin_count = torch.cat((bin_count,torch.zeros(bin_miss).to('cpu')))
        bin_count = torch.where(bin_count == 0.,total_num,bin_count)
        class_weights = torch.reciprocal(bin_count)
        return class_weights

    def forward(self, hidden_states, targets=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = (hidden_states,)
        class_size = self.bias.data.size(0)
        if targets is not None:
            class_weight = self.get_class_weight(targets,class_size).to(hidden_states)
            loss_fct = nn.CrossEntropyLoss(weight=class_weight, ignore_index=self._ignore_index)
            ce_loss = loss_fct(
                hidden_states.reshape(-1, class_size), targets.view(-1))
            metrics = {'ppl': torch.exp(ce_loss)}
            loss_and_metrics = (ce_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_logits_token

class Class2DHead(nn.Module):
    """
    Prediction head for 2D site-wise classification
    """
    def __init__(self,
                 hidden_size: int,
                 class_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 dropout_rate: float = 0.1,
                 ignore_index: int = -1):
        super().__init__()
        
        self.transform = PredictionHeadTransform(hidden_size*2, hidden_size, hidden_act, layer_norm_eps)
        self.dropout = nn.Dropout(dropout_rate, inplace=True)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, class_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(class_size))  # type: ignore
        self._ignore_index = ignore_index

    def get_class_weight(self, target_inputs: torch.Tensor, class_num: int):
        total_num = target_inputs.reshape(-1).ge(0).sum().to('cpu').to(torch.float)
        bin_count = torch.bincount(torch.masked_select(target_inputs,target_inputs.ge(0))).to('cpu').to(torch.float)
        bin_miss = class_num - bin_count.size(0)
        if bin_miss > 0:
            bin_count = torch.cat((bin_count,torch.zeros(bin_miss).to('cpu')))
        bin_count = torch.where(bin_count == 0.,total_num,bin_count)
        class_weights = torch.reciprocal(bin_count)
        return class_weights

    def construct_interaction_tensor(self, feat_1d: torch.Tensor):
        """Construct intra protein pairwise feature tensor

        Args:
            feat_1d: feature tensor generated by bert, size: [batch_size,seq_length_padded,hidden_dim]

        Outputs:
            pairwise feature tensor of size [batch_size,seq_length_padded,seq_length_padded,2*hidden_dim]
        """
        new_feat_1d = feat_1d.permute(0,2,1)
        seq_pad_len = feat_1d.shape[1]
        # interact_tensor size [batch_size,2*hidden_dim,seq_length,seq_length]
        interact_tensor = torch.cat((torch.repeat_interleave(new_feat_1d.unsqueeze(3), repeats=seq_pad_len, dim=3),
                                    torch.repeat_interleave(new_feat_1d.unsqueeze(2), repeats=seq_pad_len, dim=2)), dim=1)
        interact_tensor_t = interact_tensor.transpose(2,3) # sharing its underlying storage with the input tensor (since input is strided tensor)
        triu_idx_i,triu_idx_j =  torch.triu_indices(seq_pad_len, seq_pad_len, 1)
        interact_tensor_t[:,:,triu_idx_i,triu_idx_j] = interact_tensor[:,:,triu_idx_i,triu_idx_j] # interact_tensor is also changed
        interact_tensor.to(feat_1d.device)
        #assert (interact_tensor.transpose(2,3) == interact_tensor).all() == True
        return interact_tensor.permute(0,2,3,1)

    def forward(self, hidden_states, targets=None):
        interact_hidden_states = self.construct_interaction_tensor(hidden_states) # [bs,L,L,2*d]
        interact_hidden_states = self.transform(interact_hidden_states)
        interact_hidden_states = self.dropout(interact_hidden_states)        
        interact_hidden_states = self.decoder(interact_hidden_states) + self.bias
        outputs = (interact_hidden_states,)
        class_size = self.bias.data.size(0)
        if targets is not None:
            class_weight = self.get_class_weight(targets,class_size).to(hidden_states)
            loss_fct = nn.CrossEntropyLoss(weight=class_weight, ignore_index=self._ignore_index)
            ce_loss = loss_fct(
                interact_hidden_states.reshape(-1, class_size), targets.view(-1))
            metrics = {'ppl': torch.exp(ce_loss)}
            loss_and_metrics = (ce_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), prediction_logits_token

class ABSeqConcateHead(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 subClass_dropoutProb: float = 0.1,
                 weight_subClassLoss: float = 0.0,
                 ignore_index: int = -1):
        super().__init__()
        self.transform_token = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        self.transform_subClassHLPair = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        self.dropout_subClassHLPair = nn.Dropout(subClass_dropoutProb)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        # predictor for subclass
        self.pred_subClassHLPair = nn.Linear(hidden_size, len(ab_HL_subclass))
        
        self._ignore_index = ignore_index
        self.weight_subClassLoss = weight_subClassLoss

    def forward(self,
                hidden_states,
                targets, targets_subClassHLPair):
        # token prediction 
        hidden_states_token = self.transform_token(hidden_states)
        token_states = self.decoder(hidden_states_token) + self.bias
        vocab_size = self.bias.data.size()[0]
        print(f'vocab_size:{vocab_size}',flush=True)
        # subClassPair prediction
        hidden_states_subClassHLPair = self.transform_subClassHLPair(hidden_states[:,0])  # input: [d,]
        hidden_states_subClassHLPair = self.dropout_subClassHLPair(hidden_states_subClassHLPair)
        subClassPair_states = self.pred_subClassHLPair(hidden_states_subClassHLPair)
        outputs = (token_states, subClassPair_states,)

        if targets is not None and targets_subClassHLPair is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                token_states.view(-1, vocab_size), targets.view(-1))
            subClassPair_loss = loss_fct(
                subClassPair_states,targets_subClassHLPair)
            metrics = {'perplexity': torch.exp(masked_lm_loss),
                       'perplexity_subC_pair': torch.exp(subClassPair_loss)}
            loss_and_metrics = (masked_lm_loss + self.weight_subClassLoss * subClassPair_loss, metrics) ## **NO SUBCLASS LOSS**
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss, metrics), prediction_logits_token, prediction_logits_subClass_pair


class ABEmbedSeqConcateHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -1):
        super().__init__()
        self.transform_token = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        self.transform_subClassHLPair = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        
        #self.dropout_subClassHLPair = nn.Dropout(subClass_dropoutProb)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        #self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        # predictor for subclass
        #self.pred_subClassHLPair = nn.Linear(hidden_size, len(ab_HL_subclass))
        
        #self._ignore_index = ignore_index

    def forward(self,
                hidden_states):
        # token prediction 
        hidden_states_token = self.transform_token(hidden_states)
        #token_states = self.decoder(hidden_states_token) + self.bias
        # subClassPair prediction
        hidden_states_subClassHLPair = self.transform_subClassHLPair(hidden_states[:,0])  # input: [d,]
        #hidden_states_subClassHLPair = self.dropout_subClassHLPair(hidden_states_subClassHLPair)
        #subClassPair_states = self.pred_subClassHLPair(hidden_states_subClassHLPair)
        outputs = (hidden_states_token, hidden_states_subClassHLPair,)

        return outputs  # (loss, metrics), prediction_logits_token, prediction_logits_subClass_pair


class ABSeqIndivHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 subClass_dropoutProb: float = 0.1,
                 weight_subClassLoss: float = 0.0,
                 ignore_index: int = -1):
        super().__init__()
        self.transform_token = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        self.transform_subClassHLPair = PredictionHeadTransform(hidden_size*2, hidden_size, hidden_act, layer_norm_eps)
        self.dropout_subClassHLPair = nn.Dropout(subClass_dropoutProb)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        # predictor for subclass
        self.pred_subClassHLPair = nn.Linear(hidden_size, len(ab_HL_subclass))
        
        self._ignore_index = ignore_index
        self.weight_subClassLoss = weight_subClassLoss

    def forward(self,
                crossAtt_sequence_output_VH,
                crossAtt_sequence_output_VL, 
                targets_VH, targets_VL, targets_subClassHLPair):
        # token prediction 
        hidden_states_VH = self.transform_token(crossAtt_sequence_output_VH)
        hidden_states_VL = self.transform_token(crossAtt_sequence_output_VL)
        token_states_VH = self.decoder(hidden_states_VH) + self.bias
        token_states_VL = self.decoder(hidden_states_VL) + self.bias
        vocab_size = self.bias.data.size()[0]
        # subClassPair prediction
        hidden_states_subClassHLPair = self.transform_subClassHLPair(torch.cat((crossAtt_sequence_output_VH[:,0],crossAtt_sequence_output_VL[:,0]),dim=1)) # input: [2d,]
        hidden_states_subClassHLPair = self.dropout_subClassHLPair(hidden_states_subClassHLPair)
        subClassPair_states = self.pred_subClassHLPair(hidden_states_subClassHLPair)
        outputs = (token_states_VH, token_states_VL, subClassPair_states,) 
        if targets_VH is not None and targets_VL is not None and targets_subClassHLPair is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss_VH = loss_fct(
                token_states_VH.view(-1, vocab_size), targets_VH.view(-1))
            masked_lm_loss_VL = loss_fct(
                token_states_VL.view(-1, vocab_size), targets_VL.view(-1))
            subClassPair_loss = loss_fct(
                subClassPair_states,targets_subClassHLPair)
            metrics = {'perplexity_VH': torch.exp(masked_lm_loss_VH),
                       'perplexity_VL': torch.exp(masked_lm_loss_VL),
                       'perplexity_subC_pair': torch.exp(subClassPair_loss)}
            loss_and_metrics = (masked_lm_loss_VH + masked_lm_loss_VL + self.weight_subClassLoss*subClassPair_loss, metrics) ## **NO SUBCLASS LOSS**
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss, metrics), prediction_logits_VH, prediction_logits_VL, prediction_logits_subClassPair

class ABEmbedSeqIndivHead(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -1):
        super().__init__()
        self.transform_token = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)
        self.transform_subClassHLPair = PredictionHeadTransform(hidden_size*2, hidden_size, hidden_act, layer_norm_eps)
        #self.dropout_subClassHLPair = nn.Dropout(subClass_dropoutProb)
        
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        #self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        #self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        
        # predictor for subclass
        #self.pred_subClassHLPair = nn.Linear(hidden_size, len(ab_HL_subclass))
        
        #self._ignore_index = ignore_index

    def forward(self,
                crossAtt_sequence_output_VH,
                crossAtt_sequence_output_VL):
        # token prediction 
        hidden_states_VH = self.transform_token(crossAtt_sequence_output_VH)
        hidden_states_VL = self.transform_token(crossAtt_sequence_output_VL)
        
        #token_states_VH = self.decoder(hidden_states_VH) + self.bias
        #token_states_VL = self.decoder(hidden_states_VL) + self.bias
        
        # subClassPair prediction
        hidden_states_subClassHLPair = self.transform_subClassHLPair(torch.cat((crossAtt_sequence_output_VH[:,0],crossAtt_sequence_output_VL[:,0]),dim=1)) # input: [2d,]
        
        #hidden_states_subClassHLPair = self.dropout_subClassHLPair(hidden_states_subClassHLPair)
        #subClassPair_states = self.pred_subClassHLPair(hidden_states_subClassHLPair)

        outputs = (hidden_states_VH, hidden_states_VL, hidden_states_subClassHLPair)
        
        return outputs  #hidden_states_VH, hidden_states_VL, hidden_states_subClassHLPair

class CEntropyRegWeightNorHead(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 lamda_mlm: float,
                 lamda_ce: float,
                 lamda_l1: float,
                 layer_num: int,
                 head_num: int,
                 head_selector: typing.Tuple,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -1):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        # layers for contact prediction (add weight norm)
        self.logit = weight_norm(nn.Linear(np.sum(head_selector), 1, bias=True), dim=None)
        self.head_selector = head_selector
        self.vocab_size = vocab_size
        self.lamda_l1 = lamda_l1
        self.lamda_ce = lamda_ce
        self.lamda_mlm = lamda_mlm
        self._ignore_index = ignore_index

    def symmetrize(self, x):
        "Make layer symmetric in final two dimensions, used for contact prediction."
        return x + x.transpose(-1, -2)

    def apc(self, x):
        "Perform average product correct, used for contact prediction."
        a1 = x.sum(-1, keepdims=True)
        a2 = x.sum(-2, keepdims=True)
        a12 = x.sum((-1, -2), keepdims=True)

        avg = a1 * a2
        avg.div_(a12)  # in-place to reduce memory
        normalized = x - avg
        return normalized

    def forward(self, attentions, hidden_states, targets=None,
                contact_targets=None, valid_mask=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = (hidden_states,)

        # prepare symm attention matrix
        ## attentions: tuple(num_layer), each tensor size - [batch_size, num_head, L_max, L_max]
        attentions = torch.stack(attentions) # [layer_n, bs, head_n, L_max, L_max]
        attentions = attentions.transpose(0,1).contiguous() # [bs, layer_n, head_n, L_max, L_max]
        bs, layer_n, head_n, L_max, _ = attentions.size()
        attentions = attentions.view(bs, layer_n*head_n, L_max, L_max)
        ## select heads
        head_mask = torch.from_numpy(np.asarray(self.head_selector)).view(-1).to('cuda',dtype=torch.bool)
        attentions = attentions[:,head_mask,:,:]
        ## symm + apc
        attentions = self.apc(self.symmetrize(attentions))
        attentions = attentions.permute(0,2,3,1) # [bs,L_max,L_max,layer_n*head_n] channel at last dimen
        contact_logits = self.logit(attentions).squeeze(3) #[bs,L_max,L_max]
        
        # filter out padding, special token, unmodeled residues
        # valid_mask size [bs,L_max]
        val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
        seqpos = torch.arange(valid_mask.shape[1]).to(valid_mask.device)
        x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
        val_all_mask_mat = torch.logical_and(val_mask_mat, (y_ind - x_ind)>=6)
        val_contact_logits = contact_logits[val_all_mask_mat] # [val_pair_num,]
        val_contact_targets = contact_targets[val_all_mask_mat]

        # calculate loss
        if targets is not None and contact_targets is not None:
            # MLM loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            
            # contact prediction BCE loss
            loss_bce = nn.BCEWithLogitsLoss()
            contact_logis_loss = loss_bce(val_contact_logits, val_contact_targets)
            metrics['contact_logis_ce'] = contact_logis_loss

            logis_linear_l1 = torch.tensor(0., requires_grad=True).to(val_contact_targets.device) # exclude bias
            for name, param in self.logit.named_parameters():
                if 'weight' in name:
                    logis_linear_l1 = logis_linear_l1 + torch.norm(param,1)
            metrics['contact_logis_l1'] = logis_linear_l1

            loss_and_metrics = (self.lamda_mlm * masked_lm_loss + 
                                self.lamda_ce * contact_logis_loss + 
                                self.lamda_l1 * logis_linear_l1, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs


class CEntropyRegHead(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 lamda_mlm: float,
                 lamda_ce: float,
                 lamda_l1: float,
                 weight_short: float,
                 weight_medium: float,
                 weight_long: float,
                 gamma: float,
                 weight_s_con: float,
                 weight_s_non: float,
                 weight_m_con: float,
                 weight_m_non: float,
                 weight_l_con: float,
                 weight_l_non: float,
                 layer_num: int,
                 head_num: int,
                 head_selector: typing.Tuple,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -1):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        # layers for contact prediction
        self.logit = nn.Linear(np.sum(head_selector), 1, bias=True)
        self.head_selector = head_selector
        self.vocab_size = vocab_size
        self.lamda_l1 = lamda_l1
        self.lamda_ce = lamda_ce
        self.lamda_mlm = lamda_mlm
        self.weight_short = weight_short
        self.weight_medium = weight_medium
        self.weight_long = weight_long
        self.gamma = gamma
        self.weight_s_con = weight_s_con
        self.weight_s_non = weight_s_non
        self.weight_m_con = weight_m_con
        self.weight_m_non = weight_m_non
        self.weight_l_con = weight_l_con
        self.weight_l_non = weight_l_non
        self._ignore_index = ignore_index

    def symmetrize(self, x):
        "Make layer symmetric in final two dimensions, used for contact prediction."
        return x + x.transpose(-1, -2)

    def apc(self, x):
        "Perform average product correct, used for contact prediction."
        a1 = x.sum(-1, keepdims=True)
        a2 = x.sum(-2, keepdims=True)
        a12 = x.sum((-1, -2), keepdims=True)

        avg = a1 * a2
        avg.div_(a12)  # in-place to reduce memory
        normalized = x - avg
        return normalized

    def forward(self, attentions, hidden_states, targets=None,
                contact_targets=None, valid_mask=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = (hidden_states,)

        # prepare symm attention matrix
        ## attentions: tuple(num_layer), each tensor size - [batch_size, num_head, L_max, L_max]
        attentions = torch.stack(attentions) # [layer_n, bs, head_n, L_max, L_max]
        attentions = attentions.transpose(0,1).contiguous() # [bs, layer_n, head_n, L_max, L_max]
        bs, layer_n, head_n, L_max, _ = attentions.size()
        attentions = attentions.view(bs, layer_n*head_n, L_max, L_max)
        ## select heads
        head_mask = torch.from_numpy(np.asarray(self.head_selector)).view(-1).to('cuda',dtype=torch.bool)
        attentions = attentions[:,head_mask,:,:]
        ## symm + apc
        attentions = self.apc(self.symmetrize(attentions))
        attentions = attentions.permute(0,2,3,1) # [bs,L_max,L_max,layer_n*head_n] channel at last dimen
        contact_logits = self.logit(attentions).squeeze(3) #[bs,L_max,L_max]
        
        # filter out padding, special token, unmodeled residues
        # valid_mask size [bs,L_max]
        val_mask_mat = valid_mask[:,:,None] & valid_mask[:,None, :] # size [bs,L_max,L_max]
        seqpos = torch.arange(valid_mask.shape[1]).to(valid_mask.device)
        x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
        val_all_mask_mat = torch.logical_and(val_mask_mat, (y_ind - x_ind)>=6)
        val_contact_logits = contact_logits[val_all_mask_mat] # [val_pair_num,], logits for contact class
        val_contact_targets = contact_targets[val_all_mask_mat]

        # prepare weights for focal loss
        omega_weights = torch.ones_like(contact_logits) #[bs,L_max,L_max]
        omega_weights.masked_fill_(torch.logical_and(val_mask_mat,
                                                     torch.logical_and((y_ind - x_ind)>=6,(y_ind - x_ind)<12)),
                                   self.weight_short)
        omega_weights.masked_fill_(torch.logical_and(val_mask_mat,
                                                     torch.logical_and((y_ind - x_ind)>=12, (y_ind - x_ind)<24)),
                                   self.weight_medium)
        omega_weights.masked_fill_(torch.logical_and(val_mask_mat, (y_ind - x_ind)>=24), self.weight_long)
        omega_weights = omega_weights[val_all_mask_mat] # [val_pair_nums,]

        alpha_weights = torch.ones_like(contact_logits) # [bs,L_max,L_max]
        alpha_weights.masked_fill_(torch.logical_and(contact_targets == 1.0,
                                                     torch.logical_and((y_ind - x_ind)>=6,(y_ind - x_ind)<12)),
                                   self.weight_s_con)
        alpha_weights.masked_fill_(torch.logical_and(contact_targets == 0.0,
                                                     torch.logical_and((y_ind - x_ind)>=6,(y_ind - x_ind)<12)),
                                   self.weight_s_non)
        alpha_weights.masked_fill_(torch.logical_and(contact_targets == 1.0,
                                                     torch.logical_and((y_ind - x_ind)>=12,(y_ind - x_ind)<24)),
                                   self.weight_m_con)
        alpha_weights.masked_fill_(torch.logical_and(contact_targets == 0.0,
                                                     torch.logical_and((y_ind - x_ind)>=12,(y_ind - x_ind)<24)),
                                   self.weight_m_non)
        alpha_weights.masked_fill_(torch.logical_and(contact_targets == 1.0, (y_ind - x_ind)>=24), self.weight_l_con)
        alpha_weights.masked_fill_(torch.logical_and(contact_targets == 0.0, (y_ind - x_ind)>=24), self.weight_l_non)

        alpha_weights = alpha_weights[val_all_mask_mat] # [val_pair_nums,]
        
        if self.gamma == 0.0:
          focal_factor = 1.0
        else:
          focal_factor = torch.exp(-self.gamma*contact_targets*contact_logits - self.gamma*torch.log(1 + torch.exp(-1.0*contact_logits))) #[bs,L_max,L_max]
          focal_factor = focal_factor[val_all_mask_mat] # [val_pair_nums,]
      
        # calculate loss
        if targets is not None and contact_targets is not None:
            # MLM loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            
            # contact prediction BCE loss
            loss_bce = nn.BCEWithLogitsLoss(reduction='none') #[val_pair_nums,]
            contact_logis_loss = torch.mean(omega_weights * alpha_weights * focal_factor * loss_bce(val_contact_logits, val_contact_targets))
            metrics['contact_logis_ce'] = contact_logis_loss

            logis_linear_l1 = torch.tensor(0., requires_grad=True).to(val_contact_targets.device) # exclude bias
            for name, param in self.logit.named_parameters():
                if 'weight' in name:
                    logis_linear_l1 = logis_linear_l1 + torch.norm(param,1)
            metrics['contact_logis_l1'] = logis_linear_l1

            loss_and_metrics = (self.lamda_mlm * masked_lm_loss + 
                                self.lamda_ce * contact_logis_loss + 
                                self.lamda_l1 * logis_linear_l1, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs


    

class NonContactRegHead(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 head_selector: typing.Tuple,
                 lamda_nonCon: float,
                 lamda_mlm: float,
                 weight_short: float,
                 weight_medium: float,
                 weight_long: float,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -1):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index
        self.head_selector = head_selector
        self.lamda_nonCon = lamda_nonCon
        self.lamda_mlm = lamda_mlm
        self.weight_short = weight_short
        self.weight_medium = weight_medium
        self.weight_long = weight_long

    def forward(self, attentions, hidden_states,
                targets=None,
                contact_targets=None,
                valid_mask_nc=None,
                valid_mask=None):
       
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = (hidden_states,)
        
        
        if targets is not None and contact_targets is not None:
            # process valid_mask_nc
            # False at: unmodelled pairs
            valid_mask_mat = valid_mask_nc[:,:,None] & valid_mask_nc[:,None,:] # size [bs,L_max,L_max]
            
            # prepare weights for balancing pairs in short, medium, long
            # unmodelled, start/end token, pading: 1
            weight_balance = torch.ones_like(contact_targets,device=contact_targets.device) # [bs,L_max,L_max]
            valid_mask_mat_struct = valid_mask[:,:,None] & valid_mask[:,None,:] # [bs,L_max,L_max]
            seqpos = torch.arange(valid_mask.size(1),device=valid_mask.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            weight_balance.masked_fill_(torch.logical_and(valid_mask_mat_struct,
                                                          torch.logical_and(torch.abs(y_ind-x_ind)>=6,torch.abs(y_ind-x_ind)<12)),
                                        self.weight_short)
            weight_balance.masked_fill_(torch.logical_and(valid_mask_mat_struct,
                                                          torch.logical_and(torch.abs(y_ind-x_ind)>=12,torch.abs(y_ind-x_ind)<24)),
                                        self.weight_medium)
            weight_balance.masked_fill_(torch.logical_and(valid_mask_mat_struct,torch.abs(y_ind-x_ind)>=24), self.weight_long)
            
            # MLM loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            # attention regularization loss
            # contact_targets: [batch_size, L_max, L_max]
            # attentions: tuple(num_layer tensors), each tensor size - [batch_size, num_head, L_max, L_max]
            atten_reg_loss_sum = 0.
            atten_reg_loss_nor_sum = 0. 
            for layer_sele, atten_layer in zip(self.head_selector, attentions):
                layer_sele = torch.Tensor(layer_sele).to('cuda')
                allOne_tensor = torch.ones_like(contact_targets).to('cuda')
                # pair_sele size - [batch_size, num_head, L_max, L_max]
                pair_sele = ((allOne_tensor - contact_targets) * valid_mask_mat * weight_balance).unsqueeze(1).expand(-1,atten_layer.size(1),-1,-1)
                # after F-norm, size in sum [batch_size, num_head]
                atten_reg_loss_sum += torch.sum((layer_sele.unsqueeze(0).expand(atten_layer.size(0),-1) * torch.norm(pair_sele * atten_layer, p=2, dim=(-2,-1)))**2)
                # normalize by whole attention Fnorm
                atten_reg_loss_nor_sum += torch.sum(torch.div((layer_sele.unsqueeze(0).expand(atten_layer.size(0),-1)
                  * torch.norm(pair_sele * atten_layer, p=2, dim=(-2,-1)))**2, torch.norm(atten_layer, p=2, dim=(-2,-1))**2))
            
            atten_reg_loss = atten_reg_loss_sum / (len(attentions) * attentions[0].size(0) * attentions[0].size(1))
            metrics['nonCon_att_Fnorm2'] = atten_reg_loss
            # other mode of loss
            atten_reg_loss_local = atten_reg_loss_sum / (np.sum(self.head_selector) * attentions[0].size(0))
            metrics['nonCon_att_Fnorm2_local'] = atten_reg_loss_local
            atten_reg_loss_local_nor = atten_reg_loss_nor_sum / (np.sum(self.head_selector) * attentions[0].size(0))
            metrics['nonCon_att_Fnorm2_local_nor'] = atten_reg_loss_local_nor
                     
            loss_and_metrics = (self.lamda_mlm * masked_lm_loss + self.lamda_nonCon * atten_reg_loss_local_nor, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss_and_metric), prediction_scores

class ContactRegHead(nn.Module):
    def __init__(self,
                 hidden_size: int,
                 vocab_size: int,
                 head_selector: typing.Tuple,
                 lamda_con: float,
                 lamda_mlm: float,
                 weight_short: float,
                 weight_medium: float,
                 weight_long: float,
                 hidden_act: typing.Union[str, typing.Callable] = 'gelu',
                 layer_norm_eps: float = 1e-12,
                 ignore_index: int = -1):
        super().__init__()
        self.transform = PredictionHeadTransform(hidden_size, hidden_size, hidden_act, layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = nn.Parameter(data=torch.zeros(vocab_size))  # type: ignore
        self.vocab_size = vocab_size
        self._ignore_index = ignore_index
        self.head_selector = head_selector
        self.lamda_con = lamda_con
        self.lamda_mlm = lamda_mlm
        self.weight_short = weight_short
        self.weight_medium = weight_medium
        self.weight_long = weight_long

    def forward(self, attentions, hidden_states,
                targets=None,
                contact_targets=None,
                valid_mask=None):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        outputs = (hidden_states,)
        if targets is not None and contact_targets is not None:
            # prepare weights for balancing pairs in short, medium, long
            # unmodelled, start/end token, pading: 1
            weight_balance = torch.ones_like(contact_targets,device=contact_targets.device) # [bs,L_max,L_max]
            valid_mask_mat_struct = valid_mask[:,:,None] & valid_mask[:,None,:] # [bs,L_max,L_max]
            seqpos = torch.arange(valid_mask.size(1),device=valid_mask.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            weight_balance.masked_fill_(torch.logical_and(valid_mask_mat_struct,
                                                          torch.logical_and(torch.abs(y_ind-x_ind)>=6,torch.abs(y_ind-x_ind)<12)),
                                        self.weight_short)
            weight_balance.masked_fill_(torch.logical_and(valid_mask_mat_struct,
                                                          torch.logical_and(torch.abs(y_ind-x_ind)>=12,torch.abs(y_ind-x_ind)<24)),
                                        self.weight_medium)
            weight_balance.masked_fill_(torch.logical_and(valid_mask_mat_struct,torch.abs(y_ind-x_ind)>=24), self.weight_long)

            # MLM loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            masked_lm_loss = loss_fct(
                hidden_states.view(-1, self.vocab_size), targets.view(-1))
            metrics = {'perplexity': torch.exp(masked_lm_loss)}
            # attention regularization loss
            # contact_targets: [batch_size, L_max, L_max]
            # attentions: tuple(num_layer tensors), each tensor size - [batch_size, num_head, L_max, L_max]
            atten_reg_loss_sum = 0.
            atten_reg_loss_nor_sum = 0.
            for layer_sele, atten_layer in zip(self.head_selector, attentions):
                layer_sele = torch.Tensor(layer_sele).to('cuda')
                # pair_sele size - [batch_size, num_head, L_max, L_max]
                pair_sele = (contact_targets * weight_balance).unsqueeze(1).expand(-1,atten_layer.size(1),-1,-1)
                # after F-norm, size [batch_size, num_head]
                atten_reg_loss_sum += torch.sum((layer_sele.unsqueeze(0).expand(atten_layer.size(0),-1) * torch.norm(pair_sele * atten_layer, p=2, dim=(-2,-1)))**2)
                # normalize by whole attention Fnorm
                atten_reg_loss_nor_sum += torch.sum(torch.div((layer_sele.unsqueeze(0).expand(atten_layer.size(0),-1)
                  * torch.norm(pair_sele * atten_layer, p=2, dim=(-2,-1)))**2, torch.norm(atten_layer, p=2, dim=(-2,-1))**2))
            
            atten_reg_loss = atten_reg_loss_sum / (len(attentions) * attentions[0].size(0) * attentions[0].size(1))
            metrics['con_att_Fnorm2'] = atten_reg_loss
            # other mode of loss
            atten_reg_loss_local = atten_reg_loss_sum / (np.sum(self.head_selector) * attentions[0].size(0))
            metrics['con_att_Fnorm2_local'] = atten_reg_loss_local

            atten_reg_loss_local_nor = atten_reg_loss_nor_sum / (np.sum(self.head_selector) * attentions[0].size(0))
            metrics['con_att_Fnorm2_local_nor'] = atten_reg_loss_local_nor
                     
            loss_and_metrics = (self.lamda_mlm * masked_lm_loss - self.lamda_con * atten_reg_loss_local_nor, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss_and_metric), prediction_scores


class ValuePredictionHead(nn.Module):
    def __init__(self, hidden_size: int, middle_size: int = 128, dropout: float = 0.2):
        super().__init__()
        self.value_prediction = SimpleMLP(hidden_size, middle_size, 1, dropout)

    def forward(self, pooled_output, targets=None):
        value_pred = self.value_prediction(pooled_output)
        outputs = (value_pred,)

        if targets is not None:
            loss_fct = nn.MSELoss()
            value_pred_loss = loss_fct(value_pred, targets)
            metrics={'mse':value_pred_loss}
            loss_and_metrics = (value_pred_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), value_prediction


class SequenceClassificationHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classify = SimpleMLP(hidden_size, 512, num_labels)

    def forward(self, pooled_output, targets=None):
        logits = self.classify(pooled_output)
        outputs = (logits,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss()
            classification_loss = loss_fct(logits, targets)
            metrics = {'accuracy': accuracy(logits, targets)}
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs  # (loss), logits


class SequenceToSequenceClassificationHead(nn.Module):

    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 ignore_index: int = -100):
        super().__init__()
        self.classify = SimpleConv(
            hidden_size, 512, num_labels)
        self.num_labels = num_labels
        self._ignore_index = ignore_index

    def forward(self, sequence_output, targets=None):
        sequence_logits = self.classify(sequence_output)
        outputs = (sequence_logits,)
        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            classification_loss = loss_fct(
                sequence_logits.view(-1, self.num_labels), targets.view(-1))
            acc_fct = Accuracy(ignore_index=self._ignore_index)
            metrics = {'accuracy':
                       acc_fct(sequence_logits.view(-1, self.num_labels), targets.view(-1))}
            loss_and_metrics = (classification_loss, metrics)
            outputs = (loss_and_metrics,) + outputs
        return outputs  # (loss), sequence_logits


class PairwiseContactPredictionHead(nn.Module):

    def __init__(self, hidden_size: int, ignore_index=-100):
        super().__init__()
        self.predict = nn.Sequential(
            nn.Dropout(), nn.Linear(2 * hidden_size, 2))
        self._ignore_index = ignore_index

    def forward(self, inputs, sequence_lengths, targets=None):
        prod = inputs[:, :, None, :] * inputs[:, None, :, :]
        diff = inputs[:, :, None, :] - inputs[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        prediction = self.predict(pairwise_features)
        prediction = (prediction + prediction.transpose(1, 2)) / 2
        prediction = prediction[:, 1:-1, 1:-1].contiguous()  # remove start/stop tokens
        outputs = (prediction,)

        if targets is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=self._ignore_index)
            contact_loss = loss_fct(
                prediction.view(-1, 2), targets.view(-1))
            metrics = {'precision_at_l5':
                       self.compute_precision_at_l5(sequence_lengths, prediction, targets)}
            loss_and_metrics = (contact_loss, metrics)
            outputs = (loss_and_metrics,) + outputs

        return outputs

    def compute_precision_at_l5(self, sequence_lengths, prediction, labels):
        with torch.no_grad():
            valid_mask = labels != self._ignore_index
            seqpos = torch.arange(valid_mask.size(1), device=sequence_lengths.device)
            x_ind, y_ind = torch.meshgrid(seqpos, seqpos)
            valid_mask &= ((y_ind - x_ind) >= 6).unsqueeze(0)
            probs = F.softmax(prediction, 3)[:, :, :, 1]
            valid_mask = valid_mask.type_as(probs)
            correct = 0
            total = 0
            for length, prob, label, mask in zip(sequence_lengths, probs, labels, valid_mask):
                masked_prob = (prob * mask).view(-1)
                most_likely = masked_prob.topk(length // 5, sorted=False)
                selected = label.view(-1).gather(0, most_likely.indices)
                correct += selected.sum().float()
                total += selected.numel()
            return correct / total
