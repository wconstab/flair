from typing import List
import argparse
import numpy as np
import pickle
import random
import torch

import flair.datasets
from flair.data import Corpus, Sentence
from flair.embeddings import (
    TokenEmbeddings,
    WordEmbeddings,
    StackedEmbeddings,
    FlairEmbeddings,
    CharacterEmbeddings,
)
from flair.training_utils import EvaluationMetric
from flair.visual.training_curves import Plotter

parser = argparse.ArgumentParser()
parser.add_argument('--debug', metavar='fn', default="", help="Dump outputs into file")
parser.add_argument('--seed', default=1234)
parser.add_argument('--script', action='store_true', help="Whether to torchscript the model")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# 1. get the corpus
corpus: Corpus = flair.datasets.UD_ENGLISH().downsample(0.01)
print(corpus)

# 2. what tag do we want to predict?
tag_type = "upos"

# 3. make the tag dictionary from the corpus
tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
print(tag_dictionary.idx2item)

# initialize embeddings
embedding_types: List[TokenEmbeddings] = [
    # WordEmbeddings("glove"),
    # comment in this line to use character embeddings
    CharacterEmbeddings(),
    # comment in these lines to use contextual string embeddings
    #
    # FlairEmbeddings('news-forward'),
    #
    # FlairEmbeddings('news-backward'),
]

embeddings: StackedEmbeddings = StackedEmbeddings(embeddings=embedding_types)

# initialize sequence tagger
from flair.models import SequenceTagger

tagger: SequenceTagger = SequenceTagger(
    hidden_size=256,
    embeddings=embeddings,
    tag_dictionary=tag_dictionary,
    tag_type=tag_type,
    use_crf=True,
)
if args.script:
    tagger = torch.jit.script(tagger)

# initialize trainer
from flair.trainers import ModelTrainer

trainer: ModelTrainer = ModelTrainer(tagger, corpus)

trainer.train(
    "resources/taggers/example-ner",
    learning_rate=0.1,
    mini_batch_size=32,
    max_epochs=3,
    shuffle=False,
)

if args.debug:
    sentence = Sentence('I love PyTorch!')
    res, out = tagger.evaluate(sentence)
    torch.save(out, args.debug)

# plotter = Plotter()
# plotter.plot_training_curves("resources/taggers/example-ner/loss.tsv")
# plotter.plot_weights("resources/taggers/example-ner/weights.txt")
