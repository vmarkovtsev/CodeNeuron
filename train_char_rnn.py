import argparse
from collections import defaultdict
import gzip
import logging
import sys
from typing import List, Tuple, Dict

import humanize
import numpy

from chars import CHARS, WEIGHTS, OOV_WEIGHT


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Path to the train dataset.")
    parser.add_argument("-l", "--layers", default="512,256",
                        help="Layers configuration: number of neurons on each layer separated by "
                             "comma.")
    parser.add_argument("-m", "--length", type=int, default=100, help="RNN sequence length.")
    parser.add_argument("-b", "--batch-size", type=int, default=100, help="Batch size.")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="Number of epochs.")
    parser.add_argument("-t", "--type", default="GRU", choices=("GRU", "LSTM"),
                        help="Recurrent layer type to use.")
    parser.add_argument("-v", "--validation", type=float, default=0.2,
                        help="Fraction of the dataset to use for validation.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the resulting Tensorflow graph.")
    parser.add_argument("--optimizer", default="RMSprop", choices=("RMSprop", "Adam"),
                        help="Optimizer to apply.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout ratio.")
    parser.add_argument("--lr", default=0.01, type=float, help="Learning rate.")
    parser.add_argument("--tensorboard", default="tb_logs",
                        help="TensorBoard output logs directory.")
    logging.basicConfig(level=logging.INFO)
    return parser.parse_args()


def read_dataset(path: str, analyze_chars: bool) -> Tuple[List[str], Dict[str, List[int]]]:
    log = logging.getLogger("reader")
    texts = []
    bufsize = 1 << 20
    buffer = bytearray(bufsize)
    bufpos = 0
    read_buffer = bytearray(bufsize)
    chars = defaultdict(list)

    def append_text(end_index: int):
        text = buffer[:end_index].decode("utf-8")
        if analyze_chars:
            index = len(texts)
            for c in sorted(set(text)):
                chars[c].append(index)
        texts.append(text)

    with gzip.open(path) as gzf:
        size = gzf.readinto(read_buffer)
        while size > 0:
            sys.stderr.write("%d texts\r" % len(texts))
            rpos = 0
            while rpos < size:
                border = read_buffer.find(b"\x04", rpos)
                if border == -1:
                    border = size
                delta = border - rpos
                if bufpos + delta > bufsize:
                    raise OverflowError(
                        "%d %d %d %d" % (bufpos, delta, bufpos + delta, bufsize))
                buffer[bufpos:bufpos + delta] = read_buffer[rpos:border]
                if border < size:
                    append_text(bufpos + delta)
                    bufpos = 0
                else:
                    bufpos += delta
                rpos = border + 1
            size = gzf.readinto(read_buffer)
        if bufpos > 0:
            append_text(bufpos)
    log.info("%d texts, avg len %d, %d distinct chars, total chars %d",
             len(texts), numpy.mean([len(t) for t in texts]), len(chars),
             sum(len(t) for t in texts))
    return texts, chars


def create_char_rnn_model(args: argparse.Namespace):
    # late import prevent from loading Tensorflow too soon
    from keras import layers, models, initializers, optimizers
    log = logging.getLogger("model")

    def add_rnn():
        input = layers.Input(batch_shape=(args.batch_size, args.length), dtype="uint8")
        log.info("Added %s", input)
        embedding = layers.Embedding(
            200, 200, embeddings_initializer=initializers.Identity(), trainable=False)(input)
        log.info("Added %s", embedding)
        layer = embedding
        layer_sizes = [int(n) for n in args.layers.split(",")]
        for i, nn in enumerate(layer_sizes):
            layer = getattr(layers, args.type)(
                nn, return_sequences=(i < len(layer_sizes) - 1), implementation=2,
                dropout=args.dropout)(layer)
            log.info("Added %s", layer)
        return input, layer

    forward_input, forward_output = add_rnn()
    reverse_input, reverse_output = add_rnn()
    merged = layers.Concatenate()([forward_output, reverse_output])
    log.info("Added %s", merged)
    decision = layers.Dense(len(CHARS) + 1, activation="softmax")(merged)
    log.info("Added %s", decision)
    model = models.Model(inputs=[forward_input, reverse_input], outputs=[decision])
    optimizer = getattr(optimizers, args.optimizer)(lr=args.lr)
    log.info("Compiling...")
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    log.info("Done")
    return model


def train_char_rnn_model(model, dataset: List[str], args: argparse.Namespace):
    from keras import callbacks, utils

    if args.length % 2 != 0:
        raise ValueError("--length must be even")
    log = logging.getLogger("train")
    tensorboard = callbacks.TensorBoard(log_dir=args.tensorboard, histogram_freq=1)

    numpy.random.seed(7)
    log.info("Splitting into validation and train...")
    valid_doc_indices = set(numpy.random.choice(
        numpy.arange(len(dataset)), int(len(dataset) * args.validation), replace=False))
    train_doc_indices = set(range(len(dataset))) - valid_doc_indices
    train_docs = [dataset[i] for i in sorted(train_doc_indices)]
    valid_docs = [dataset[i] for i in sorted(valid_doc_indices)]

    # we cannot reach first l / 2 (forward) and last l / 2 (backward)
    # this is still a little bit imprecise - we count \x02 and \x03 which mark the code boundaries
    valid_size = sum(len(text) - args.length for text in valid_docs)
    train_size = sum(len(text) - args.length for text in train_docs)
    log.info("train samples: %d\tvalidation samples: %d\t~%.3f",
             train_size, valid_size, valid_size / (valid_size + train_size))

    class Feeder(utils.Sequence):
        def __init__(self, texts: List[str]):
            self.texts = texts
            pos = 0
            self.index = index = numpy.zeros(len(texts) + 1, dtype=numpy.int32)
            for i, text in enumerate(texts):
                pos += len(text) - text.count("\x02") - text.count("\x03") - args.length
                index[i + 1] = pos
            # this takes much memory but is the best we can do.
            self.batches = numpy.arange(pos, dtype=numpy.uint32)
            log.info("Batches occupied %s", humanize.naturalsize(self.batches.size))
            numpy.random.shuffle(self.batches)

        def __len__(self):
            return self.index[-1] // args.batch_size

        def __getitem__(self, item):
            centers = self.batches[item * args.batch_size:(item + 1) * args.batch_size]
            batch = ([numpy.zeros((args.batch_size, args.length), dtype=numpy.uint8)
                      for _ in range(2)],
                     [numpy.zeros((args.batch_size, len(CHARS) + 1), dtype=numpy.float32)])
            text_indices = numpy.searchsorted(self.index, centers) - 1
            for bi, (center, text_index) in enumerate(zip(centers, text_indices)):
                text = self.texts[text_index]
                target = center - self.index[text_index]
                for x, c in enumerate(text[args.length // 2:]):
                    if c != "\x02" and c != "\x03":
                        target -= 1
                        if target == 0:
                            break
                text_i = x - 1
                batch_i = args.batch_size - 1
                while text_i >= 0 and batch_i >= 0:
                    c = text[text_i]
                    text_i -= 1
                    if c == "\x02" or c == "\x03":
                        continue
                    batch[0][0][bi][batch_i] = CHARS.get(c, len(CHARS))
                    batch_i -= 1
                text_i = x + 1
                batch_i = args.batch_size - 1
                while text_i < len(text) and batch_i >= 0:
                    c = text[text_i]
                    text_i += 1
                    if c == "\x02" or c == "\x03":
                        continue
                    batch[0][1][bi][batch_i] = CHARS.get(c, len(CHARS))
                    batch_i -= 1
                batch[1][0][bi][CHARS.get(text[x], len(CHARS))] = 1
            return batch

    log.info("Creating the training feeder")
    train_feeder = Feeder(train_docs)
    log.info("Creating the validation feeder")
    valid_feeder = Feeder(valid_docs)
    log.info("model.fit_generator")
    model.fit_generator(generator=train_feeder,
                        validation_data=valid_feeder,
                        validation_steps=len(valid_feeder),
                        steps_per_epoch=len(train_feeder),
                        epochs=args.epochs,
                        class_weight=[v for (c, v) in sorted(WEIGHTS.items())] + [OOV_WEIGHT],
                        callbacks=[tensorboard],
                        use_multiprocessing=True)


def main():
    args = setup()
    dataset, _ = read_dataset(args.input, False)
    model_char = create_char_rnn_model(args)
    train_char_rnn_model(model_char, dataset, args)

if __name__ == "__main__":
    sys.exit(main())
