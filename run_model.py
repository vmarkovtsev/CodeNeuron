import argparse
import logging
import os
import sys

import numpy

from chars import CHARS


# RNNs require the context, so the beginning and the ending of each phrase cannot be processed.
# We prepend and append some silly text to overcome that.
header = footer = "Meditation. I Am Happy, I Am Good, I Am Happy, I Am Good."


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", required=True,
                        help="Path to the trained model in Tensorflow GraphDef format.")
    parser.add_argument("--disable-filtering", action="store_true",
                        help="Disable anti-false-positives postprocessing.")
    logging.basicConfig(level=logging.ERROR)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return parser.parse_args()


def main():
    args = setup()
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.ERROR)
    graph_def = tf.GraphDef()
    with open(args.model, "rb") as fin:
        graph_def.ParseFromString(fin.read())
    tf.import_graph_def(graph_def, name="")
    del graph_def
    graph = tf.get_default_graph()
    input1 = graph.get_operation_by_name("input_1_1").outputs[0]
    input2 = graph.get_operation_by_name("input_2_1").outputs[0]
    output = graph.get_operation_by_name("output").inputs[0]
    batch_size, length = (d.value for d in input1.shape)
    text = sys.stdin.read()
    text = header + text + footer
    batches = [[], []]
    for x in range(length // 2, len(text) - length // 2 - 1):
        before = text[x - length // 2:x]
        after = text[x + 1:x + 1 + length // 2]
        before_arr = numpy.zeros(length, dtype=numpy.uint8)
        after_arr = numpy.zeros_like(before_arr)
        for i, c in enumerate(reversed(before)):
            before_arr[length - 1 - i] = CHARS.get(c, len(CHARS))
        for i, c in enumerate(reversed(after)):
            after_arr[length - 1 - i] = CHARS.get(c, len(CHARS))
        batches[0].append(before_arr)
        batches[1].append(after_arr)
    inputs_size = len(batches[0])
    if inputs_size == 0:
        return
    target_size = int(numpy.ceil(inputs_size / batch_size)) * batch_size
    while len(batches[0]) < target_size:
        batches[0].append(numpy.zeros(length, dtype=numpy.uint8))
        batches[1].append(numpy.zeros(length, dtype=numpy.uint8))
    batches[0] = numpy.array(batches[0])
    batches[1] = numpy.array(batches[1])
    result = []
    for batch1, batch2 in zip(numpy.split(batches[0], target_size // batch_size),
                              numpy.split(batches[1], target_size // batch_size)):
        with tf.Session() as session:
            probs = session.run(output, {input1: batch1, input2: batch2})
            result.extend(numpy.argmax(probs, axis=-1))
    result = result[len(header):inputs_size - len(footer)]
    text = text[len(header):-len(footer)]
    if not args.disable_filtering:
        # simple heuristics to reduce false positives
        # no opening without a closing
        code_mode = False
        for i, r in enumerate(reversed(result)):
            if r == 1:
                if code_mode:
                    result[len(result) - 1 - i] = 2
                code_mode = True
            elif r == 0:
                if not code_mode:
                    result[len(result) - 1 - i] = 2
                code_mode = False
        # no closing without an opening
        code_mode = False
        for i, r in enumerate(result):
            if r == 0:
                if code_mode:
                    result[i] = 2
                code_mode = True
            elif r == 1:
                if not code_mode:
                    result[i] = 2
                code_mode = False
    print()
    sys.stdout.write(text[:length // 2])
    for x, r in zip(text[length // 2: len(text) - length // 2 - 1], result):
        if r == 2:
            sys.stdout.write(x)
            continue
        if r == 0:
            sys.stdout.write("<code>" + x)
        else:
            sys.stdout.write("</code>" + x)
    print(text[len(text) - length // 2 - 1:])


if __name__ == "__main__":
    sys.exit(main())
