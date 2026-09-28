#!/usr/bin/env python3
"""Developer tool: build tiny TFLite models for the Android instrumented tests.

Each model mimics a single-class YOLO head on a 64x64 input (84 anchors) and outputs one
box at anchor 0 — centre (0.5, 0.5), size (0.5, 0.25), normalized — whose score is the
mean of the RED input channel. Feeding a red vs. a blue frame therefore checks input
layout (NCHW/NHWC), RGB channel order, 0..1 normalisation, letterboxing and decoding on
a real device, without a trained detector.

    python training/make_android_test_models.py --out app/src/androidTest/assets
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

ANCHORS = 8 * 8 + 4 * 4 + 2 * 2  # strides 8/16/32 on 64x64


def build(nchw: bool) -> bytes:
    shape = [1, 3, 64, 64] if nchw else [1, 64, 64, 3]

    class Head(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape, tf.float32)])
        def __call__(self, x):
            red = x[:, 0, :, :] if nchw else x[:, :, :, 0]
            score = tf.reshape(tf.reduce_mean(red), [1])
            column = tf.concat([tf.constant([0.5, 0.5, 0.5, 0.25]), score], axis=0)
            out = tf.concat([tf.reshape(column, [5, 1]), tf.zeros([5, ANCHORS - 1])], axis=1)
            return tf.reshape(out, [1, 5, ANCHORS])

    m = Head()
    conv = tf.lite.TFLiteConverter.from_concrete_functions([m.__call__.get_concrete_function()], m)
    return conv.convert()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for nchw, name in ((True, "tiny_red_nchw.tflite"), (False, "tiny_red_nhwc.tflite")):
        data = build(nchw)
        (args.out / name).write_bytes(data)
        print(f"wrote {args.out / name} ({len(data)} bytes)")


if __name__ == "__main__":
    main()
