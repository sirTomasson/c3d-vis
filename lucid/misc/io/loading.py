# Copyright 2018 The Lucid Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Methods for loading arbitrary data from arbitrary sources.

This module takes a URL, infers its underlying data type and how to locate it,
loads the data into memory and returns a convenient representation.

This should support for example PNG images, JSON files, npy files, etc.
"""

from __future__ import absolute_import, division, print_function

import os
import json
import logging
import numpy as np
import PIL.Image
import tensorflow.compat.v1 as tf
from google.protobuf.message import DecodeError

from lucid.misc.io.reading import read_handle
from lucid import modelzoo


# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


def _load_npy(handle, **kwargs):
    """Load npy file as numpy array."""
    return np.load(handle, **kwargs)


def _load_img(handle, target_dtype=np.float32, size=None, **kwargs):
    """Load image file as numpy array."""

    image_pil = PIL.Image.open(handle, **kwargs)

    # resize the image to the requested size, if one was specified
    if size is not None:
        if len(size) > 2:
            size = size[:2]
            log.warning("`_load_img()` received size: {}, trimming to first two dims!".format(size))
        image_pil = image_pil.resize(size, resample=PIL.Image.LANCZOS)

    image_array = np.asarray(image_pil)

    # remove alpha channel if it contains no information
    # if image_array.shape[-1] > 3 and 'A' not in image_pil.mode:
    # image_array = image_array[..., :-1]

    image_dtype = image_array.dtype
    image_max_value = np.iinfo(image_dtype).max  # ...for uint8 that's 255, etc.

    # using np.divide should avoid an extra copy compared to doing division first
    ndimage = np.divide(image_array, image_max_value, dtype=target_dtype)

    rank = len(ndimage.shape)
    if rank == 3:
        return ndimage
    elif rank == 2:
        return np.repeat(np.expand_dims(ndimage, axis=2), 3, axis=2)
    else:
        message = "Loaded image has more dimensions than expected: {}".format(rank)
        raise NotImplementedError(message)


def _load_json(handle, **kwargs):
    """Load json file as python object."""
    return json.load(handle, **kwargs)


def _load_text(handle, split=False, encoding="utf-8"):
    """Load and decode a string."""
    string = handle.read().decode(encoding)
    return string.splitlines() if split else string


def _load_graphdef_protobuf(handle, **kwargs):
    """Load GraphDef from a binary proto file."""
    # as_graph_def
    graph_def = tf.GraphDef.FromString(handle.read())

    # check if this is a lucid-saved model
    # metadata = modelzoo.util.extract_metadata(graph_def)
    # if metadata is not None:
    #   url = handle.name
    #   return modelzoo.vision_base.Model.load_from_metadata(url, metadata)

    # else return a normal graph_def
    return graph_def


loaders = {
    ".png": _load_img,
    ".jpg": _load_img,
    ".jpeg": _load_img,
    ".npy": _load_npy,
    ".npz": _load_npy,
    ".json": _load_json,
    ".txt": _load_text,
    ".md": _load_text,
    ".pb": _load_graphdef_protobuf,
}


def load(url_or_handle, cache=None, **kwargs):
    """Load a file.

    File format is inferred from url. File retrieval strategy is inferred from
    URL. Returned object type is inferred from url extension.

    Args:
      url_or_handle: a (reachable) URL, or an already open file handle

    Raises:
      RuntimeError: If file extension or URL is not supported.
    """

    ext = get_extension(url_or_handle)
    try:
        loader = loaders[ext.lower()]
        message = "Using inferred loader '%s' due to passed file extension '%s'."
        log.debug(message, loader.__name__[6:], ext)
        return load_using_loader(url_or_handle, loader, cache, **kwargs)

    except KeyError:

        log.warning("Unknown extension '%s', attempting to load as image.", ext)
        try:
            with read_handle(url_or_handle, cache=cache) as handle:
                result = _load_img(handle)
        except Exception as e:
            message = "Could not load resource %s as image. Supported extensions: %s"
            log.error(message, url_or_handle, list(loaders))
            raise RuntimeError(message.format(url_or_handle, list(loaders)))
        else:
            log.info("Unknown extension '%s' successfully loaded as image.", ext)
            return result


# Helpers

def load_using_loader(url_or_handle, loader, cache, **kwargs):
    if is_handle(url_or_handle):
        result = loader(url_or_handle, **kwargs)
    else:
        url = url_or_handle
        try:
            with read_handle(url, cache=cache) as handle:
                result = loader(handle, **kwargs)
        except (DecodeError, ValueError):
            log.warning("While loading '%s' an error occurred. Purging cache once and trying again; if this fails we will raise an Exception!", url)
            # since this may have been cached, it's our responsibility to try again once
            # since we use a handle here, the next DecodeError should propagate upwards
            with read_handle(url, cache='purge') as handle:
                result = load_using_loader(handle, loader, cache, **kwargs)
    return result


def is_handle(url_or_handle):
    return hasattr(url_or_handle, "read") and hasattr(url_or_handle, "name")


def get_extension(url_or_handle):
    if is_handle(url_or_handle):
        _, ext = os.path.splitext(url_or_handle.name)
    else:
        _, ext = os.path.splitext(url_or_handle)
    if not ext:
        raise RuntimeError("No extension in URL: " + url_or_handle)
    return ext
