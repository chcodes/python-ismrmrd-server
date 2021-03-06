
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft

import fireflow as ff


def groups(iterable, predicate):
    group = []
    for item in iterable:
        group.append(item)

        if predicate(item):
            yield group
            group = []


def conditionalGroups(iterable, predicateAccept, predicateFinish):
    group = []
    try:
        for item in iterable:
            if predicateAccept(item):
                group.append(item)

            if predicateFinish(item):
                yield group
                group = []
    finally:
        logging.info("Received StopIteration")
        iterable.send_close()


def process(connection, config, params):
    logging.info("Processing connection.")
    logging.info("Config: \n%s", config.decode("utf-8"))
    logging.info("Params: \n%s", params.decode("utf-8"))

    # for group in groups(connection, lambda acq: acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)):
    for group in conditionalGroups(connection, lambda acq: not acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA), lambda acq: acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE)):
        image = process_group(group, config, params)

        logging.info("Sending image to client:\n%s", image)
        connection.send_image(image)


def process_group(group, config, params):

    # Folder for debug output files
    debugFolder = "/tmp/share/dependency"

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.info("Created folder " + debugFolder + " for debug output files")

    # Format data into single [cha RO PE] array
    data = [acquisition.data for acquisition in group]
    data = np.stack(data, axis=-1)

    logging.info("Raw data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "raw.npy", data)

    # Image reconstruction with pc-mri
    data = ff.phasecontrast2d(data)
    # data = ff.phasecontrast2d_cnn(data)

    logging.info("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

    # Normalize and convert to int16
    data *= 32768/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Format as ISMRMRD image data
    image = ismrmrd.Image.from_array(data, acquisition=group[0])
    image.image_index = 1

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'GADGETRON_DataRole': 'Image',
                         'GADGETRON_ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'GADGETRON_WindowCenter': '16384',
                         'GADGETRON_WindowWidth': '32768'})
    xml = meta.serialize()
    logging.info("XML: %s", xml)
    logging.info("Image data has %d elements", image.data.size)

    image.attribute_string = xml
    return image
