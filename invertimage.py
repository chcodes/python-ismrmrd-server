
import ismrmrd
import os
import itertools
import logging
import numpy as np

def process(connection, config, params):
    logging.info("Processing connection.")
    logging.info("Config: \n%s", config.decode("utf-8"))
    logging.info("Params: \n%s", params.decode("utf-8"))

    try:
        for origImage in connection:
        #     logging.info("Processing image...\n%s", origImage)
            logging.info("Processing image...")

            newImage = process_image(origImage, config, params)

        #     logging.info("Sending image to client:\n%s", newImage)
            logging.info("Sending image to client:")
            connection.send_image(newImage)
    finally:
        logging.info("InvertImage in Finally")
        connection.send_close()

def process_image(image, config, params):
    logging.info("!!! process_image() called")
    # Folder for debug output files
    debugFolder = "/tmp/share/dependency"

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.info("Created folder " + debugFolder + " for debug output files")

    # Extract image data itself
    data = image.data
    logging.info("Original image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "imgOrig.npy", data)

    # Normalize and convert to int16
    data = data.astype(np.float64)
    data *= 32768/data.max()
    data = np.around(data)
    data = data.astype(np.int16)

    # Invert image contrast
    data = 32768-data
    data = np.abs(data)
    np.save(debugFolder + "/" + "imgInverted.npy", data)

    # Update the MRD variable with new data
    image.data.ravel()[:] = data.ravel()

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
