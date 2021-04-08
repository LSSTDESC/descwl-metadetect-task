# import lsst.daf.persistence as dafPersist  # Gen2 version
from __future__ import annotations
import typing

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
from lsst.pipe.base.struct import Struct
import lsst.afw.table as afwTable
from lsst.pipe.tasks.coaddBase import makeSkyInfo
import lsst.utils


class MetadetectConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("tract", "patch", "skymap"),
                            # running all bands
                            # not having instrument makes it possible to combine
                            # calexp
                           ):
    calExpList = cT.Input(
        doc="Input exposures to be resampled and optionally PSF-matched onto a SkyMap projection/patch",
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/box and projection/wcs for coadded exposures",
        name="skyMap",
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    # This is an output catalog that is not (yet) used in this skeleton
    catalog = cT.Output(
        doc=("Output catalog"),
        name='metadetectObj',
        storageClass="DataFrame",
        dimensions=("tract", "patch", "skymap"),
    )
    coadd = cT.Output(
        doc=("Coadded image"),
        name="ngmixCoadd",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "instrument")
    )
    # Come back and apply jointcal/fgcm later to the calexp s.


class MetadetectConfig(pipeBase.PipelineTaskConfig, pipelineConnections=MetadetectConnections):
    """ Configuration parameters for the `MetadetectTask`.
    """

    pass


class MetadetectTask(pipeBase.PipelineTask):
    """ Metadetection.
    """

    ConfigClass = MetadetectConfig
    _DefaultName = "metadetect"

    # @pipeBase.timeMethod
    def run(self, calExpList: typing.List[lsst.afw.image.ExposureF], skyInfo: pipeBase.Struct) -> pipeBase.Struct:

        print(len(calExpList))  # checking if we got something here

        # We need to explicitly get the images since we deferred loading.
        # The line below is just an example illustrating this.
        # We should preferably get them sequentially instead of loading all.
        calExpList = [calexp.get() for calexp in calExpList[:10]]

        # The destination WCS and BBox can be accessed from skyInfo
        coaddWcs = skyInfo.wcs
        coaddBBox = skyInfo.bbox

        # Erin Sheldon has to fill in the interfaces here and
        # replace calExpList[0] with the coadd.
        coaddedImage = calExpList[0]

        # Create an empty catalogue with minimal schema
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        cat = afwTable.SourceCatalog(table)
        return pipeBase.Struct(coadd=coaddedImage, catalog=cat)

    # @lsst.utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC: pipeBase.ButlerQuantumContext,
                   inputRefs: pipeBase.InputQuantizedConnection,
                   outputRefs: pipeBase.OutputQuantizedConnection):
        """Construct warps and then coadds

        Notes
        -----

        PipelineTask (Gen3) entry point to warp. This method is analogous to
        `runDataRef`. See lsst.pipe.tasks.makeCoaddTempExp.py for comparison.
        """
        # Read in the inputs via the butler
        inputs = butlerQC.get(inputRefs)

        # Process the skyMap to WCS and other useful sky information
        skyMap = inputs.pop("skyMap")  # skyInfo below will contain this skyMap
        quantumDataId = butlerQC.quantum.dataId
        skyInfo = makeSkyInfo(skyMap, tractId=quantumDataId["tract"], patchId=quantumDataId["patch"])

        # Run the warp and coaddition code
        outputs = self.run(inputs["calExpList"], skyInfo=skyInfo)

        # Persist the results via the butler
        butlerQC.put(outputs.coadd, outputRefs.coadd)

