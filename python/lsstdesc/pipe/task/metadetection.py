# import lsst.daf.persistence as dafPersist
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT


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
    catalog = cT.Output(
        doc=("Output catalog"),
        name='metadetectObj',
        storageClass="DataFrame",
        dimensions=("tract", "patch", "skymap"),
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
    def run(self, calExpList, skyMap):
        print(len(calExpList))
        print(skyMap)
        import pandas as pd
        df = pd.DataFrame([idx for idx, _  in enumerate(calExpList)])
        return pipeBase.Struct(catalog=df)

    def runQuantum(self, butlerQC: pipeBase.ButlerQuantumContext,
                         inputRefs: pipeBase.InputQuantizedConnection,
                         outputRefs: pipeBase.OutputQuantizedConnection):

        inputs = butlerQC.get(inputRefs)
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

