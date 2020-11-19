## Modelled after assembleCoaddTask
import lsst.pex.config as pexConfig
import lsst.daf.persistence as dafPersist
import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.utils as utils
from lsst.pipe.tasks.coaddBase import CoaddBaseTask, makeSkyInfo

class MetadetectConfig(pexConfig.Config):
    """ Configuration parameters for the `MetadetectTask`.
    """

    doWrite = pexConfig.Field(
    doc="Persist coadd?",
    dtype=bool,
    default=False,
)

class MetadetectConnections(pipeBase.PipelineTaskConnections,
                            dimensions=("tract", "patch", "band", "skymap"),
                            defaultTemplates={"inputCoaddName": "deep",
                                              "outputCoaddName": "deep",
                                              "warpType": "direct",
                                              }):
    inputWarps = pipeBase.connectionTypes.Input(
        doc=("Input list of warps to be stacked"),
        name="{inputCoaddName}Coadd_{warpType}Warp",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "visit", "instrument"),
        deferLoad=True,
        multiple=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc="Input definition of geometry/box and projection/wcs for coadded exposures",
        name="{inputCoaddName}Coadd_skyMap",
        storageClass="skyMap",
        dimensions=("skyMap",),
    )

    def __init__(self, *, config=None):
        super().__init__(config=config)

        templateValues = {name: getattr(config.connections, name) for name in self.defaultTemplates}
        templateValues['warpType'] = config.warpType

class MetadetectTask(pipeBase.PipelineTask, pipeBase.CmdLineTask):
    """ Metadetection.
    """

    ConfigClass = MetadetectConfig
    _DefaultName = "metadetect"

    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @utils.inheritDoc(pipeBase.PipelineTask)
    def runQuantum(self, butlerQC,
                   inputRefs,
                   outputRefs):
        """Perform metadection from a set of calexps.

        PipelineTask (Gen3) entry point to Metadetect a set of Calexps.
        Analogous to `runDataRef` (Gen2), it prepares all the data products to be
        passed to `run`, and processes the results before returning a struct
        of results to be written out. Metadetect cannot fit all Calexps in memory.
        Therefore, its inputs are accessed subregion by subregion
        by the Gen3 `DeferredDatasetHandle` that is analagous to the Gen2
        `lsst.daf.persistence.ButlerDataRef`. Any updates to this method should
        correspond to an update in `runDataRef` while both entry points
        are used.
        """
        inputData = butlerQC.get(inputRefs)

        # Construct skyInfo expected by run
        # Do not remove skyMap from inputData in case makeSupplementaryDataGen3 needs it
        skyMap = inputData["skyMap"]
        outputDataId = butlerQC.quantum.dataId

        inputData["skyInfo"] = makeSkyInfo(skyMap,
                                           tractId=outputDataId["tract"],
                                           patchId=outputDataId["patch"])

        # Construct a list of input Deferred Datasets
        # These quack a bit like Gen2 DataRefs
        refList = inputData['calexp']  # assembleCoaddTask

        refList = [ref.dataId for ref in inputRefs.calExpList]  # makeCoaddTempExp

        # Construct a list of packed integer IDs expected by `run`.
        ccdIdLIst = [refId.pack("visit_detector") for refId in refIdList]

        # Extract integer visitId requested by `run`
        visits = [refId['visit'] for refId in refIdList]
        assert(all(visits[0] == visit for visit in visits))
        visitId = visits[0]


        # Perform some middle steps as `runDataRef` does
        inputs = self.prepareInputs(refList)  # assembleCoaddTask
        self.prepareCalibratedExposures(**inputData)
        results = self.run(**inputData, visitId=visitId, ccdIdList=ccdIdList, refIdList=refIdList,
                           skyInfo=skyInfo)
        #butlerQC.put(?)  # put what?

    @pipeBase.timeMethod
    def runDataRef(self,
                   dataRef,
                   selectionDataList=None,
                   warpRefList=None):
        """ Pipebase.CmdLineTask entry point to coadd a set of warps/calexps
        """
        pass

    def prepareInputs(self, refList):
        """Prepare the calexps for metadetection by measuring the weight for each calexp and the scaling for
        the photometric zero point.

        Each calexp has its own photometric zeropoint and background variance. Before metadetecting these
        Calexps together, compute a scale factor to normalize the photometric zeropoint and compute the weight
        for each Calexp.

        Parameters
        ----------
        refList: `list`
            List of data references to tempExp

        Returns
        -------
        result: `lsst.pipe.base.Struct`
            Result struct with components:

            - ``temExpRefList``: `list` of data references to tempExp.
            - ``weightList``: `list` of weightings.
            - ``imageScalerList``: `list` of image scalers.
        """

        for tempExpRef in refList:
            tempExp = tempExpRef.get(datasetType=tempExpName, immediate=True)
        return pipeBase.Struct(tempExpRefList=tempExpRefList, weightList=weightList,
                               imageScalerList=imageScalerList)


    def prepareCalibratedExposures(self, calExpList, backgroundList=None, skyCorrList=None):
        """Calibrate and add backgrounds to input calExpList in place
        TODO DM-17062: apply jointcal/meas_mosaic here
        Parameters
        ----------
        calExpList : `list` of `lsst.afw.image.Exposure`
            Sequence of calexps to be modified in place
        backgroundList : `list` of `lsst.afw.math.backgroundList`
            Sequence of backgrounds to be added back in if bgSubtracted=False
        skyCorrList : `list` of `lsst.afw.math.backgroundList`
            Sequence of background corrections to be subtracted if doApplySkyCorr=True
        """
        backgroundList = len(calExpList)*[None] if backgroundList is None else backgroundList
        skyCorrList = len(calExpList)*[None] if skyCorrList is None else skyCorrList
        for calexp, background, skyCorr in zip(calExpList, backgroundList, skyCorrList):
            mi = calexp.maskedImage
            if not self.config.bgSubtracted:
                mi += background.getImage()
            if self.config.doApplySkyCorr:
                mi -= skyCorr.getImage()

    @pipeBase.timeMethod
    def run(self, calExpList, ccdIdList, skyInfo, visitId=0, refIdList=None):
        pass