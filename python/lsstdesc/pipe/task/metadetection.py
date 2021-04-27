# import lsst.daf.persistence as dafPersist  # Gen2 version
from __future__ import annotations
import typing

import lsst.pipe.base as pipeBase
import lsst.pipe.base.connectionTypes as cT
import lsst.afw.table as afwTable
from lsst.pipe.tasks.coaddBase import makeSkyInfo
import lsst.utils
import lsst.geom as geom
from descwl_coadd.coadd import MultiBandCoaddsDM

BANDS = ['g', 'r', 'i', 'z']


class MetadetectConnections(
    pipeBase.PipelineTaskConnections,
    dimensions=("tract", "patch", "skymap"),
    # running all bands
    # not having instrument makes it possible to
    # combine
    # calexp
):
    calExpList = cT.Input(
        doc=(
            "Input exposures to be resampled and optionally PSF-matched "
            "onto a SkyMap projection/patch",
        ),
        name="calexp",
        storageClass="ExposureF",
        dimensions=("instrument", "visit", "detector"),
        multiple=True,
        deferLoad=True,
    )
    skyMap = pipeBase.connectionTypes.Input(
        doc=(
            "Input definition of geometry/box and "
            "projection/wcs for coadded exposures"
        ),
        name="skyMap",
        storageClass="SkyMap",
        dimensions=("skymap",),
    )
    # This is an output catalog that is not (yet) used in this skeleton
    catalog = cT.Output(
        doc=("Output catalog"),
        name='metadetectObjV1',
        storageClass="DataFrame",
        dimensions=("tract", "patch", "skymap"),
    )
    coadd = cT.Output(
        doc=("Coadded image"),
        name="coaddsInCellsV1",
        storageClass="ExposureF",
        dimensions=("tract", "patch", "skymap", "band", "instrument")
    )
    # Come back and apply jointcal/fgcm later to the calexp s.


class MetadetectConfig(pipeBase.PipelineTaskConfig,
                       pipelineConnections=MetadetectConnections):
    """ Configuration parameters for the `MetadetectTask`.
    """

    pass


class MetadetectTask(pipeBase.PipelineTask):
    """ Metadetection.
    """

    ConfigClass = MetadetectConfig
    _DefaultName = "metadetect"

    # @pipeBase.timeMethod
    def run(self,
            calExpList: typing.List[lsst.afw.image.ExposureF],
            skyInfo: pipeBase.Struct) -> pipeBase.Struct:
        # import pdb

        print(len(calExpList))  # checking if we got something here

        # We need to explicitly get the images since we deferred loading.
        # The line below is just an example illustrating this.
        # We should preferably get them sequentially instead of loading all.
        # calExpList = [calexp.get() for calexp in calExpList[:10]]

        # for calexp in calExpList[0:10]:
        #     # import numpy as np
        #     # import esutil as eu
        #     import pdb
        #     print('band:', calexp.dataId['band'])
        #     calexp = calexp.get()
        #     # print(calexp.variance.array)
        #     # m = calexp.mask.array
        #     # v = calexp.variance.array
        #     # w = np.where(m == 0)
        #     #
        #     # eu.stat.print_stats(v.ravel())
        #     # eu.stat.print_stats(v[w].ravel())
        #     #
        #     pdb.set_trace()
        #
        # from IPython import embed; embed()
        # pdb.set_trace()

        # Erin Sheldon has to fill in the interfaces here and
        # replace calExpList[0] with the coadd.
        # coaddedImage = calExpList[0].get()

        data = make_inputs(
            explist=calExpList,
            skyInfo=skyInfo,
            num_to_keep=3,
        )

        mbc = MultiBandCoaddsDM(
            interp_bright=True,  # make configurable
            data=data['band_data'],
            coadd_wcs=data['coadd_wcs'],
            coadd_bbox=data['coadd_bbox'],
            psf_dims=data['psf_dims'],
            byband=False,
            # show=send_show,
            # loglevel=loglevel,
        )
        coadd_obs = mbc.coadds['all']

        # Create an empty catalogue with minimal schema
        schema = afwTable.SourceTable.makeMinimalSchema()
        table = afwTable.SourceTable.make(schema)
        cat = afwTable.SourceCatalog(table)
        return pipeBase.Struct(coadd=coadd_obs.coadd_exp, catalog=cat)

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
        skyInfo = makeSkyInfo(
            skyMap, tractId=quantumDataId["tract"],
            patchId=quantumDataId["patch"],
        )

        # Run the warp and coaddition code
        outputs = self.run(inputs["calExpList"], skyInfo=skyInfo)

        # Persist the results via the butler
        butlerQC.put(outputs.coadd, outputRefs.coadd)


def make_inputs(explist, skyInfo, num_to_keep=None):
    """
    TODO noise is just same as exp
    """

    band_data = {}
    for band in BANDS:
        blist = []
        for exp in explist:
            tband = exp.dataId['band']
            if tband == band:
                blist.append({'exp': exp})

        if len(blist) > 0:
            band_data[band] = blist

    if len(band_data) == 0:
        raise ValueError('no data found')

    if num_to_keep is not None:
        for band in band_data:
            # offset for the test dataset in which the early ones are not
            # overlapping
            ntot = len(band_data[band])
            mid = ntot // 2
            band_data[band] = band_data[band][mid:mid+num_to_keep]
            # band_data[band] = band_data[band][:num_to_keep]

    # copy data form disk
    for band in band_data:
        for i in range(len(band_data[band])):
            band_data[band][i]['exp'] = band_data[band][i]['exp'].get()

            # make noise exp here
            band_data[band][i]['noise_exp'] = band_data[band][i]['exp']

    # TODO set BRIGHT bit here for bright stars

    # base psf size on last exp
    band = list(band_data.keys())[0]
    psf = band_data[band][0]['exp'].getPsf()
    pos = geom.Point2D(x=100, y=100)
    psfim = psf.computeImage(pos)

    psf_dims = psfim.array.shape
    psf_dims = (max(psf_dims), ) * 2

    return {
        'band_data': band_data,
        'coadd_wcs': skyInfo.wcs,
        'coadd_bbox': skyInfo.bbox,
        'psf_dims': psf_dims,
    }
