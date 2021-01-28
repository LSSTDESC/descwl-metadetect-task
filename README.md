# descwl-metadetect-task
DM Stack task to run metadetect

## How to run the task?

Setup the project directory

`setup -j -r descwl-metadetect-task`

Export the Butler repository for convenience

`export REPO=/project/hsc/gen3repo/rc2v21_0_0_rc1_ssw48`

Run the task using the example command:

`pipetask run -t lsstdesc.pipe.task.metadetection.MetadetectTask -b $REPO --input HSC/runs/RC2/v21_0_0_rc1 --output u/$USER/mdetTest -d "skymap='hsc_rings_v1' AND tract=9615 AND patch=45" --register-dataset-types`

Note:

    1. The `--input HSC/runs/RC2/v21_0_0_rc1` should be excluded from the second time onwards.

    2. The `--register-dataset-types` is needed only the first time to register a new data product. It doesn't hurt to keep it around, except that you might get multiple data products if you made a typo in naming them.

To check if the run is successful:

`butler query-datasets $REPO --collections u/$USER/* metadetectObj`

The data products will be stored in sub-folder within `$REPO/u/$USER/mdetTest` with a timestamp.
