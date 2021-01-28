#!/bin/bash

##Gen2
#export COSMOS_I=1228^1230^1232^1238^1240^1242^1244^1246^1248^19658^19660^19662^19680^19682^19684^19694^19696^19698^19708^19710^19712^30482^30484^30486^30488^30490^30492^30494^30496^30498^30500^30502^30504
#metadetectTask.py /datasets/hsc/repo/ --calib /datasets/hsc/repo/CALIB/ --rerun RC/w_2020_42/DM-27244:private/$USER/RC2/w_2020_42/ --id tract=9813 patch=4,4 filter=HSC-I --no-versions

#Gen3
export REPO=/project/hsc/gen3repo/rc2v21_0_0_rc1_ssw48
setup -j -r .
pipetask run -t lsstdesc.pipe.task.metadetection.MetadetectTask -b $REPO --input HSC/runs/RC2/v21_0_0_rc1 --output u/$USER/mdetTest -d "skymap='hsc_rings_v1' AND tract=9615 AND patch=45" --register-dataset-types
butler query-datasets $REPO --collections u/$USER/* metadetectObj
