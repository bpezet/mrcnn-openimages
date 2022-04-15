#!/bin/bash

my_script="~/objet_detetction/mrcnn-openimages/script.sh"

oar_job_id=`oarsub $my_script | grep "OAR_JOB_ID" | cut -d '=' -f2`

oar_stdout_file="OAR.$oar_job_id.stdout"

until oarstat -s -j $oar_job_id | grep Running ; do
    echo "Waiting for the job to start..."
    sleep 1
done

echo "Job $oar_job_id is started !"
